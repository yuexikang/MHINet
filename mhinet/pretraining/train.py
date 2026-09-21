"""Standalone shared-descriptor trainer. One epoch; exact mid-epoch resume."""
import argparse
import fcntl
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from mhinet.config import RuntimePaths, sha256_file
from mhinet.engine.checkpointing import save_checkpoint, load_checkpoint, capture_rng_state, restore_rng_state
from mhinet.engine.ghim_losses import ghim_loss
from mhinet.ops.geometry import safe_project_points
from mhinet.pretraining.data import SharedPairDataset
from mhinet.pretraining.loss import descriptor_loss
from mhinet.pretraining.model import build_shared_network, export_shared
from mhinet.pretraining.evaluation import retrieval_metrics
from mhinet.pretraining.overlap_metrics import overlap_projection_error, summarize_overlap


def losses(model, batch, device, *, queries, seed):
    tensors = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
    outputs = model(tensors['images'])
    ghim = ghim_loss(outputs['stage1'], tensors['H_gt_norm'], tensors['mask_A_overlap'])
    desc, metrics = descriptor_loss(outputs['pyramid'], tensors['H_gt_norm'],
        tensors['mask_A_overlap'], tensors['mask_B_overlap'], queries=queries, seed=seed)
    loss = ghim['total'] + desc
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Non-finite loss: {batch["pair_id"]}')
    return loss, outputs, dict(loss=float(loss.detach()), ghim={k: float(v.detach()) for k, v in ghim.items()},
                               descriptor=float(desc.detach()), scales=metrics)


@torch.no_grad()
def validate(model, dataset, device, output, step, workers, queries):
    model.eval()
    folder = output / 'validation' / f'step_{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    failed = 0
    start = time.perf_counter()
    for i, batch in enumerate(tqdm(DataLoader(dataset, batch_size=1, num_workers=workers), desc=f'val {step}')):
        _, result, info = losses(model, batch, device, queries=queries, seed=i)
        c = 1 - 1/784
        corners = torch.tensor([[[-c, -c], [c, -c], [c, c], [-c, c]]], device=device)
        pred, valid, _ = safe_project_points(result['H0_norm'], corners)
        truth, gt_valid, _ = safe_project_points(batch['H_gt_norm'].to(device), corners)
        good = bool(valid.all() and gt_valid.all() and result['stage1_valid'].all())
        error = float(((pred - truth) * 392).norm(dim=-1).mean()) if good else None
        if good:
            errors.append(error)
        else:
            failed += 1
        retrieval = retrieval_metrics(result['pyramid'], batch['H_gt_norm'].to(device),
            [batch['mask_A_overlap'].to(device), batch['mask_B_overlap'].to(device)], seed=i,
            images=batch['images'], folder=folder / batch['pair_id'][0] if i < 3 else None)
        overlap = overlap_projection_error(result['H0_norm'],batch['H_gt_norm'].to(device),
            batch['mask_A_overlap'].to(device),batch['mask_B_overlap'].to(device),
            fit_valid=bool(result['stage1_valid'][0]))
        rows.append(dict(pair_id=batch['pair_id'][0], H0_mace_px=error,
                         H0_overlap=overlap, retrieval=retrieval, **info))
        if i < 3:
            from mhinet.visualization.visualization import write_iteration_overlays
            write_iteration_overlays(batch['images'][0], batch['H_gt_norm'][0], [], [],
                folder / batch['pair_id'][0], pair_id=batch['pair_id'][0], h0=result['H0_norm'][0].cpu(),
                ghim_valid=bool(result['stage1_valid'][0]))
    with (folder / 'pairs.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    summary = dict(step=step, pairs=len(rows), failed_H0=failed, H0_failure_rate=failed / max(1,len(rows)),
        H0_mace_px=float(np.mean(errors)) if errors else None,
        loss=float(np.mean([r['loss'] for r in rows])),
        descriptor=float(np.mean([r['descriptor'] for r in rows])),
        elapsed_seconds=time.perf_counter()-start,
        note='scales: sampled InfoNCE candidates; retrieval: full feature-grid candidates, 32 valid queries/direction/scale, errors in 784-input pixels.')
    summary['retrieval'] = {}
    summary['H0_overlap'] = summarize_overlap([row['H0_overlap'] for row in rows])
    for key in rows[0]['retrieval']:
        values = np.asarray([e for row in rows for e in row['retrieval'][key]['errors_input_px']])
        summary['retrieval'][key] = dict(queries=len(values),
            mean_input_px=float(values.mean()) if len(values) else None,
            **{f'recall_{t}px':float((values<=t).mean()) if len(values) else None for t in (1,3,5)})
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    model.train()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-steps', type=int, help='Diagnostic cap; does not change cosine horizon')
    p.add_argument('--limit-train', type=int)
    p.add_argument('--limit-val', type=int)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if (config['accumulation'] < 1 or config['queries'] < 2 or config['save_every'] < 1
        or args.workers < 0 or any(x is not None and x < 1
                                 for x in (args.max_steps,args.limit_train,args.limit_val))):
        raise ValueError('Invalid training/diagnostic limits')
    seed = config['seed']
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # The shared loss uses deterministic gather-based bilinear sampling, so
    # strict mode can also request deterministic Flash Attention backward.
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    runtime = replace(RuntimePaths.from_json(config['runtime']), device='cuda:0')
    tier = config.get('tier',1)
    if tier not in (1,2,3):
        raise ValueError('Unsupported dataset tier')
    train = SharedPairDataset(runtime.data_root / 'train/pairs.jsonl', tier=tier, max_pairs=args.limit_train)
    val = SharedPairDataset(runtime.data_root / 'val/pairs.jsonl', tier=tier, max_pairs=args.limit_val)
    if not len(train) or not len(val):
        raise ValueError('Empty train or val split')
    if ({x.geo_group for x in train.index} & {x.geo_group for x in val.index}
        or {x.parent_group for x in train.index} & {x.parent_group for x in val.index}):
        raise ValueError('Train/val group leakage')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / '.training.lock').open('a')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f'Another trainer owns {output}') from error
    accumulation = config['accumulation']
    total_steps = math.ceil(len(train) / accumulation)
    end = min(total_steps, args.max_steps or total_steps)
    metadata = dict(task='shared_descriptor_v1', config=config,
        implementation={name: sha256_file(Path(__file__).parent / name)
                        for name in ('model.py','loss.py','train.py','evaluation.py','data.py','overlap_metrics.py')},
        train_sha256=sha256_file(train.manifest), val_sha256=sha256_file(val.manifest),
        dino_sha256=sha256_file(runtime.dino_checkpoint),
        pyramid_sha256=sha256_file(runtime.pyramid_checkpoint),
        ghim_sha256=sha256_file(runtime.selected_stage1_checkpoint),
        train_pairs=len(train), val_pairs=len(val), total_steps=total_steps)
    initialization = config.get('initialize_checkpoint')
    if initialization:
        metadata['initialize_checkpoint_sha256'] = sha256_file(initialization)
    model, provenance = build_shared_network(runtime, lora=config['lora'])
    if initialization and not (output/'latest.pt').exists():
        initial = torch.load(initialization,map_location='cpu',weights_only=True,mmap=True)
        if initial.get('metadata',{}).get('task') != 'shared_descriptor_v1':
            raise ValueError('Initialization must be a shared-descriptor checkpoint')
        if initial['metadata']['config']['lora'] != config['lora']:
            raise ValueError('Initialization LoRA topology mismatch')
        if initial['metadata']['dino_sha256'] != metadata['dino_sha256']:
            raise ValueError('Initialization DINO provenance mismatch')
        model.load_state_dict(initial['model'],strict=True)
        provenance['initialized_from'] = str(initialization)
        provenance['source_optimizer_step'] = initial['progress']['optimizer_step']
        del initial
    groups = model.optimizer_groups()
    lr_scale = config.get('learning_rate_scale',1.)
    if not 0 < lr_scale <= 1:
        raise ValueError('learning_rate_scale must be in (0,1]')
    for group in groups:
        group['lr'] *= lr_scale
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    warmup = max(1, round(total_steps * .05))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        fraction = min(1., (step - warmup) / max(1, total_steps - warmup))
        return .1 + .9 * .5 * (1 + math.cos(math.pi * fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    checkpoint_path = output / 'latest.pt'
    cursor, completed = 0, 0
    if checkpoint_path.exists():
        state = load_checkpoint(checkpoint_path, model=model, optimizer=optimizer,
                                scheduler=scheduler, map_location='cpu', expected_metadata=metadata)
        cursor, completed = state['data_progress']['cursor'], state['optimizer_step']
    else:
        # No checkpoint means a fresh run. Archive only files owned by this trainer
        # so old metrics/visualizations cannot masquerade as the new experiment.
        owned = [output / name for name in ('train.jsonl','run.json','validation','shared_descriptor.pt')
                 if (output / name).exists()]
        if owned:
            archive = output / f'previous_no_checkpoint_{time.time_ns()}'
            archive.mkdir()
            for path in owned:
                shutil.move(str(path), str(archive / path.name))
        (output / 'train.jsonl').write_text('')
    (output / 'run.json').write_text(json.dumps(dict(**metadata, provenance=provenance,
        torch=str(torch.__version__), output=str(output)), indent=2, default=str))
    print(f'Tier={tier}; train={len(train)}; val={len(val)}; steps={total_steps}; '
          f'LoRA={config["lora"]}; initialization={initialization}; '
          f'peak_lrs={[(g["name"],g["lr"]*lr_scale) for g in model.optimizer_groups()]}',flush=True)
    if completed >= end:
        if not (output / 'validation' / f'step_{completed:06d}' / 'summary.json').exists() or not (output / 'shared_descriptor.pt').exists():
            summary = validate(model,val,runtime.device,output,completed,args.workers,config['queries'])
            export_shared(model,output/'shared_descriptor.pt',dict(metadata,step=completed,validation=summary))
        print(f'Already completed step {completed}; output: {output}',flush=True)
        return
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed)).tolist()
    loader = iter(DataLoader(Subset(train, order[cursor:]), batch_size=1,
                            num_workers=args.workers, pin_memory=True,
                            generator=torch.Generator().manual_seed(seed)))
    model.train()
    bar = tqdm(range(completed, end), initial=completed, total=total_steps, desc='shared pretrain')
    for old_step in bar:
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        micro_count = min(accumulation, len(train) - cursor)
        records = []
        for micro in range(micro_count):
            batch = next(loader)
            loss, result, info = losses(model, batch, runtime.device,
                queries=config['queries'], seed=seed + cursor)
            (loss / micro_count).backward()
            records.append(info)
            cursor += 1
            del loss, result
        grad = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step()
        step = old_step + 1
        record = dict(step=step, cursor=cursor, loss=float(np.mean([x['loss'] for x in records])),
            gradient_norm=float(grad), microbatches=records, seconds=time.perf_counter()-start,
            peak_memory_gb=torch.cuda.max_memory_allocated()/1e9)
        with (output / 'train.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
        bar.set_postfix(loss=f'{record["loss"]:.4f}', GB=f'{record["peak_memory_gb"]:.1f}')
        if step % 50 == 0 or step == 1 or step == end:
            print(f'TRAIN step={step}/{total_steps} pairs={cursor}/{len(train)} '
                  f'loss={record["loss"]:.6f} peak_GB={record["peak_memory_gb"]:.2f}',flush=True)
        if step % config['save_every'] == 0 or step == end:
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, scheduler=scheduler,
                optimizer_step=step, data_progress={'cursor': cursor}, metadata=metadata)
        if step % 5000 == 0 or step == end:
            rng = capture_rng_state()
            summary = validate(model, val, runtime.device, output, step, args.workers, config['queries'])
            restore_rng_state(rng)
            # Final export includes all shared trainable components and LoRA when enabled.
            export_shared(model, output / 'shared_descriptor.pt', dict(metadata, step=step, validation=summary))
    print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
