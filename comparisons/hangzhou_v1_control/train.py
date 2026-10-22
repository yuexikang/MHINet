"""V1 architecture/native losses, with the Hangzhou v2 sample and training budget.

The frozen v1 source is unchanged. Only DINO is pretrained. This compares complete
training recipes, not an architecture-only ablation: v1 retains H/control losses.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from mhinet.config import RuntimePaths
from mhinet.downstream.semidense import SemidenseMatcher, SemidenseConfig
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.feature_provider import build_fresh_shared
from mhinet.v2.training import SemidenseSystemV2
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.pilot import write_json, metrics
from mhinet.v2.train_hangzhou import now, singleton, means, save


def build_system(runtime, config):
    # Identical fresh shared tensors to v2; replace only the parameter-free H path.
    shared = build_fresh_shared(runtime)
    from experiments.stage1_loma_shared_v1.stage1_head import Stage1HeadFromSharedMVT
    from mhinet.models.feature_provider import SafeMatchabilityWeightedHomographyFitter
    shared.stage1_head.enable_amp = True
    source = SimpleNamespace(locator=shared.stage1_head,
                             homography_fitter=SafeMatchabilityWeightedHomographyFitter())
    shared.stage1_head = Stage1HeadFromSharedMVT(source).to(runtime.device)
    shared.geometry_mode = 'h0'
    matcher = SemidenseMatcher(SemidenseConfig(**config['matcher']))
    # This container only configures trainability; inference/losses are native v1.
    return SemidenseSystemV2(shared, matcher, 'joint').to(runtime.device)


def batch_loss(system, batch, device, shared=None):
    from experiments.loretta_stage1_h.losses import Stage1HLoss
    gt = batch['geometry'].to(device)
    shared = system(batch['images'].to(device)) if shared is None else shared
    stage_loss, stage_record, _ = Stage1HLoss()(shared['stage1'],
        dict(H_A_to_B_norm=gt.H[None], mask_A_overlap=gt.mask_a[None]))
    match_loss, records = system.matcher.training_losses(shared, gt.H[None], gt.mask_a[None], gt.mask_b[None])
    loss = stage_loss + match_loss
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite v1 training loss')
    # Common metric only; this adapted loss does not contribute gradients to v1.
    with torch.no_grad():
        _, common = flow_head_loss(shared['stage1'], geometry=gt)
    record = dict(records[0], **{f'stage1_{k}': float(v.detach()) for k,v in stage_record.items()},
                  flow_epe_px=common['flow_epe_px'])
    return loss, record


@torch.no_grad()
def validate(system, dataset, config, device, epoch):
    system.eval(); rows = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for index, batch in enumerate(dataset):
            # Per-pair seeds prevent earlier H0 branch changes shifting later RNG.
            torch.manual_seed(config['seed']+1000+index)
            shared = system(batch['images'].to(device))
            loss, record = batch_loss(system, batch, device, shared)
            gt = batch['geometry'].to(device)
            pred = system.matcher.infer(shared)[0]
            rows.append(dict(pair_id=batch['pair_id'][0], group=batch['group'], loss=float(loss),
                             **record, **metrics(pred, gt.H, [gt.mask_a, gt.mask_b])))
    return dict(epoch=epoch, checked_at=now(), mean=means(rows), pairs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text()); device = 'cuda:0'
    output = Path(config['output'] + ('_smoke' if args.smoke else ''))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    random.seed(config['seed']); np.random.seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    runtime = replace(RuntimePaths.from_json(config['runtime']), device=device)
    selection = json.loads(Path(config['selection']).read_text())
    write_json(output/'selection.json', selection); write_json(output/'config.json', config)
    train = GeometryPairDataset(config['dataset'], selection['train'])
    val = GeometryPairDataset(config['dataset'], selection['val'])
    system = build_system(runtime, config)
    from mhinet.v2.frozen_locator import initialize_locator
    initialization = initialize_locator(system, config)
    parameters = [p for p in system.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    write_json(output/'initialization.json', dict(**initialization, dino=str(runtime.dino_checkpoint),
        trainable_parameters=sum(p.numel() for p in parameters), train_pairs=len(train), val_pairs=len(val),
        architecture='v1 native H0 + D8/D2 + QRRU', losses='native Stage1HLoss + native Lc/Lf/Lq',
        label_source='approximate geographic H, no independently annotated points/flow', started_at=now()))
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    best, bad, final_epoch = float('inf'), 0, 0
    if not args.smoke:
        write_json(output/'status.json', dict(status='initial_validation', epoch=0, checked_at=now()))
        initial = validate(system, val, config, device, 0)
        write_json(output/'epoch_000.json', initial)
        print(json.dumps(dict(event='before', **initial['mean'])), flush=True)
    status = 'running'
    for epoch in range(1, config['epochs']+1):
        system.train(); started = time.perf_counter(); records = []
        loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=singleton,
            num_workers=config['workers'], pin_memory=True,
            generator=torch.Generator().manual_seed(config['seed']+epoch))
        optimizer.zero_grad(set_to_none=True)
        with (output/'train.jsonl').open('a') as stream:
            for index, batch in enumerate(loader):
                loss, record = batch_loss(system, batch, device)
                count = min(config['accumulation'], len(train)-(index//config['accumulation'])*config['accumulation'])
                (loss/count).backward()
                if (index+1) % config['accumulation'] == 0 or index+1 == len(train):
                    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    if args.smoke:
                        modules = dict(mvt=system.shared.mvt, vgg=system.shared.vgg,
                            decoder=system.shared.dedode, stage1=system.shared.stage1_head, qrru=system.matcher.qrru)
                        print(json.dumps(dict(event='gradients', **{k:sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None) for k,m in modules.items()},
                            dino_has_gradient=any(p.grad is not None for p in system.shared.dino.parameters()))), flush=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                record.update(epoch=epoch, pair=index+1, pair_id=batch['pair_id'][0], loss=float(loss.detach()), lr=optimizer.param_groups[0]['lr'])
                records.append(record); stream.write(json.dumps(record)+'\n'); stream.flush()
                if stop[0] or (args.smoke and index+1 >= config['accumulation']*2):
                    break
        if args.smoke:
            checked = validate(system, GeometryPairDataset(config['dataset'], selection['val'][:2]), config, device, 0)
            write_json(output/'smoke.json', dict(status='passed', training=means(records), validation=checked['mean'], peak_mb=torch.cuda.max_memory_allocated()/2**20))
            print('Smoke passed', flush=True); return
        if stop[0]:
            status = 'interrupted'; break
        validation = validate(system, val, config, device, epoch)
        validation.update(train_mean=means(records), seconds=time.perf_counter()-started, lr=optimizer.param_groups[0]['lr'])
        write_json(output/f'epoch_{epoch:03d}.json', validation)
        improved = validation['mean']['loss'] < best-config['min_delta']
        best = validation['mean']['loss'] if improved else best
        bad = 0 if improved else bad+1
        scheduler.step()
        save(output/'latest.pt', system, optimizer, scheduler, epoch, best, bad, config)
        if improved:
            save(output/'best.pt', system, optimizer, scheduler, epoch, best, bad, config)
        final_epoch = epoch
        status = 'early_stopped' if epoch >= config['min_epochs'] and bad >= config['patience'] else 'running'
        write_json(output/'status.json', dict(status=status, epoch=epoch, best_val_loss=best, bad_epochs=bad, val=validation['mean'], checked_at=now()))
        print(json.dumps(dict(event='epoch', epoch=epoch, status=status, seconds=validation['seconds'], val=validation['mean'])), flush=True)
        if status == 'early_stopped':
            break
    if status == 'running':
        status = 'completed'
    write_json(output/'status.json', dict(status=status, epoch=final_epoch, best_val_loss=best, checked_at=now()))
    from comparisons.hangzhou_v1_control.report import write_report
    write_report(output, Path(config['v2_run']))


if __name__ == '__main__':
    main()
