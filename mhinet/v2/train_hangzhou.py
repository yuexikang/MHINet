"""Small joint run, fresh downstream weights, cosine LR and validation early stop.

Only DINO is pretrained. Run files are ordinary config, JSONL and checkpoints;
there is no code-hash registration chain. Epoch snapshots contain downstream
weights only, with the original DINO path recorded in the configuration.
"""
import argparse
from dataclasses import replace, asdict
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from mhinet.config import RuntimePaths
from .config import SemidenseConfigV2
from .data import GeometryPairDataset
from .feature_provider import build_fresh_shared
from .training import SemidenseSystemV2, batch_loss
from .semidense import SemidenseMatcherV2
from .pilot import select_rows, write_json, metrics, summarize
from .observe_qrru import observe


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def singleton(items):
    return items[0]


def means(rows):
    keys = {k for r in rows for k, v in r.items() if isinstance(v, (int, float))}
    return {k: float(np.mean([r[k] for r in rows if isinstance(r.get(k), (int, float))])) for k in keys}


@torch.no_grad()
def validate(system, dataset, config, device, epoch):
    system.eval()
    records, probes = [], []
    # Restore training RNG after validation. Every epoch gets the same queries.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.startswith('cuda') else []):
        torch.manual_seed(config['seed']+1000)
        for index in range(len(dataset)):
            batch = dataset[index]
            images, gt = batch['images'].to(device), batch['geometry'].to(device)
            shared = system(images)
            loss, record = batch_loss(system, batch, device, {**config, 'fixed_qrru_queries': True}, shared=shared)
            prediction = system.matcher.infer(shared)[0]
            reference = metrics(prediction, gt.H, [gt.mask_a, gt.mask_b])
            records.append(dict(pair_id=batch['pair_id'][0], group=batch['group'],
                                loss=float(loss), **record, **reference))
            if index % config['val_per_group'] < 2:
                probes.append(dict(pair_id=batch['pair_id'][0], group=batch['group'],
                                   **observe(system, shared, gt, seed=2000+index)))
    return dict(epoch=epoch, checked_at=now(), mean=means(records), pairs=records, qrru=probes)


def save(path, system, optimizer, scheduler, epoch, best, bad_epochs, config):
    state = {k: v.detach().cpu() for k, v in system.state_dict().items()
             if not k.startswith('shared.descriptor.encoder.frozen_dinov3.model.')}
    value = dict(model=state, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 epoch=epoch, best=best, bad_epochs=bad_epochs, config=config,
                 matcher=asdict(system.matcher.config), rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all())
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary); temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Two training updates; no formal run')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output'])
    if args.smoke:
        output = output.with_name(output.name+'_smoke')
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2); torch.manual_seed(config['seed'])
    np.random.seed(config['seed']); random.seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    runtime = replace(RuntimePaths.from_json(config['runtime']), device=args.device)
    selection_path = output/'selection.json'
    if args.resume:
        selection = json.loads(selection_path.read_text())
    else:
        root = Path(config['dataset'])
        selection = {split: select_rows(root, split, config[split+'_per_group'], config['seed']) for split in ('train', 'val')}
        train_blocks = {r['anchor_block_id'] for r in selection['train']}
        val_blocks = {r['anchor_block_id'] for r in selection['val']}
        if train_blocks & val_blocks:
            raise ValueError('Train/val spatial blocks overlap')
        write_json(selection_path, selection)
    write_json(output/'config.json', config)
    train = GeometryPairDataset(config['dataset'], selection['train'])
    val = GeometryPairDataset(config['dataset'], selection['val'])
    shared = build_fresh_shared(runtime)
    matcher = SemidenseMatcherV2(SemidenseConfigV2(**config['matcher']))
    system = SemidenseSystemV2(shared, matcher, 'joint').to(args.device)
    parameters = [p for p in system.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    first_epoch, best, bad_epochs = 1, float('inf'), 0
    if args.resume:
        state = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
        if state['config'] != config:
            raise ValueError('Resume config differs from the saved run')
        result = system.load_state_dict(state['model'], strict=False)
        if result.unexpected_keys or any(not k.startswith('shared.descriptor.encoder.frozen_dinov3.model.') for k in result.missing_keys):
            raise ValueError('Downstream checkpoint structure mismatch')
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        first_epoch, best, bad_epochs = state['epoch']+1, state['best'], state['bad_epochs']
        torch.set_rng_state(state['rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    write_json(output/'initialization.json', dict(dino=str(runtime.dino_checkpoint),
        pretrained_modules=['DINOv3 L11/L17'], fresh_modules=['MVT','VGG','CGMDP','Flow Head','D8/D2 temperatures','QRRU'],
        trainable_parameters=sum(p.numel() for p in parameters),
        frozen_dino_parameters=sum(p.numel() for p in system.shared.dino.parameters()),
        train_pairs=len(train), val_pairs=len(val), train_gt_types=sorted({r.get('gt_type','unknown') for r in selection['train']}),
        dormant_branch='D1 is not executed and excluded from training', started_at=now()))
    write_json(output/'status.json', dict(status='initial_validation' if first_epoch == 1 else 'running',
                                        epoch=first_epoch-1, checked_at=now()))
    stop = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    if first_epoch == 1 and not args.smoke:
        initial = validate(system, val, config, args.device, 0)
        write_json(output/'epoch_000.json', initial)
        save(output/'initial.pt', system, optimizer, scheduler, 0, best, 0, config)
        print(json.dumps(dict(event='before', **initial['mean'])), flush=True)
        write_json(output/'status.json', dict(status='running', epoch=0, checked_at=now()))
    final_epoch, status = first_epoch-1, 'running'
    for epoch in range(first_epoch, config['epochs']+1):
        if stop[0]:
            status = 'interrupted'; break
        system.train(); started = time.perf_counter(); rows = []
        loader = DataLoader(train, batch_size=1, shuffle=True, collate_fn=singleton,
            num_workers=config['workers'], pin_memory=True,
            generator=torch.Generator().manual_seed(config['seed']+epoch))
        optimizer.zero_grad(set_to_none=True)
        with (output/'train.jsonl').open('a') as stream:
            for index, batch in enumerate(loader):
                loss, record = batch_loss(system, batch, args.device, config)
                count = min(config['accumulation'], len(train)-(index//config['accumulation'])*config['accumulation'])
                (loss/count).backward()
                update = (index+1) % config['accumulation'] == 0 or index+1 == len(train)
                if update:
                    grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    if args.smoke:
                        groups = dict(mvt=system.shared.mvt, vgg=system.shared.vgg,
                                      decoder=system.shared.dedode, flow=system.shared.stage1_head,
                                      qrru=system.matcher.qrru)
                        gradient_report = {name: sum(float(p.grad.detach().abs().sum()) for p in module.parameters()
                                           if p.grad is not None) for name, module in groups.items()}
                        gradient_report['dino_has_gradient'] = any(p.grad is not None for p in system.shared.dino.parameters())
                        print(json.dumps(dict(event='gradient_smoke', **gradient_report)), flush=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                record.update(epoch=epoch, pair=index+1, pair_id=batch['pair_id'][0], loss=float(loss.detach()),
                              lr=optimizer.param_groups[0]['lr'])
                rows.append(record); stream.write(json.dumps(record)+'\n'); stream.flush()
                if args.smoke and index+1 >= config['accumulation']*2:
                    break
                if stop[0] and update:
                    break
        if args.smoke:
            write_json(output/'smoke.json', dict(status='passed', training=means(rows), peak_mb=torch.cuda.max_memory_allocated()/2**20))
            print(json.dumps(dict(event='smoke_passed', **means(rows))), flush=True)
            return
        if stop[0]:
            # Interrupted partial epochs are explicitly marked, not full epochs.
            save(output/'interrupted.pt', system, optimizer, scheduler, epoch-1, best, bad_epochs, config)
            status = 'interrupted'; break
        validation = validate(system, val, config, args.device, epoch)
        validation.update(train_mean=means(rows), seconds=time.perf_counter()-started,
                          lr=optimizer.param_groups[0]['lr'])
        write_json(output/f'epoch_{epoch:03d}.json', validation)
        score = validation['mean']['loss']
        improved = score < best-config['min_delta']
        best = score if improved else best
        bad_epochs = 0 if improved else bad_epochs+1
        scheduler.step()
        save(output/'latest.pt', system, optimizer, scheduler, epoch, best, bad_epochs, config)
        if improved:
            save(output/'best.pt', system, optimizer, scheduler, epoch, best, bad_epochs, config)
        if epoch == config['epochs']//2:
            save(output/'middle.pt', system, optimizer, scheduler, epoch, best, bad_epochs, config)
        final_epoch = epoch
        status = 'early_stopped' if epoch >= config['min_epochs'] and bad_epochs >= config['patience'] else 'running'
        write_json(output/'status.json', dict(status=status, epoch=epoch, best_val_loss=best, bad_epochs=bad_epochs,
                                            val=validation['mean'], checked_at=now()))
        print(json.dumps(dict(event='epoch', epoch=epoch, status=status, seconds=validation['seconds'],
                             train_loss=validation['train_mean']['loss'], val=validation['mean'])), flush=True)
        if status == 'early_stopped':
            break
    if status == 'running':
        status = 'completed'
    epochs = sorted(int(p.stem.split('_')[1]) for p in output.glob('epoch_*.json'))
    if epochs:
        comparison = {name: json.loads((output/f'epoch_{e:03d}.json').read_text())
                      for name, e in zip(('before','middle','after'), (0, epochs[len(epochs)//2], epochs[-1]))}
        write_json(output/'qrru_before_middle_after.json', comparison)
        from .report_hangzhou import write_report
        write_report(output)
    write_json(output/'status.json', dict(status=status, epoch=final_epoch, best_val_loss=best,
                                        checked_at=now(), next_action='Review QRRU before/middle/after; no automatic scale-up'))


if __name__ == '__main__':
    main()
