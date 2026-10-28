"""Full RGB: train MVT/Flow, freeze locator, train through modulated D2, then test."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader

from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.frozen_locator import load_locator_weights
from mhinet.v2.losses import flow_head_loss
from mhinet.v2.coarse_matcher import flow_prior_from_shared
from mhinet.v2.flow_d8_d2.matching import local_coarse
from mhinet.v2.flow_d8_d2.run import measure
from mhinet.v2.qrru_corners.experiment import load_models, singleton
from mhinet.v2.joint_single.path import coarse, sample, source_grid
from mhinet.v2.joint_single.loss import global_d8, local_loss
from .model import load_adapter, center_indices, match_windows, infer
from .experiment import loss_fn
from .no_flow import global_infer


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def datasets(config, smoke=False):
    rgb, hz = Path(config['rgb_root']), Path(config['hangzhou_root'])
    specs = {s: (rgb/s, rgb/s/'pairs.jsonl') for s in ('train', 'val', 'test')}
    specs.update({f'hangzhou_{s}': (hz, hz/'manifests'/f'{s}.jsonl') for s in config['hangzhou_tests']})
    result, inventory = {}, {}
    identities = []
    for split, (root, manifest) in specs.items():
        rows = [json.loads(line) for line in manifest.open() if line.strip()]
        if split in ('train', 'val', 'test'):
            identities.append({r['source_path'] for r in rows})
        if smoke: rows = rows[:8 if split == 'train' else 2]
        result[split] = GeometryPairDataset(root, rows)
        inventory[split] = dict(manifest=str(manifest), root=str(root), pairs=len(rows))
    assert not any(a & b for i, a in enumerate(identities) for b in identities[i+1:]), 'RGB source split overlap'
    return result, inventory


def loader(dataset, workers, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=workers,
                      collate_fn=singleton, generator=torch.Generator().manual_seed(seed))


def flow_points(shared):
    field = shared['flow16_target_norm'][0]
    y, x = torch.meshgrid(torch.arange(32, device=field.device), torch.arange(32, device=field.device), indexing='ij')
    a = (torch.stack((x, y), -1).reshape(-1, 2)+.5)*16-.5
    return a, (field.flatten(1).T+1)*256-.5


def totals(records):
    result = {k: sum(r[k] for r in records) for k in ('n', 'visible', 'error_sum', 'correct1', 'correct3', 'correct5', 'correct10')}
    result.update(epe=result['error_sum']/max(1, result['visible']),
                  p3_visible=result['correct3']/max(1, result['visible']))
    result.update({f'p{t}': result[f'correct{t}']/max(1, result['n']) for t in (1, 3, 5, 10)})
    return result


def advance_early_stop(anchor, bad_epochs, score, min_delta):
    """Meaningful progress controls patience only, never best checkpoint saving."""
    return (score, 0) if score > anchor+min_delta else (anchor, bad_epochs+1)


def matching_loss(shared, matcher, adapter, gt, config):
    # Discrete parent selection is detached; D8 logits and sampled D2 features remain live.
    with torch.no_grad():
        prior = flow_prior_from_shared(shared, 0, (64, 64))
        si, _, _, _ = local_coarse(shared['pyramid'][8][0].float(), prior, matcher)
        si = si[:128]
        si = si[torch.randperm(len(si), device=si.device)[:config['train_windows']]]
        source = source_grid(si.device)[si]
    l8 = global_d8(shared, matcher, gt)
    if not len(source):
        return l8, dict(d8_global=float(l8.detach()), d8_local=0., d2=0., labels=0)
    r = coarse(shared, matcher, source)
    local8, _, _, _ = local_loss(source, r['p8'], r['candidates8'], r['logits8'], r['valid8'], gt)
    # Train features/classification with fixed window coordinates; no QRRU or Flow gradient.
    axis = (torch.arange(8, device=source.device)-3.5)*2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    delta = torch.stack((x, y), -1).reshape(1, 64, 2)
    ca, cb = source[:, None]+delta, r['p8'].detach()[:, None]+delta
    d2 = shared['pyramid'][2][0].float()
    queries = ca[:, center_indices(source.device)]
    with torch.no_grad():
        truth, known, visible = gt.query((queries+.5)/256-1)
    windows = dict(a=sample(d2[0], ca), b=sample(d2[1], cb),
                   valid_a=(ca >= 0).all(-1) & (ca <= 511).all(-1),
                   valid_b=(cb >= 0).all(-1) & (cb <= 511).all(-1),
                   candidates=cb, truth=(truth+1)*256-.5, labeled=known & visible)
    fine = match_windows(adapter, windows, matcher.tau_f)
    l2, labels = loss_fn(fine, windows)
    total = config['d8_global_weight']*l8+config['d8_local_weight']*local8+config['d2_weight']*l2
    return total, dict(d8_global=float(l8.detach()), d8_local=float(local8.detach()), d2=float(l2.detach()), labels=labels)


@torch.no_grad()
def evaluate(model, matcher, adapter, dataset, device, workers, stage, all_variants=False):
    model.eval(); matcher.eval(); adapter.eval()
    rows = []
    for i, batch in enumerate(loader(dataset, workers)):
        model.geometry_mode = 'none' if all_variants else 'flow_d16'
        shared = model(batch['images'].to(device), pyramid_scales=() if stage == 'flow' else (8, 4, 2))
        gt = batch['geometry'].to(device)
        predictions = {}
        if all_variants:
            predictions['global_no_flow'] = global_infer(shared, matcher, adapter)
            from mhinet.v2.feature_provider import FlowStage1Adapter, _flow_fields
            stage1 = FlowStage1Adapter.forward(model.stage1_head, shared['context'], position_dtype=shared['pair_descriptors'].dtype)
            shared['stage1'] = stage1
            shared.update(_flow_fields(stage1))
            predictions['global_flow'] = global_infer(shared, matcher, adapter, flow_prior_from_shared(shared, 0, (64, 64)))
        a, b = flow_points(shared)
        predictions['flow'] = dict(points_a=a, points_b=b)
        if stage == 'd2': predictions['local_flow'] = infer(shared, matcher, adapter)
        row = dict(pair_id=batch['pair_id'][0])
        for name, prediction in predictions.items():
            if not torch.isfinite(prediction['points_b']).all(): raise FloatingPointError(name)
            row[name] = measure(prediction['points_a'], prediction['points_b'], gt)[0]
        if stage == 'flow':
            loss, _ = flow_head_loss(shared['stage1'], geometry=gt)
            row['validation_loss'] = float(loss)
        rows.append(row)
        if (i+1) % 256 == 0: print(json.dumps(dict(event='evaluation', stage=stage, pairs=i+1, total=len(dataset))), flush=True)
    names = [k for k in rows[0] if k not in ('pair_id', 'validation_loss')]
    model.geometry_mode = 'flow_d16'
    return dict(summary={k: totals([r[k] for r in rows]) for k in names}, pairs=rows)


def snapshot(model, matcher, adapter, stage):
    if stage == 'flow': return dict(mvt=model.mvt.state_dict(), head=model.stage1_head.state_dict())
    return dict(vgg=model.vgg.state_dict(), dedode=model.dedode.state_dict(), adapter=adapter.state_dict(),
                temperatures={k: v for k, v in matcher.state_dict().items() if not k.startswith('qrru.')})


def restore(state, model, matcher, adapter, stage):
    if stage == 'flow':
        model.mvt.load_state_dict(state['mvt']); model.stage1_head.load_state_dict(state['head'])
    else:
        model.vgg.load_state_dict(state['vgg']); model.dedode.load_state_dict(state['dedode'])
        adapter.load_state_dict(state['adapter'])
        matcher.load_state_dict({**matcher.state_dict(), **state['temperatures']})


def train_stage(stage, model, matcher, adapter, ds, config, device, out, smoke, resume):
    out.mkdir(parents=True, exist_ok=True)
    modules = dict(mvt=model.mvt, head=model.stage1_head, vgg=model.vgg, dedode=model.dedode, matcher=matcher, adapter=adapter)
    flow = stage == 'flow'
    model.set_training_groups(mvt=flow, ghim_head=flow, vgg=not flow, dedode=not flow)
    matcher.requires_grad_(not flow); matcher.qrru.requires_grad_(False)
    adapter.requires_grad_(not flow)
    rates = config['flow_lr' if flow else 'd2_lr']
    groups = [dict(name=k, params=[p for p in modules[k].parameters() if p.requires_grad], lr=lr) for k, lr in rates.items()]
    params = [p for g in groups for p in g['params']]
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'])
    total_steps = math.ceil(len(ds['train'])/config['accumulation'])*config['epochs']
    def multiplier(step):
        if step < config['warmup_steps']:
            return .1+.9*step/config['warmup_steps']
        progress = (step-config['warmup_steps'])/max(1, total_steps-config['warmup_steps'])
        return config['cosine_min_ratio']+(1-config['cosine_min_ratio'])*(1+math.cos(math.pi*progress))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    start_epoch, best = 0, -1.
    early_best, bad_epochs = -1., 0
    latest = out/'latest.pt'
    def save(name, epoch, score):
        state = dict(**snapshot(model, matcher, adapter, stage), config=config, matcher_config=asdict(matcher.config),
                     stage=stage, epoch=epoch, score=score, best=best, optimizer=optimizer.state_dict(),
                     early_best=early_best, bad_epochs=bad_epochs,
                     locator_checkpoint=(config.get('frozen_flow_checkpoint') or str(out.parent/'flow'/'best.pt')) if stage == 'd2' else config['initial_locator'],
                     scheduler=scheduler.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                     python_rng=random.getstate())
        temporary = out/(name+'.tmp')
        torch.save(state, temporary); temporary.replace(out/name)
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        restore(state, model, matcher, adapter, stage)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_epoch, best = state['epoch'], state['best']
        if 'early_best' in state:
            early_best, bad_epochs = state['early_best'], state['bad_epochs']
        else:
            # Adopt the new stopping policy at an old epoch boundary without losing progress.
            key, metric = ('flow', 'p3_visible') if flow else ('local_flow', 'p3')
            early_best = json.loads((out/'epoch_000.json').read_text())['summary'][key][metric]
            for previous in range(1, start_epoch+1):
                score = json.loads((out/f'epoch_{previous:03d}.json').read_text())['summary'][key][metric]
                early_best, bad_epochs = advance_early_stop(early_best, bad_epochs, score, config['early_stop_min_delta'])
        if state['config']['epochs'] != config['epochs']:
            for group, base in zip(optimizer.param_groups, scheduler.base_lrs):
                group['lr'] = base*multiplier(scheduler.last_epoch)
            scheduler._last_lr = [g['lr'] for g in optimizer.param_groups]
            print(json.dumps(dict(event='schedule_rebased', old_epochs=state['config']['epochs'],
                                  epochs=config['epochs'], optimizer_step=scheduler.last_epoch)), flush=True)
        torch.set_rng_state(state['rng'].cpu()); torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
        random.setstate(state['python_rng'])
        del state
    elif (out/'best.pt').exists():
        raise FileExistsError(f'{out}: use --resume or a new output directory')
    else:
        initial = evaluate(model, matcher, adapter, ds['val'], device, config['workers'], stage)
        write(out/'epoch_000.json', initial)
        best = initial['summary']['flow' if flow else 'local_flow']['p3_visible' if flow else 'p3']
        early_best = best
        save('best.pt', 0, best); save('latest.pt', 0, best)
    print(json.dumps(dict(event='stage_start', stage=stage, initial_best=best,
                          trainable={g['name']: sum(p.numel() for p in g['params']) for g in groups})), flush=True)
    def should_stop(epoch):
        return (config['early_stopping'] and epoch >= config['early_stop_min_epochs']
                and bad_epochs >= config['early_stop_patience'])
    last_epoch = start_epoch
    for epoch in range(start_epoch+1, config['epochs']+1):
        if should_stop(last_epoch): break
        model.train(); matcher.train(); matcher.qrru.eval(); adapter.train(not flow)
        started = time.perf_counter(); total_loss = 0.; last_metrics = {}
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        batches = loader(ds['train'], config['workers'], True, config['seed']+epoch)
        for index, batch in enumerate(batches):
            shared = model(batch['images'].to(device), pyramid_scales=() if flow else (8, 4, 2))
            gt = batch['geometry'].to(device)
            if not gt.exact: raise ValueError('Full RGB training requires exact H')
            if flow: loss, last_metrics = flow_head_loss(shared['stage1'], geometry=gt, **config['flow_loss'])
            else: loss, last_metrics = matching_loss(shared, matcher, adapter, gt, config)
            if not torch.isfinite(loss): raise FloatingPointError(last_metrics)
            count = min(config['accumulation'], len(ds['train'])-(index//config['accumulation'])*config['accumulation'])
            (loss/count).backward(); total_loss += float(loss.detach())
            if (index+1) % config['accumulation'] == 0 or index+1 == len(ds['train']):
                norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                if smoke:
                    gradients = {k: sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
                                 for k, m in {**modules, 'dino': model.dino, 'qrru': matcher.qrru}.items()}
                    assert gradients['dino'] == gradients['qrru'] == 0, gradients
                    if flow: assert gradients['mvt'] > 0 and gradients['head'] > 0 and gradients['vgg'] == 0, gradients
                    else: assert gradients['mvt'] == gradients['head'] == 0 and gradients['vgg'] > 0 and gradients['dedode'] > 0 and gradients['adapter'] > 0, gradients
                    print(json.dumps(dict(event='gradients', stage=stage, values=gradients)), flush=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if index == 0 or (index+1) % 100 == 0:
                print(json.dumps(dict(event='train', stage=stage, epoch=epoch, pairs=index+1,
                                      loss=total_loss/(index+1), seconds=time.perf_counter()-started,
                                      peak_gib=torch.cuda.max_memory_allocated(device)/2**30, metrics=last_metrics)), flush=True)
        train_seconds = time.perf_counter()-started
        result = evaluate(model, matcher, adapter, ds['val'], device, config['workers'], stage)
        write(out/f'epoch_{epoch:03d}.json', result)
        score = result['summary']['flow' if flow else 'local_flow']['p3_visible' if flow else 'p3']
        early_best, bad_epochs = advance_early_stop(early_best, bad_epochs, score, config['early_stop_min_delta'])
        if score > best:
            best = score; save('best.pt', epoch, score)
        save('latest.pt', epoch, score)
        record = dict(epoch=epoch, train_loss=total_loss/len(ds['train']), train_seconds=train_seconds,
                      seconds=time.perf_counter()-started, best=best, validation=result['summary'],
                      early_best=early_best, bad_epochs=bad_epochs,
                      lr={g['name']: g['lr'] for g in optimizer.param_groups}, peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        with (out/'history.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        print(json.dumps(dict(event='epoch', stage=stage, **record)), flush=True)
        last_epoch = epoch
        if should_stop(epoch):
            print(json.dumps(dict(event='early_stop', stage=stage, epoch=epoch, bad_epochs=bad_epochs)), flush=True)
            break
    state = torch.load(out/'best.pt', map_location=device, weights_only=False)
    restore(state, model, matcher, adapter, stage)
    write(out/'done.json', dict(best_epoch=state['epoch'], best=state['score'], epochs=last_epoch,
                               max_epochs=config['epochs'], early_stopped=should_stop(last_epoch)))
    del state, optimizer, scheduler
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/mhi_v2/full_rgb_d2.json'))
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=['all', 'flow', 'd2'], default='all')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(2); torch.manual_seed(config['seed']); random.seed(config['seed'])
    if args.smoke: config.update(epochs=1, workers=0)
    out = args.output or Path(config['output']+('_smoke' if args.smoke else ''))
    out.mkdir(parents=True, exist_ok=True)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    ds, inventory = datasets(config, args.smoke)
    write(out/'data.json', inventory); write(out/'config.json', config)
    model, matcher = load_models(args.device)
    load_locator_weights(model, config['initial_locator'])
    downstream = torch.load(config['initial_downstream'], map_location='cpu', weights_only=False)
    model.vgg.load_state_dict(downstream['vgg']); model.dedode.load_state_dict(downstream['dedode'])
    matcher.load_state_dict({**matcher.state_dict(), **downstream['temperatures']})
    del downstream
    adapter = load_adapter(config['initial_adapter'], args.device)
    if args.stage == 'd2':
        # Independent continuation preserves the locator used by the original D2 run.
        load_locator_weights(model, config.get('frozen_flow_checkpoint') or out/'flow'/'best.pt')
    def prohibit(*unused): raise RuntimeError('QRRU is disabled')
    matcher.qrru.register_forward_pre_hook(prohibit)
    stages = ('flow', 'd2') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        train_stage(stage, model, matcher, adapter, ds, config, args.device, out/stage, args.smoke, args.resume)
    final_stage = stages[-1]
    for checkpoint in (('best', 'latest') if config.get('test_last_epoch') else ('best',)):
        if checkpoint == 'latest':
            state = torch.load(out/final_stage/'latest.pt', map_location=args.device, weights_only=False)
            restore(state, model, matcher, adapter, final_stage)
            del state
        for name, dataset in ds.items():
            if name in ('train', 'val'): continue
            result = evaluate(model, matcher, adapter, dataset, args.device, config['workers'], final_stage, all_variants=final_stage=='d2')
            filename = f'test_{name}.json' if args.stage == 'all' and checkpoint == 'best' else f'{final_stage}_{checkpoint}_test_{name}.json'
            write(out/filename, result)
            print(json.dumps(dict(event='test', stage=final_stage, checkpoint=checkpoint, split=name, summary=result['summary'])), flush=True)
    completion = dict(status='complete', stages=list(stages), tests=[k for k in ds if k not in ('train', 'val')])
    if args.stage == 'all': write(out/'done.json', completion)
    else:
        write(out/f'done_{args.stage}.json', completion)
        if all((out/f'done_{s}.json').exists() for s in ('flow', 'd2')):
            write(out/'done.json', dict(completion, stages=['flow','d2']))
    print('PIPELINE COMPLETE', flush=True)


if __name__ == '__main__': main()
