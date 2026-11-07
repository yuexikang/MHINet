"""Joint RGB512 training: frozen Flow0, trainable D8/D2 + Pair + QRRU refinement.

Data-parallel over N GPUs: every rank owns ``accumulation`` pairs per optimizer
step (4 pairs x 3 ranks = global 12). Gradients are averaged through one flat
FP32 buffer instead of ``DistributedDataParallel`` because the model is a set of
sibling modules with a permanently frozen first stage and per-step conditional
losses.

Frozen:  DINOv3, MVT, Flow0 head (coarse warp + matchability + no-match logit).
Trained: VGG, CGMDP decoder (optionally with the validated residual depth),
         D8/D2 temperatures, Pair modulation and the QRRU local refiner.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from mhinet.config import sha256_file
from mhinet.v2.config import SemidenseConfigV2
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.frozen_locator import load_locator_weights
from mhinet.v2.local_refinement import LocalRefinement, local_refinement_loss
from mhinet.v2.pair_matching.coarse import select_local_parents
from mhinet.v2.pair_matching.continuous import coarse, sample
from mhinet.v2.pair_matching.deep_d2 import configure_d2
from mhinet.v2.pair_matching.local import (
    LocalPairModulation, center_indices, extract_windows, match_windows)
from mhinet.v2.pair_matching.losses import global_d8, local_loss, loss_fn
from mhinet.v2.pair_matching.metrics import measure
from mhinet.v2.pipeline import MatchingTemperatures, ROOT, asset_path, build_shared

METRIC_KEYS = ('n', 'visible', 'error_sum', 'correct1', 'correct3', 'correct5', 'correct10')


# ---------------------------------------------------------------- small helpers

def timestamp():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def singleton(items):
    return items[0]


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.temporary')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(target)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def append_jsonl(path, row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False) + '\n')


# ---------------------------------------------------------------- configuration

def read_config(path, assets_root):
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    required = {'runtime', 'rgb_root', 'frozen_flow_checkpoint', 'output', 'seed', 'epochs',
                'world_size', 'accumulation', 'workers', 'weight_decay', 'cosine_min_ratio',
                'warmup_steps', 'grad_clip', 'lr', 'train_windows', 'refine_points', 'refine',
                'loss_weights', 'd2_refinement', 'early_stopping', 'matcher'}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f'Missing joint-training fields: {missing}')
    if config['early_stopping']:
        raise ValueError('This protocol fixes early_stopping=false (20 fixed epochs)')
    if set(config['lr']) != {'vgg', 'dedode', 'adapter', 'matcher', 'refiner'}:
        raise ValueError('lr must name exactly vgg/dedode/adapter/matcher/refiner')
    if set(config['loss_weights']) != {'d8_global', 'd8_local', 'd2', 'refine'}:
        raise ValueError('loss_weights must name d8_global/d8_local/d2/refine')
    for name in ('epochs', 'world_size', 'accumulation', 'workers', 'train_windows', 'refine_points'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['refine']['enabled'] and config['loss_weights']['refine'] <= 0:
        raise ValueError('Refinement is enabled with a non-positive loss weight')
    config['output'] = str(asset_path(config['output'], assets_root))
    for key in ('rgb_root', 'frozen_flow_checkpoint'):
        config[key] = str(asset_path(config[key], assets_root))
    config['runtime'] = (config['runtime'] if isinstance(config['runtime'], dict)
                         else json.loads(Path(str(asset_path(config['runtime'], assets_root))).read_text()))
    config['matcher_config'] = SemidenseConfigV2(**config['matcher'])
    return config


# ---------------------------------------------------------------- model assembly

class JointContext:
    """The trainable scope and the two permanent frozen halves."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.max_parents = 128
        torch.manual_seed(config['seed'])
        self.shared = build_shared(config['runtime'], device, ROOT)
        configure_d2(self.shared, config['d2_refinement'])
        load_locator_weights(self.shared, config['frozen_flow_checkpoint'])
        # DINOv3 is forced frozen inside the provider; keep the whole first stage off.
        self.shared.set_training_groups(mvt=False, vgg=True, dedode=True, ghim_head=False)
        self.matcher = MatchingTemperatures(config['matcher_config']).to(device)
        self.adapter = LocalPairModulation('pair').to(device)
        self.refiner = LocalRefinement(radius=config['refine']['radius'],
                                       chunk_size=config['refine']['chunk_size']).to(device)
        self.train()

    def train(self):
        self.shared.train()
        self.matcher.train()
        self.adapter.train()
        self.refiner.train()

    def eval(self):
        self.shared.eval()
        self.matcher.eval()
        self.adapter.eval()
        self.refiner.eval()

    def groups(self):
        c = self.config
        return [
            dict(name='vgg', lr=c['lr']['vgg'], params=[p for p in self.shared.vgg.parameters() if p.requires_grad]),
            dict(name='dedode', lr=c['lr']['dedode'], params=[p for p in self.shared.dedode.parameters() if p.requires_grad]),
            dict(name='adapter', lr=c['lr']['adapter'], params=[p for p in self.adapter.parameters() if p.requires_grad]),
            dict(name='matcher', lr=c['lr']['matcher'], params=[p for p in self.matcher.parameters() if p.requires_grad]),
            dict(name='refiner', lr=c['lr']['refiner'], params=[p for p in self.refiner.parameters() if p.requires_grad]),
        ]

    def frozen_modules(self):
        return [('mvt', self.shared.mvt), ('head', self.shared.stage1_head)]

    def snapshot(self):
        return dict(vgg=self.shared.vgg.state_dict(), dedode=self.shared.dedode.state_dict(),
                    adapter=self.adapter.state_dict(), temperatures=self.matcher.state_dict(),
                    refiner=self.refiner.state_dict())

    def restore(self, state):
        self.shared.vgg.load_state_dict(state['vgg'], strict=True)
        self.shared.dedode.load_state_dict(state['dedode'], strict=True)
        self.adapter.load_state_dict(state['adapter'], strict=True)
        self.matcher.load_state_dict(state['temperatures'], strict=True)
        self.refiner.load_state_dict(state['refiner'], strict=True)


def state_digest(modules):
    """Bit-level digest of frozen weights, so a stray gradient cannot pass unnoticed."""
    digest = hashlib.sha256()
    for name, module in modules:
        for key, value in sorted(module.state_dict().items()):
            digest.update(f'{name}.{key}'.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().reshape(-1).to('cpu', torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


def parameter_digest(params):
    digest = hashlib.sha256()
    for parameter in params:
        digest.update(parameter.detach().reshape(-1).to('cpu', torch.float32).contiguous().numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- parallel

class Parallel:
    def __init__(self):
        self.enabled = 'RANK' in os.environ and int(os.environ.get('WORLD_SIZE', '1')) > 1
        if self.enabled:
            dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200))
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            local = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local)
            self.device = torch.device('cuda', local)
        else:
            self.rank, self.world = 0, 1
            torch.cuda.set_device(0)
            self.device = torch.device('cuda', 0)
        self.main = self.rank == 0

    def barrier(self):
        if self.enabled:
            dist.barrier()

    def gather(self, value):
        if not self.enabled:
            return [value]
        bucket = [None] * self.world
        dist.all_gather_object(bucket, value)
        return bucket


class GradientSync:
    """copy -> all_reduce -> divide; then clip and step on identical replicas."""

    def __init__(self, params, device):
        self.params = list(params)
        total = sum(p.numel() for p in self.params)
        self.flat = torch.zeros(total, device=device, dtype=torch.float32)
        views, offset = [], 0
        for parameter in self.params:
            views.append(self.flat[offset:offset + parameter.numel()].view_as(parameter))
            offset += parameter.numel()
        self.views = views
        self.elements = total

    def average(self, world):
        if world == 1:
            return
        for view, parameter in zip(self.views, self.params):
            if parameter.grad is None:
                view.zero_()
            else:
                view.copy_(parameter.grad)
        dist.all_reduce(self.flat)
        self.flat.div_(world)
        for view, parameter in zip(self.views, self.params):
            parameter.grad = view if view.dtype == parameter.dtype else view.to(parameter.dtype)


# ---------------------------------------------------------------- loss

def pair_terms(ctx, shared, gt, generator):
    """D8 global + D8 local + D2 + QRRU refinement (localization and harm)."""
    c = ctx.config
    d2 = shared['pyramid'][2][0].float()
    l8g = global_d8(shared, ctx.matcher, gt)
    with torch.no_grad():
        selection = select_local_parents(shared, ctx.matcher, ctx.max_parents)
    take = min(c['train_windows'], len(selection))
    if not take:
        zero = d2.sum() * 0
        return zero, dict(d8_global=float(l8g.detach()), d8_local=0., d2=0., refine=0.,
                          localization=0., harm=0., labels=0, refine_points=0, windows=0)
    order = torch.randperm(len(selection), generator=generator, device=selection.device)[:take]
    source = selection[order]
    r = coarse(shared, ctx.matcher, source)
    l8l, _, _, _ = local_loss(source, r['p8'], r['candidates8'], r['logits8'], r['valid8'], gt)
    axis = (torch.arange(8, device=source.device) - 3.5) * 2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    delta = torch.stack((x, y), -1).reshape(1, 64, 2)
    corners_a, corners_b = source[:, None] + delta, r['p8'].detach()[:, None] + delta
    queries = corners_a[:, center_indices(source.device)]
    with torch.no_grad():
        truth, known, visible = gt.query((queries + .5) / 256 - 1)
    windows = dict(a=sample(d2[0], corners_a), b=sample(d2[1], corners_b),
                   valid_a=(corners_a >= 0).all(-1) & (corners_a <= 511).all(-1),
                   valid_b=(corners_b >= 0).all(-1) & (corners_b <= 511).all(-1),
                   candidates=corners_b, truth=(truth + 1) * 256 - .5, labeled=known & visible)
    fine = match_windows(ctx.adapter, windows, ctx.matcher.tau_f)
    l2, labels = loss_fn(fine, windows)
    points_a = queries.reshape(-1, 2)
    points_b = fine['points_b'].reshape(-1, 2)
    target = windows['truth'].reshape(-1, 2)
    valid = (windows['labeled'].reshape(-1) & torch.isfinite(points_b).all(-1)
             & (points_b >= 0).all(-1) & (points_b <= 511).all(-1))
    keep = valid.nonzero().flatten()
    if c['refine']['enabled'] and len(keep):
        keep = keep[torch.randperm(len(keep), generator=generator, device=keep.device)[:c['refine_points']]]
    if c['refine']['enabled']:
        refined = ctx.refiner(d2, points_a[keep], points_b[keep])
        lref, detail = local_refinement_loss(
            points_b[keep], refined, target[keep],
            torch.ones_like(keep, dtype=torch.bool), harm_weight=c['refine']['harm_weight'])
    else:
        lref = l2 * 0
        detail = dict(localization=l2.detach() * 0, harm=l2.detach() * 0)
    w = c['loss_weights']
    total = w['d8_global'] * l8g + w['d8_local'] * l8l + w['d2'] * l2 + w['refine'] * lref
    metrics = dict(d8_global=float(l8g.detach()), d8_local=float(l8l.detach()), d2=float(l2.detach()),
                   refine=float(lref.detach()), localization=float(detail['localization'].detach()),
                   harm=float(detail['harm'].detach()), labels=int(labels),
                   refine_points=int(len(keep)), windows=int(take))
    return total, metrics


def pair_view(shared, index):
    """Zero-copy single-pair view of a batched shared output.

    ``pyramid[scale]`` is Bx2xCxHxW and the flow fields are Bx...x32x32, while the
    matcher code addresses pair 0 only. Slicing keeps the validated per-pair code
    path untouched; only the backbone forward is batched.
    """
    view = dict(shared)
    view['pyramid'] = {scale: tensor[index:index + 1] for scale, tensor in shared['pyramid'].items()}
    for key in ('flow16_norm', 'flow16_target_norm', 'flow16_matchability', 'flow16_valid'):
        value = shared.get(key)
        if isinstance(value, torch.Tensor) and value.shape[0] > 1:
            view[key] = value[index:index + 1]
    return view


def merge_metrics(rows):
    """Average the numeric per-pair metrics so logging keeps the old schema."""
    numeric = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (int, float))})
    merged = {}
    for key in numeric:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        merged[key] = float(sum(values) / len(values)) if values else 0.
    return merged


# ---------------------------------------------------------------- evaluation

def empty_counts():
    return {key: 0 for key in METRIC_KEYS}


def add_counts(target, value):
    for key in METRIC_KEYS:
        target[key] += value[key]


def counts_summary(counts):
    visible = max(1, counts['visible'])
    total = max(1, counts['n'])
    summary = dict(counts)
    summary['epe'] = counts['error_sum'] / visible
    summary['p3_visible'] = counts['correct3'] / visible
    summary.update({f'p{t}': counts[f'correct{t}'] / total for t in (1, 3, 5, 10)})
    return summary


@torch.no_grad()
def evaluate(ctx, dataset, parallel, workers, limit=0):
    """Full cascaded validation, sharded by rank; counts are exact integers."""
    ctx.eval()
    total = len(dataset) if not limit else min(limit, len(dataset))
    indices = list(range(parallel.rank, total, parallel.world))
    pre, post = empty_counts(), empty_counts()
    worse = points = 0
    started = time.perf_counter()
    loader = DataLoader(Subset(dataset, indices), batch_size=1, num_workers=workers,
                        collate_fn=singleton)
    for index, batch in enumerate(loader):
        shared = ctx.shared(batch['images'].to(parallel.device), pyramid_scales=(8, 4, 2))
        gt = batch['geometry'].to(parallel.device)
        windows = extract_windows(shared, ctx.matcher, ctx.max_parents)
        fine = match_windows(ctx.adapter, windows, ctx.matcher.tau_f)
        points_a = windows['source'].reshape(-1, 2)
        points_b = fine['points_b'].reshape(-1, 2)
        before, error_before, good = measure(points_a, points_b, gt)
        add_counts(pre, before)
        if ctx.config['refine']['enabled']:
            refined = ctx.refiner(shared['pyramid'][2][0].float(), points_a, points_b)
        else:
            refined = points_b
        after, error_after, _ = measure(points_a, refined, gt)
        add_counts(post, after)
        worse += int((((error_after - error_before) > 0) & good).sum())
        points += int(good.sum())
        if (index + 1) % 200 == 0:
            print(json.dumps(dict(event='evaluation', done=index + 1, total=len(indices))), flush=True)
    payload = dict(pre=pre, post=post, worse=worse, points=points, pairs=len(indices),
                   seconds=time.perf_counter() - started)
    merged_pre, merged_post = empty_counts(), empty_counts()
    worse = points = pairs = 0
    slowest = 0.
    for part in parallel.gather(payload):
        add_counts(merged_pre, part['pre'])
        add_counts(merged_post, part['post'])
        worse += part['worse']
        points += part['points']
        pairs += part['pairs']
        slowest = max(slowest, part['seconds'])
    summary = dict(pairs=int(pairs), points=int(points), worse=int(worse),
                   worse_fraction=(worse / points) if points else None,
                   pre_refine=counts_summary(merged_pre), post_refine=counts_summary(merged_post),
                   seconds=slowest, scope='full_val' if not limit else 'fixed_val_subset')
    summary['delta_p3_visible'] = summary['post_refine']['p3_visible'] - summary['pre_refine']['p3_visible']
    ctx.train()
    return summary


# ---------------------------------------------------------------- planning

def load_split(config, name):
    root = Path(config['rgb_root']) / name
    rows = [json.loads(line) for line in (root / 'pairs.jsonl').open() if line.strip()]
    return GeometryPairDataset(root, rows)


def plan_steps(config, pairs, world):
    # Integer division on purpose: every rank must run exactly the same number of
    # optimizer steps, otherwise the collectives desynchronise at the epoch tail.
    per_epoch = pairs // (world * config['accumulation'])
    return per_epoch, per_epoch * config['epochs']


def multiplier_factory(config, total_steps):
    warmup, floor = config['warmup_steps'], config['cosine_min_ratio']

    def multiplier(step):
        if warmup and step < warmup:
            return .1 + .9 * step / warmup
        progress = min(1., max(0., (step - warmup) / max(1, total_steps - warmup)))
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    return multiplier


def shard_order(pairs, epoch, seed, parallel, accumulation):
    """Rank-strided order with a shared, exactly divisible length.

    Truncating to a multiple of ``world * accumulation`` keeps every rank's shard
    length a multiple of ``accumulation``, so all ranks execute the same number of
    optimizer steps and the gradient all-reduce never desynchronises. At most
    ``world * accumulation - 1`` pairs per epoch are unused.
    """
    usable = (pairs // (parallel.world * accumulation)) * (parallel.world * accumulation)
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(pairs, generator=generator).tolist()[:usable]
    return order[parallel.rank::parallel.world]


def epoch_loader(dataset, order, cursor, workers):
    return DataLoader(dataset, batch_size=1, sampler=order[cursor:], num_workers=workers,
                      collate_fn=singleton, pin_memory=True,
                      prefetch_factor=(4 if workers else None), persistent_workers=False)


def make_optimizer(ctx):
    groups = ctx.groups()
    for group in groups:
        if not group['params']:
            raise ValueError(f'Empty parameter group: {group["name"]}')
    optimizer = torch.optim.AdamW([dict(name=g['name'], params=g['params'], lr=g['lr']) for g in groups],
                                  weight_decay=ctx.config['weight_decay'])
    return optimizer, groups


def save(ctx, path, optimizer, scheduler, metadata, epoch, step, cursor, best, best_epoch, score):
    torch.save(dict(**ctx.snapshot(), config=ctx.config, matcher_config=asdict(ctx.matcher.config),
                    stage='joint', epoch=epoch, step=step, cursor=cursor, best=best,
                    best_epoch=best_epoch, score=score, optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(), python_rng=random.getstate(),
                    frozen_flow_sha256=sha256_file(ctx.config['frozen_flow_checkpoint']),
                    metadata=metadata), path)


# ---------------------------------------------------------------- audit

def audit(ctx, parallel, params, frozen_digest):
    """G1 gates: no frozen gradients, identical replicas, refiner starts as identity."""
    bad = [name for name, module in ctx.frozen_modules()
           for parameter in module.parameters() if parameter.grad is not None]
    if bad:
        raise RuntimeError(f'Frozen parameters received gradients: {bad[:5]}')
    if state_digest(ctx.frozen_modules()) != frozen_digest:
        raise RuntimeError('Frozen Flow0 weights changed during the audit steps')
    digests = parallel.gather(parameter_digest(params))
    if len(set(digests)) != 1:
        raise RuntimeError(f'Replicas diverged after the audit steps: {digests}')
    no_grad = [g['name'] for g in ctx.groups() if any(p.grad is None for p in g['params'])]
    if parallel.main:
        print(json.dumps(dict(event='audit', status='pass', replicas=len(digests),
                              parameter_digest=digests[0][:16], frozen_digest=frozen_digest[:16],
                              groups_without_grad=no_grad,
                              peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)), flush=True)


# ---------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/mhi_v2/joint_rgb512_ddp20.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--limit-val', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--accumulation', type=int,
                        help='Override config accumulation (single-card parity runs)')
    parser.add_argument('--batch-forward', dest='batch_forward', type=int, choices=(0, 1),
                        help='1: one B=accumulation backbone forward per step; 0: per-pair forwards')
    parser.add_argument('--audit', action='store_true', help='run G1 checks, then exit')
    parser.add_argument('--audit-steps', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    config = read_config(args.config, ROOT)
    if args.epochs:
        config['epochs'] = args.epochs
    if args.accumulation:
        config['accumulation'] = args.accumulation
    if args.batch_forward is not None:
        config['batch_forward'] = bool(args.batch_forward)
    parallel = Parallel()
    output = Path(args.output or config['output']).resolve()
    train, val = load_split(config, 'train'), load_split(config, 'val')
    if args.limit_train:
        train = Subset(train, list(range(min(args.limit_train, len(train)))))
    pairs = len(train)
    per_epoch, total_steps = plan_steps(config, pairs, parallel.world)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    payload = dict(experiment='mhi_v2_joint_ddp', world_size=parallel.world,
                   accumulation=config['accumulation'],
                   global_batch=parallel.world * config['accumulation'],
                   train_pairs=pairs, val_pairs=len(val), epochs=config['epochs'],
                   steps_per_epoch=per_epoch, total_steps=total_steps,
                   frozen=['DINOv3', 'MVT', 'Flow0 head'],
                   trained=['VGG', 'CGMDP' + ('+DeepD2' if config['d2_refinement'] != 'none' else ''),
                            'D8/D2 temperatures', 'Pair modulation', 'QRRU refiner'],
                   frozen_flow_checkpoint=config['frozen_flow_checkpoint'])
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    output.mkdir(parents=True, exist_ok=True)
    lock = None
    if parallel.main:
        lock = (output / '.training.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / 'pid.txt').write_text(f'{os.getpid()}\n')
        write_json(output / 'run.json', dict(
            **payload, started_at=timestamp(),
            config={k: v for k, v in config.items() if k != 'matcher_config'},
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        write_json(output / 'status.json', dict(
            state='initializing', epoch=0, epochs=config['epochs'], step=0,
            total_steps=total_steps, cursor=0, epoch_pairs=0,
            best=-1., best_epoch=0, updated_at=timestamp()))
        print(json.dumps(dict(event='start', **payload), ensure_ascii=False), flush=True)

    device = parallel.device
    ctx = JointContext(config, device)
    optimizer, groups = make_optimizer(ctx)
    params = [p for group in groups for p in group['params']]
    sync = GradientSync(params, device)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier_factory(config, total_steps))
    frozen_digest = state_digest(ctx.frozen_modules())
    metadata = dict(experiment='mhi_v2_joint_ddp', config=config, world_size=parallel.world,
                    accumulation=config['accumulation'], global_batch=parallel.world * config['accumulation'],
                    matcher=asdict(ctx.matcher.config), frozen_digest=frozen_digest,
                    frozen_flow_sha256=sha256_file(config['frozen_flow_checkpoint']),
                    total_steps=total_steps, steps_per_epoch=per_epoch, train_pairs=pairs,
                    val_pairs=len(val), torch_version=str(torch.__version__), started_at=timestamp())
    if parallel.main:
        print(json.dumps(dict(event='parameters',
                              **{g['name']: sum(p.numel() for p in g['params']) for g in groups},
                              total=sum(p.numel() for p in params), sync_elements=sync.elements,
                              frozen_digest=frozen_digest[:16])), flush=True)

    epoch, cursor, step = 0, 0, 0
    best, best_epoch = -1., 0
    latest = output / 'latest.pt'
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError(latest)
        state = torch.load(latest, map_location=device, weights_only=False)
        if state['metadata']['world_size'] != parallel.world:
            raise ValueError('Resume world_size differs from the checkpoint')
        if state['metadata']['config']['epochs'] != config['epochs']:
            raise ValueError('Resume epochs differs from the checkpoint')
        ctx.restore(state)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch, cursor, step = state['epoch'], state['cursor'], state['step']
        best, best_epoch = state['best'], state['best_epoch']
        torch.set_rng_state(state['rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        random.setstate(state['python_rng'])
        del state
        if parallel.main:
            print(json.dumps(dict(event='resumed', epoch=epoch, cursor=cursor, step=step,
                                  best=best)), flush=True)

    stop = {'requested': False}

    def request_stop(signum, frame):
        stop['requested'] = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)

    def check_frozen(tag):
        if state_digest(ctx.frozen_modules()) != frozen_digest:
            raise RuntimeError(f'Frozen Flow0 weights changed: {tag}')

    if step == 0 and not args.audit and not args.limit_train:
        summary = evaluate(ctx, val, parallel, config['workers'])
        if parallel.main:
            write_json(output / 'validation' / 'initial.json', summary)
            print(json.dumps(dict(event='initial_validation', **summary)), flush=True)
        parallel.barrier()

    history = output / 'history.jsonl'
    audit_done = False
    while epoch < config['epochs'] and step < total_steps and not stop['requested']:
        epoch += 1
        order = shard_order(pairs, epoch, config['seed'], parallel, config['accumulation'])
        if cursor >= len(order):
            cursor = 0
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        seen, loss_sum, last_metrics = 0, 0., {}
        gradient = 0.
        generator = torch.Generator(device=device).manual_seed(
            config['seed'] * 7919 + epoch * 131 + parallel.rank)
        pending = []

        def run_step(chunk, seen_before):
            nonlocal cursor, step, loss_sum, seen, last_metrics, gradient
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if config.get('batch_forward', True):
                # One backbone forward and one backward for the whole step: the
                # four pairs stay the same accumulation group, only the shape seen
                # by DINO/VGG/CGMDP changes from (1,2,3,512,512) to (4,2,3,512,512).
                images = torch.cat([item['images'] for item in chunk], 0).to(device)
                shared = ctx.shared(images, pyramid_scales=(8, 4, 2))
                losses, rows = [], []
                for index, item in enumerate(chunk):
                    loss, metrics = pair_terms(ctx, pair_view(shared, index),
                                               item['geometry'].to(device), generator)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite joint loss: {metrics}')
                    losses.append(loss)
                    rows.append(metrics)
                loss = torch.stack(losses).mean()
                loss.backward()
                last_metrics = merge_metrics(rows)
                loss_sum += float(loss.detach()) * len(chunk)
                seen += len(chunk)
            else:
                for item in chunk:
                    shared = ctx.shared(item['images'].to(device), pyramid_scales=(8, 4, 2))
                    gt = item['geometry'].to(device)
                    loss, last_metrics = pair_terms(ctx, shared, gt, generator)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite joint loss: {last_metrics}')
                    (loss / len(chunk)).backward()
                    loss_sum += float(loss.detach())
                    seen += 1
            cursor += len(chunk)
            sync.average(parallel.world)
            gradient = torch.nn.utils.clip_grad_norm_(params, config['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if (parallel.main and config.get('save_every_steps')
                    and step % config['save_every_steps'] == 0 and step < total_steps):
                save(ctx, latest, optimizer, scheduler, metadata, epoch, step, cursor,
                     best, best_epoch, best)
            if parallel.main and (step % 5 == 0 or step == 1 or cursor >= len(order)):
                row = dict(event='train', step=step, total_steps=total_steps, epoch=epoch,
                           epochs=config['epochs'], cursor=cursor, epoch_pairs=len(order),
                           pairs=seen, loss=loss_sum / max(1, seen), gradient_norm=float(gradient),
                           learning_rate=optimizer.param_groups[0]['lr'],
                           seconds_per_step=time.perf_counter() - step_started,
                           updated_at=timestamp(), metrics=last_metrics)
                append_jsonl(output / 'train.jsonl', row)
                write_json(output / 'status.json', dict(
                    state='training', epoch=epoch, epochs=config['epochs'], step=step,
                    total_steps=total_steps, cursor=cursor, epoch_pairs=len(order),
                    loss=row['loss'], seconds_per_step=row['seconds_per_step'],
                    metrics=last_metrics, best=best, best_epoch=best_epoch,
                    last_validation=read_json(output / 'validation' / 'last.json'),
                    updated_at=timestamp()))
                print(json.dumps(row, allow_nan=False), flush=True)

        loader = epoch_loader(train, order, cursor, config['workers'])
        for batch in loader:
            pending.append(batch)
            if len(pending) < config['accumulation']:
                continue
            chunk, pending = pending, []
            run_step(chunk, seen)
            if stop['requested'] or step >= total_steps:
                break
            if args.audit and step >= args.audit_steps:
                break
        if pending and not stop['requested'] and step < total_steps:
            run_step(pending, seen)
        if args.audit and step >= args.audit_steps:
            audit_done = True
            break
        if stop['requested']:
            break
        check_frozen(f'epoch {epoch}')
        summary = evaluate(ctx, val, parallel, config['workers'])
        if parallel.main:
            score = summary['post_refine']['p3_visible']
            improved = score > best
            if improved:
                best, best_epoch = score, epoch
            write_json(output / 'validation' / 'last.json', summary)
            write_json(output / 'validation' / f'epoch_{epoch:03d}.json', summary)
            if improved:
                save(ctx, output / 'best.pt', optimizer, scheduler, metadata, epoch, step, 0, best, best_epoch, score)
            save(ctx, output / f'epoch_{epoch:03d}.pt', optimizer, scheduler, metadata, epoch, step, 0, best, best_epoch, score)
            save(ctx, latest, optimizer, scheduler, metadata, epoch, step, 0, best, best_epoch, score)
            record = dict(epoch=epoch, step=step, pairs=seen, loss=loss_sum / max(1, seen),
                          seconds=time.perf_counter() - started, score=score, best=best,
                          best_epoch=best_epoch, improved=improved, validation=summary,
                          lr={g['name']: g['lr'] for g in optimizer.param_groups},
                          peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
            append_jsonl(history, record)
            print(json.dumps(dict(event='epoch', **record), allow_nan=False), flush=True)
        cursor = 0
        parallel.barrier()
    if audit_done:
        pass

    if args.audit:
        audit(ctx, parallel, params, frozen_digest)
        if parallel.enabled:
            dist.destroy_process_group()
        return 0

    if parallel.main:
        save(ctx, latest, optimizer, scheduler, metadata, epoch, step, cursor, best, best_epoch, best)
        write_json(output / 'status.json', dict(state='completed' if step >= total_steps else 'stopped',
                                                epoch=epoch, epochs=config['epochs'], step=step,
                                                total_steps=total_steps, best=best,
                                                best_epoch=best_epoch, updated_at=timestamp()))
        print(json.dumps(dict(event='finish', step=step, total_steps=total_steps, best=best,
                              best_epoch=best_epoch)), flush=True)
    parallel.barrier()
    if parallel.enabled:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
