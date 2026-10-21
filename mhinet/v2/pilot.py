"""Small Hangzhou QRRU experiments: plain configuration, logs and weights.

Run with ``python -m mhinet.v2.pilot --config configs/mhi_v2/hangzhou_pilot.json``.
The geographic transforms are approximate. Training uses a target-space dead
zone, and validation measures agreement with that reference, not GCP accuracy.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch.torch_version import TorchVersion

from mhinet.config import RuntimePaths
from mhinet.dataio.data import load_rgb_bicubic
from mhinet.ops.geometry import pixel_homography_to_normalized
from mhinet.downstream.supervision import coarse_labels, qrru_targets, to_uv
from mhinet.downstream.loma_reference.matching_utils import normalized_cell_centers, project_normalized
from .config import SemidenseConfigV2
from .feature_provider import build_v2_shared
from .semidense import SemidenseMatcherV2
from .training import SemidenseSystemV2


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def select_rows(root, split, per_group, seed):
    """Reservoir sampling across each entire G0..G3 group, not a sorted prefix."""
    rng = random.Random(seed)
    groups = {f"G{i}": [] for i in range(4)}
    counts = dict.fromkeys(groups, 0)
    with (root/"manifests"/f"{split}.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            group = row["G"]
            counts[group] += 1
            bucket = groups[group]
            if len(bucket) < per_group:
                bucket.append(row)
            else:
                index = rng.randrange(counts[group])
                if index < per_group:
                    bucket[index] = row
    if any(len(v) != per_group for v in groups.values()):
        raise ValueError(f"Insufficient {split} rows for each group")
    return [row for group in groups.values() for row in group]


def load_pair(root, row, device):
    if row['size_A'] != [512, 512] or row['size_B'] != [512, 512]:
        raise ValueError('Pilot requires stored 512 images')
    images = torch.stack([load_rgb_bicubic(root/row[f'image_{s}'], 512) for s in 'AB'])
    masks = []
    for side in 'AB':
        with Image.open(root/row[f'valid_{side}']) as image:
            masks.append(torch.from_numpy(np.array(image.convert('L'), copy=True)).float()[None]/255)
    H = pixel_homography_to_normalized(torch.tensor(row['H_A_to_B'], dtype=torch.float64),
                                      (512, 512), (512, 512)).float()
    return images[None].to(device), H.to(device), [m.to(device) for m in masks]


def load_system(config, device):
    runtime = replace(RuntimePaths.from_json(config['runtime']), device=device)
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(config['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    matcher_config = SemidenseConfigV2.from_dict(payload['metadata']['matcher'])
    matcher_config = replace(matcher_config, **config.get('matcher', {}))
    shared, _ = build_v2_shared(runtime, matcher_config.geometry_mode)
    system = SemidenseSystemV2(shared, SemidenseMatcherV2(matcher_config), 'qrru').to(device)
    system.load_state_dict(payload['model'])
    if config.get('qrru_weights'):
        weights = torch.load(config['qrru_weights'], map_location='cpu', weights_only=True)
        system.matcher.qrru.load_state_dict(weights['qrru'])
    return system.eval()


def metrics(result, H, masks):
    from mhinet.downstream.supervision import valid_points
    a, b = result['points_a'], result['points_b']
    truth, legal = project_normalized((a+.5)/256-1, H)
    error = ((truth+1)*256-.5-b).norm(dim=-1)
    valid = legal & valid_points(masks[0], (a+.5)/256-1) & valid_points(masks[1], (b+.5)/256-1)
    n = max(1, len(a))
    return dict(matches=len(a), pre1=float(((error <= 1) & valid).sum())/n,
                pre3=float(((error <= 3) & valid).sum())/n,
                rmse_valid=float(error[valid].square().mean().sqrt()) if bool(valid.any()) else None,
                success20_at3=bool(((error <= 3) & valid).sum() >= 20))


@torch.no_grad()
def evaluate(system, root, rows, device):
    system.eval()
    records = []
    for row in rows:
        images, H, masks = load_pair(root, row, device)
        result = system.matcher.infer(system(images))[0]
        records.append(dict(pair_id=row['pair_id'], group=row['G'], **metrics(result, H, masks)))
    return dict(pairs=records, mean=summarize(records))


def summarize(records):
    keys = ('matches', 'pre1', 'pre3', 'rmse_valid', 'success20_at3')
    return {k: float(np.mean([r[k] for r in records if r[k] is not None]))
            if any(r[k] is not None for r in records) else None for k in keys}


def weak_qrru_loss(system, features, H, masks, tolerance_px):
    """Half real candidates, half perturbed geographic centers; center-only loss.

    A radial dead zone avoids forcing errors below the supplied heuristic
    tolerance to zero. That tolerance is not a measured ground-truth bound.
    """
    qrru = system.matcher.qrru
    d2 = features['pyramid'][2][0].float()
    with torch.no_grad():
        prediction = system.matcher.infer(features, refine=False)[0]
        pa = (prediction['points_a']+.5)/2-.5
        pb = (prediction['points_b']+.5)/2-.5
        _, _, truth, visible = qrru_targets(pa, pb, H, *masks, (256, 256))
        eligible = (visible & ((pb-truth).norm(dim=-1) <= 6)).nonzero().flatten()
        keep = eligible[torch.randperm(len(eligible), device=d2.device)[:64]]
        pa, pb = pa[keep], pb[keep]
        real_count = len(pa)
        indices, _ = coarse_labels(H, *masks, (256, 256))
        indices = indices[torch.randperm(len(indices), device=d2.device)[:128-real_count]]
        grid = normalized_cell_centers(256, 256, device=d2.device).reshape(-1, 2)[indices]
        target, _ = project_normalized(grid, H)
        pa = torch.cat((pa, to_uv(grid, (256, 256))))
        pb = torch.cat((pb, to_uv(target, (256, 256))+torch.randn_like(target).mul(1.5).clamp(-3, 3)))
        _, _, truth, valid = qrru_targets(pa, pb, H, *masks, (256, 256))
    if not bool(valid.any()):
        raise ValueError('No valid weak QRRU labels')
    projected = qrru.project(d2)
    result = qrru(projected[0], projected[1], pa, pb, projected=True)
    error = (result['centers'][:, valid]-truth[valid]).norm(dim=-1)*2
    weights = error.new_tensor([.8**(len(error)-1-i) for i in range(len(error))])
    loss = ((error-tolerance_px).clamp_min(0).mean(-1)*weights).sum()/weights.sum()
    return loss, dict(initial_epe_px=float((pb[valid]-truth[valid]).norm(dim=-1).mean()*2),
                      final_epe_px=float(error[-1].detach().mean()), real_queries=real_count,
                      tolerance_px=tolerance_px)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = Path(config['output']); output.mkdir(parents=True, exist_ok=False)
    root = Path(config['dataset'])
    torch.set_num_threads(2)
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    train = select_rows(root, 'train', config['train_per_group'], config['seed'])
    val = select_rows(root, 'val', config['val_per_group'], config['seed'])
    if {r['anchor_block_id'] for r in train} & {r['anchor_block_id'] for r in val}:
        raise ValueError('Selected train/val spatial blocks overlap')
    write_json(output/'selection.json', dict(train=train, val=val))
    write_json(output/'config.json', config)
    system = load_system(config, args.device)
    write_json(output/'before.json', evaluate(system, root, val, args.device))
    optimizer = torch.optim.AdamW(system.matcher.qrru.parameters(), lr=config['learning_rate'])
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    order = list(range(len(train))); random.Random(config['seed']).shuffle(order)
    step = 0
    system.train()
    with (output/'train.jsonl').open('w') as stream:
        for step in range(1, config['steps']+1):
            row = train[order[(step-1) % len(order)]]
            images, H, masks = load_pair(root, row, args.device)
            start = time.perf_counter()
            features = system(images)
            optimizer.zero_grad(set_to_none=True)
            loss, record = weak_qrru_loss(system, features, H, masks, row['soft_radius_B_px'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pilot loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(system.matcher.qrru.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record.update(step=step, pair_id=row['pair_id'], loss=float(loss.detach()),
                          gradient_norm=float(grad), seconds=time.perf_counter()-start)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            if step % 16 == 0:
                print(json.dumps(record), flush=True)
            if stop[0]:
                break
    torch.save(dict(qrru=system.matcher.qrru.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, config=config), output/'qrru.pt')
    if not stop[0]:
        write_json(output/'after.json', evaluate(system, root, val, args.device))
    write_json(output/'status.json', dict(status='interrupted' if stop[0] else 'completed', steps=step))


if __name__ == '__main__':
    main()
