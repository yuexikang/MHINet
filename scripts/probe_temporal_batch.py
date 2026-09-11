"""Isolated real-data EBS8 training throughput/gradient probe; no checkpoints."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from mhinet.config import RuntimePaths, sha256_file
from mhinet.models.model import build_model
from mhinet.engine.train import _safe_train_dataset, _seed_everything, warmup_cosine_factor
from mhinet.engine.losses import sequence_corner_l1
from mhinet.engine.ghim_losses import ghim_loss
from mhinet.ops.geometry import image_corners, safe_project_points, normalized_to_pixel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, choices=(1, 2), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    _seed_everything(0)
    runtime = RuntimePaths.from_json('configs/runtime_paths.temporal4.server.json')
    model, build = build_model(runtime)
    model.set_training_phase('frozen_dino_mvt')
    model.train()
    dataset, _ = _safe_train_dataset(runtime, max_pairs=16)
    dataset.include_overlap_mask = True
    config_path = Path('configs/train_frozen_dino_mvt_temporal4_ebs8.json')
    config = json.loads(config_path.read_text())
    groups = model.optimizer_group_spec()
    for group in groups:
        group['lr'] = config['learning_rates'][group['name']]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        warmup_cosine_factor(step, total_steps=10000, warmup_steps=500, minimum_ratio=.1))
    bs = args.batch_size
    corners = image_corners((784, 784), normalized=True, device=runtime.device)[None]
    initial = []
    # Before any optimizer update, compare identical input pairs across BS runs.
    with torch.no_grad():
        for start in range(0, 16, bs):
            samples = [dataset[i] for i in range(start, start+bs)]
            out = model(torch.stack([s['images'] for s in samples]).to(runtime.device))
            q, valid, _ = safe_project_points(out['H0_norm'], corners.expand(bs, -1, -1))
            initial.extend([{'pair_id': s['pair_id'], 'H0_corners_px': v,
                             'valid': bool(ok)} for s,v,ok in zip(samples,
                normalized_to_pixel(q, (784,784)).cpu().tolist(), out['stage1_valid'].cpu().tolist())])
            del out
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for step in range(9):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        loss_parts = dict(geo=0., mat=0., cls=0., H=0., sequence=0.)
        valid_pairs = 0
        for offset in range(0, 8, bs):
            indices = [(step*8+offset+j)%16 for j in range(bs)]
            samples = [dataset[i] for i in indices]
            images = torch.stack([s['images'] for s in samples]).to(runtime.device)
            gt = torch.stack([s['H_gt_norm'] for s in samples]).to(runtime.device)
            masks = torch.stack([s['mask_A_overlap'] for s in samples]).to(runtime.device)
            outputs = model(images)
            assert outputs['shared_call_counts']['dedode_scale1'] == 0
            assert outputs['shared_call_counts']['dino'] == outputs['shared_call_counts']['mvt'] == 1
            seq = sequence_corner_l1(outputs, gt)
            aux = ghim_loss(outputs['ghim_outputs'], gt, masks)
            loss = seq['loss'] + aux['total']
            assert torch.isfinite(loss)
            (loss * bs / 8).backward()
            total += float(loss.detach())*bs/8
            valid_pairs += seq['valid_pairs']
            for key in ('geo', 'mat', 'cls', 'H'):
                loss_parts[key] += float(aux[key].detach())*bs/8
            loss_parts['sequence'] += float(seq['loss'].detach())*bs/8
            del outputs, loss, seq, aux, images, gt, masks
        # This synchronization/audit overhead is present in both runs.
        gradients = {}
        for name, parameters in model._all_parameter_groups().items():
            grads = [p.grad for p in parameters if p.grad is not None]
            if name in ('dino', 'mvt'):
                assert not grads
            gradients[name] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        rows.append({'step': step+1, 'measured': step>=3, 'seconds': time.perf_counter()-start_time,
                     'loss': total, 'parts': loss_parts, 'valid_pairs': valid_pairs,
                     'preclip_norm': float(norm), 'gradient_norms': gradients})
        print(json.dumps(rows[-1]), flush=True)
    measured = [r['seconds'] for r in rows if r['measured']]
    result = {'status': 'completed', 'batch_size': bs, 'accumulation': 8//bs, 'effective_batch': 8,
              'warmup_steps': 3, 'measured_steps': 6, 'mean_step_seconds': statistics.mean(measured),
              'median_step_seconds': statistics.median(measured), 'pairs_per_second': 8/statistics.mean(measured),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'initial_H0': initial, 'rows': rows, 'config_sha256': sha256_file(config_path),
              'manifest_sha256': sha256_file(runtime.data_root/'train/pairs.jsonl'),
              'architecture_sha256': build['architecture_sha256'], 'provider': build['provider'],
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'scope': '16 real train pairs cycling; includes image reads/stack/H2D/full forward/loss/backward/optimizer and gradient audit; not convergence validation'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
