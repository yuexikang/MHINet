"""Two real-data optimizer steps; no formal training checkpoint is produced."""
import json
from pathlib import Path
import torch
from mhinet.config import RuntimePaths
from mhinet.models.model import build_model
from mhinet.engine.train import _safe_train_dataset, warmup_cosine_factor
from mhinet.engine.losses import sequence_corner_l1
from mhinet.engine.ghim_losses import ghim_loss

torch.manual_seed(0)
torch.set_num_threads(1)
runtime = RuntimePaths.from_json('configs/runtime_paths.server.json')
model, build = build_model(runtime)
report = model.set_training_phase('frozen_dino_mvt')
model.train()
provider = model.feature_provider
assert not provider.dino.training and not provider.mvt.training
assert provider.stage1_head.training and provider.dedode.training
dataset, _ = _safe_train_dataset(runtime, max_pairs=2)
dataset.include_overlap_mask = True
optimizer = torch.optim.AdamW(model.optimizer_group_spec(), weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
    warmup_cosine_factor(step, total_steps=10000, warmup_steps=500, minimum_ratio=.1))
rows = []
for step in (1, 2):
    sample = dataset[0]
    image = sample['images'][None].to(runtime.device)
    gt = sample['H_gt_norm'][None].to(runtime.device)
    mask = sample['mask_A_overlap'][None].to(runtime.device)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(image)
    assert outputs['shared_call_counts']['dino'] == 1
    assert outputs['shared_call_counts']['mvt'] == 1
    assert outputs['shared_call_counts']['dedode_scale1'] == 0
    assert outputs['H0_norm'].requires_grad
    ghim = ghim_loss(outputs['ghim_outputs'], gt, mask)
    sequence = sequence_corner_l1(outputs, gt)['loss']
    loss = sequence + ghim['total']
    loss.backward()
    norms = {}
    for name, parameters in model._all_parameter_groups().items():
        grads = [p.grad for p in parameters if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads), name
        norms[name] = sum(float(g.float().square().sum()) for g in grads) ** .5
        if name in ('dino', 'mvt'):
            assert not grads, name
        elif name == 'stage1_head_parameters' or step == 2:
            assert norms[name] > 0, name
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    rows.append({'step': step, 'loss_total': float(loss.detach()),
                 'sequence_l1_px': float(sequence.detach()),
                 'ghim': {k: float(v.detach()) for k, v in ghim.items()},
                 'gradient_norms': norms, 'calls': outputs['shared_call_counts']})
    del outputs, loss, sequence, ghim
    print(json.dumps(rows[-1]), flush=True)
for group in report['groups'].values():
    group.pop('optimizer_parameter_ids', None)
result = {'status': 'passed', 'scope': 'two-step real train-pair smoke, not model accuracy validation',
          'torch': torch.__version__, 'cuda': torch.version.cuda,
          'parameter_report': report, 'steps': rows,
          'recipe': 'head LR 1e-6; formal 500-step warmup; BS1 diagnostic (formal accumulation 4)',
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'provider': build['provider']}
Path('artifacts/frozen_dino_mvt_two_step.json').write_text(json.dumps(result, indent=2) + '\n')
