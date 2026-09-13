"""Real two-step full-training smoke plus isolated MVT loss-gradient routes."""
import json
from pathlib import Path
import torch
from mhinet.config import RuntimePaths, sha256_file
from mhinet.engine.train import TrainConfig, _safe_train_dataset, _seed_everything, warmup_cosine_factor
from mhinet.models.model import build_model
from mhinet.ops.correlation import HGuidedLocalCorrelation
from mhinet.engine.losses import sequence_corner_l1
from mhinet.engine.ghim_losses import ghim_loss

torch.set_num_threads(1)
_seed_everything(0)
cfg = TrainConfig.from_json('configs/train_frozen_dino_temporal4_ebs4_20k.json')
runtime = RuntimePaths.from_json('configs/runtime_paths.temporal4.server.json')
model, build = build_model(runtime)
report = model.set_training_phase(cfg.profile)
model.train()
for module in model.modules():
    if isinstance(module, HGuidedLocalCorrelation):
        module.activation_checkpoint_training = False
assert model.feature_provider.mvt.training
assert not model.feature_provider.dino.training
groups = model.optimizer_group_spec()
for group in groups:
    group['lr'] = cfg.raw['learning_rates'][group['name']]
optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
    warmup_cosine_factor(s, total_steps=20000, warmup_steps=1000, minimum_ratio=.1))
dataset, _ = _safe_train_dataset(runtime, max_pairs=1)
dataset.include_overlap_mask = True
sample = dataset[0]
images = sample['images'][None].to(runtime.device)
gt = sample['H_gt_norm'][None].to(runtime.device)
mask = sample['mask_A_overlap'][None].to(runtime.device)
shared = {}
hook = model.feature_provider.register_forward_hook(lambda m,a,o: shared.update(o))
rows = []
for step in (1,2):
    optimizer.zero_grad(set_to_none=True)
    out = model(images)
    seq = sequence_corner_l1(out, gt)['loss']
    aux = ghim_loss(out['ghim_outputs'], gt, mask)
    loss = seq + aux['total']
    routes = {}
    if step == 2:
        context = shared['context']
        h0 = shared['H0_norm']
        dh = torch.autograd.grad(seq, h0, retain_graph=True)[0]
        gh = torch.autograd.grad(h0, context, dh.detach(), retain_graph=True)[0]
        routes['MHIR_loss_via_H0_to_MVT_context'] = float(gh.detach().norm())
        del dh, gh
        desc = tuple(shared['pyramid'][s] for s in (8,4,2))
        dd = torch.autograd.grad(seq, desc, retain_graph=True)
        gd = torch.autograd.grad(desc, context, tuple(g.detach() for g in dd), retain_graph=True)[0]
        routes['MHIR_loss_via_CGMDP_to_MVT_context'] = float(gd.detach().norm())
        del dd, gd, desc, h0, context
        assert all(v > 0 for v in routes.values())
    loss.backward()
    norms = {}
    for name, params in model._all_parameter_groups().items():
        grads = [p.grad for p in params if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads), name
        norms[name] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt()) if grads else 0.
        if name == 'dino':
            assert not grads
        elif step == 2:
            assert norms[name] > 0, name
    assert out['shared_call_counts']['dedode_scale1'] == 0
    assert out['shared_call_counts']['dino'] == out['shared_call_counts']['mvt'] == 1
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    rows.append({'step':step, 'loss':float(loss.detach()), 'gradient_norms':norms, 'routes':routes})
    print(json.dumps(rows[-1]), flush=True)
    shared.clear()
    del out, loss, seq, aux
hook.remove()
for group in report['groups'].values():
    group.pop('optimizer_parameter_ids',None)
result = {'status':'passed', 'scope':'two real-data BS1 updates with branch audit; not long-run accuracy evidence',
          'parameter_report':report,'steps':rows,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
          'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'config_sha256':cfg.sha256,
          'manifest_sha256':sha256_file(runtime.data_root/'train/pairs.jsonl'),
          'provider':build['provider'],'torch':torch.__version__}
Path('artifacts/frozen_dino_two_step.json').write_text(json.dumps(result,indent=2)+'\n')
