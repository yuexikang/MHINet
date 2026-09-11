"""Same-weight, nonzero-projection GPU comparison of correlation checkpointing."""
import json
from pathlib import Path
import torch
from mhinet.config import RuntimePaths
from mhinet.models.model import build_model
from mhinet.engine.train import _safe_train_dataset, _seed_everything
from mhinet.engine.losses import sequence_corner_l1
from mhinet.engine.ghim_losses import ghim_loss
from mhinet.ops.correlation import HGuidedLocalCorrelation

torch.set_num_threads(1)
_seed_everything(0)
runtime = RuntimePaths.from_json('configs/runtime_paths.temporal4.server.json')
model, _ = build_model(runtime)
model.set_training_phase('frozen_dino_mvt')
model.train()
dataset, _ = _safe_train_dataset(runtime, max_pairs=1)
dataset.include_overlap_mask = True
sample = dataset[0]
images = sample['images'][None].to(runtime.device)
gt = sample['H_gt_norm'][None].to(runtime.device)
mask = sample['mask_A_overlap'][None].to(runtime.device)


def forward(enabled):
    for module in model.modules():
        if isinstance(module, HGuidedLocalCorrelation):
            module.activation_checkpoint_training = enabled
    out = model(images)
    loss = sequence_corner_l1(out, gt)['loss'] + ghim_loss(out['ghim_outputs'], gt, mask)['total']
    return loss, out


# One update makes the zero-initialized projection nonzero, exposing upstream gradients.
optimizer = torch.optim.AdamW(model.optimizer_group_spec(), weight_decay=1e-4)
for group in optimizer.param_groups:
    group['lr'] *= .002
loss, out = forward(True)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
optimizer.step()
del loss, out
records = []
for enabled in (True, False):
    optimizer.zero_grad(set_to_none=True)
    loss, out = forward(enabled)
    loss.backward()
    records.append({'loss': float(loss.detach()),
                    'H': torch.cat((out['H0_norm'][:,None],out['H_updates_norm']),1).detach().cpu(),
                    'grads': {n:p.grad.detach().float().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}})
    del loss, out
a,b = records
assert a['grads'].keys() == b['grads'].keys()
failed=[];max_abs=0.;diff2=0.;ref2=0.
for name,x in a['grads'].items():
    y=b['grads'][name]
    max_abs=max(max_abs,float((x-y).abs().max()))
    diff2+=float((x-y).square().sum());ref2+=float(x.square().sum())
    try:
        torch.testing.assert_close(x,y,rtol=1e-3,atol=1e-5)
    except AssertionError:
        failed.append(name)
result={'status':'passed' if not failed and abs(a['loss']-b['loss'])<1e-6 and torch.equal(a['H'],b['H']) else 'failed',
        'scope':'same weights after one optimizer step; one real train pair; all parameter gradients',
        'loss_on':a['loss'],'loss_off':b['loss'],'H_max_abs_diff':float((a['H']-b['H']).abs().max()),
        'gradient_max_abs_diff':max_abs,'gradient_relative_l2_diff':(diff2/max(ref2,1e-30))**.5,
        'gradient_tolerance':{'rtol':1e-3,'atol':1e-5},'failed_parameter_names':failed,
        'gradient_tensor_count':len(a['grads'])}
Path('artifacts/temporal4_checkpoint_same_weight_gradients.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
