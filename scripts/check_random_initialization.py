"""Two-step gradient audit for the pretrained-removal experiment."""
import json
from pathlib import Path
import torch
from mhinet.engine.train import TrainConfig,_seed_everything,_safe_train_dataset
from mhinet.config import RuntimePaths
from mhinet.models.model import build_model
from mhinet.models.initialization import initialize_ablation
from mhinet.engine.losses import sequence_corner_l1
from mhinet.engine.ghim_losses import ghim_loss
from mhinet.ops.correlation import HGuidedLocalCorrelation

torch.set_num_threads(1)
cfg=TrainConfig.from_json('configs/train_tier1_one_epoch_b.json')
r=RuntimePaths.from_json('configs/runtime_paths.quadrant.server.json')
_seed_everything(0)
model,_=build_model(r);model.set_training_phase('frozen_dino')
initialization=initialize_ablation(model,cfg.raw['initialization'],0)
for m in model.modules():
    if isinstance(m,HGuidedLocalCorrelation):m.activation_checkpoint_training=False
model.train()
groups=model.optimizer_group_spec()
for g in groups:g['lr']=cfg.raw['learning_rates'][g['name']]/650
opt=torch.optim.AdamW(groups,weight_decay=1e-4)
data,_=_safe_train_dataset(r,max_pairs=2,tier=1);data.include_overlap_mask=True
rows=[]
for step in range(2):
    s=data[step];opt.zero_grad(set_to_none=True)
    out=model(s['images'][None].to(r.device));gt=s['H_gt_norm'][None].to(r.device)
    loss=sequence_corner_l1(out,gt)['loss']+ghim_loss(out['ghim_outputs'],gt,s['mask_A_overlap'][None].to(r.device))['total']
    assert torch.isfinite(loss)
    loss.backward()
    norms={}
    for name,params in model._all_parameter_groups().items():
        grads=[p.grad for p in params if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads)
        norms[name]=sum(float(g.float().square().sum()) for g in grads)**.5
    assert norms['dino']==0 and norms['mvt']>0 and norms['stage1_head_parameters']>0
    row={'step':step+1,'loss':float(loss.detach()),'H0_valid':bool(out['stage1_valid'][0]),'gradient_norms':norms}
    rows.append(row);print(row,flush=True)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
Path('artifacts/random_initialization_audit.json').write_text(json.dumps({'initialization':initialization,'steps':rows},indent=2)+'\n')
