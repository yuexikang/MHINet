"""Restore real optimizer/scheduler/RNG states without an optimizer step or training."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from mhinet.v2.pipeline import load_mhi_v2
from mhinet.v2.pair_matching.training import restore
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--assets-root',type=Path,required=True)
p.add_argument('--runtime',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(2)
model=load_mhi_v2(device='cpu',runtime=json.loads(a.runtime.read_text()),assets_root=a.assets_root)
modules=dict(mvt=model.shared.mvt,head=model.shared.stage1_head,vgg=model.shared.vgg,dedode=model.shared.dedode,matcher=model.matcher,adapter=model.adapter)
report={}
for stage in ('flow','d2'):
 state=torch.load(a.assets_root/f'outputs/full_rgb_d2_10epoch_20261028/{stage}/latest.pt',map_location='cpu',weights_only=False,mmap=True)
 restore(state,model.shared,model.matcher,model.adapter,stage)
 flow=stage=='flow'
 model.shared.set_training_groups(mvt=flow,ghim_head=flow,vgg=not flow,dedode=not flow)
 model.matcher.requires_grad_(not flow);model.adapter.requires_grad_(not flow)
 rates=state['config']['flow_lr' if flow else 'd2_lr']
 groups=[dict(name=k,params=[p for p in modules[k].parameters() if p.requires_grad],lr=lr) for k,lr in rates.items()]
 optimizer=torch.optim.AdamW(groups,weight_decay=state['config']['weight_decay'])
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
 print('groups',stage,[(g['name'],len(g['params'])) for g in groups],[(g['name'],len(g['params'])) for g in state['optimizer']['param_groups']],flush=True)
 optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
 actual=optimizer.state_dict()
 assert actual['param_groups']==state['optimizer']['param_groups']
 assert scheduler.state_dict()==state['scheduler']
 for group in optimizer.param_groups:
  for param in group['params']:
   values=optimizer.state.get(param,{})
   for key in ('exp_avg','exp_avg_sq'):
    if key in values:assert values[key].shape==param.shape
 for key,values in state['optimizer']['state'].items():
  for name,value in values.items():
   if isinstance(value,torch.Tensor):torch.testing.assert_close(actual['state'][key][name],value,rtol=0,atol=0)
 torch.set_rng_state(state['rng']);random.setstate(state['python_rng'])
 assert torch.equal(torch.get_rng_state(),state['rng'])
 assert random.getstate()==state['python_rng']
 report[stage]=dict(epoch=state['epoch'],scheduler_epoch=scheduler.last_epoch,
                   optimizer_groups={g['name']:len(g['params']) for g in optimizer.param_groups},
                   optimizer_tensor_states=len(actual['state']),cpu_rng_restored=True,python_rng_restored=True,
                   cuda_rng_checkpoint_bytes=state['cuda_rng'].numel(),optimizer_steps_executed=0)
 print(stage,report[stage],flush=True)
 del state,optimizer,scheduler,actual
# Check actual legacy U0 tensors against the shared architecture without inference/training.
from mhinet.v2.frozen_locator import load_locator_weights
load_locator_weights(model.shared,a.assets_root/'weight/mhi_v2_u0.pt')
report['legacy_u0']='strict MVT/head load passed; original file untouched'
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
