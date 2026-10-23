"""Matched-budget T1/T2 training on shared frozen-U0 feature caches."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .dataset import load_config, dump
from .evaluate import load_records
from .losses import existence_loss
from .nomatch_head import NoMatchHead


def train():
    torch.set_num_threads(4)
    config=load_config();out=Path(config['output'])
    if not json.loads((out/'stage_a.json').read_text())['train_head']:
        print('Phase A gate passed: no new head required.');return
    torch.manual_seed(config['seed'])
    rng=np.random.default_rng(config['seed'])
    rows=[json.loads(s) for s in (out/'pairs.jsonl').read_text().splitlines()]
    bases=[r for r in rows if r['split']=='train' and r['kind']=='partial']
    clean,texture,clean_state,texture_state=[],[],[],[]
    for row in bases:
        for direction in ('ab','ba'):
            for suffix,features,states in (('',clean,clean_state),('_texture',texture,texture_state)):
                r=torch.load(out/'cache'/f'{row["pair_id"]}{suffix}_{direction}.pt',weights_only=False,map_location='cpu')
                features.append(r['features']);states.append(torch.from_numpy(r['state']).long())
    clean,texture=torch.stack(clean),torch.stack(texture)
    clean_state,texture_state=torch.stack(clean_state),torch.stack(texture_state)
    cal=load_records('cal')
    cal_features=torch.stack([r['features'] for r in cal])
    cal_state=torch.stack([torch.from_numpy(r['state']).long() for r in cal])
    template=NoMatchHead()
    models={n:copy.deepcopy(template).to(d) for n,d in (('T1','cuda:1'),('T2','cuda:2'))}
    assert all(torch.equal(a.cpu(),b.cpu()) for a,b in zip(models['T1'].parameters(),models['T2'].parameters()))
    optimizers={n:torch.optim.AdamW(m.parameters(),lr=config['lr'],weight_decay=config['weight_decay']) for n,m in models.items()}
    best_loss=float('inf');best=None;stale=0;history=[];step_count=0
    for epoch in range(1,config['epochs']+1):
        order=rng.permutation(len(clean));training={n:[] for n in models}
        for m in models.values():m.train()
        for offset in range(0,len(order),config['batch_size']):
            indices=torch.from_numpy(order[offset:offset+config['batch_size']])
            # Same source pair budget. Alternate half the pairs each epoch;
            # both directions of a pair use the same clean/texture variant.
            use_texture=((indices//2+epoch)%2==0)
            for name,model in models.items():
                device=next(model.parameters()).device
                if name=='T1':x,y=clean[indices],clean_state[indices]
                else:
                    x=torch.where(use_texture[:,None,None],texture[indices],clean[indices])
                    y=torch.where(use_texture[:,None],texture_state[indices],clean_state[indices])
                optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True)
                loss=existence_loss(model(x.to(device)),y.to(device));loss.backward();optimizer.step()
                training[name].append(float(loss.detach()))
            step_count+=1
        validation={}
        for name,model in models.items():
            model.eval();device=next(model.parameters()).device;losses=[]
            with torch.no_grad():
                # Aggregate exact global positive/negative normalization across
                # calibration, independent of minibatch composition.
                logits=torch.cat([model(x.to(device)).cpu() for x in cal_features.split(config['batch_size'])])
                validation[name]=float(existence_loss(logits,cal_state))
        joint=float(np.mean(list(validation.values())))
        entry=dict(epoch=epoch,steps_each=step_count,train={n:float(np.mean(v)) for n,v in training.items()},calibration=validation,joint_calibration=joint)
        history.append(entry);print(json.dumps(entry),flush=True)
        if joint<best_loss:
            best_loss=joint;stale=0
            best={n:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for n,m in models.items()}
            best_epoch=epoch
        else:stale+=1
        dump(out/'train_history.json',history)
        if stale>=config['patience']:break
    for name in models:
        torch.save(dict(head=best[name],epoch=best_epoch,trained_epochs=epoch,steps_each=step_count,
            config=config,trainable_parameters=sum(p.numel() for p in models[name].parameters())),out/f'{name}.pt')
    dump(out/'training_summary.json',dict(best_epoch=best_epoch,trained_epochs=epoch,steps_each=step_count,
        trainable_parameters=sum(p.numel() for p in template.parameters()),initialization_identical=True,
        examples_per_epoch_each=len(clean),head_forward_examples_each=len(clean)*epoch,
        u0_updates=0,shared_cache='common frozen U0 FP32 forward cache, no U0 recomputation during head training',
        early_stop='joint calibration mean with common best epoch',last=history[-1]))


if __name__=='__main__':train()
