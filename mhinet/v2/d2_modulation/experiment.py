"""Prepare frozen RGB windows, train self/pair adapters, evaluate held-out data."""
import argparse
import json
import random
import os
import time
from pathlib import Path
import torch
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.qrru_corners.experiment import selections, load_models
from .model import LocalPairModulation, extract_windows, match_windows

ROOT = Path('outputs/d2_modulation')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def load_record(path, device):
    r = torch.load(path, map_location='cpu', weights_only=False)
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in r.items()}


def splits():
    selection = selections()
    rgb, hz = selection['synthetic'], selection['hangzhou']
    return {name: (root, rows) for name, root, rows in (
        ('train', rgb['root'], rgb['train']), ('val', rgb['root'], rgb['val']),
        ('test', rgb['test_root'], rgb['test']), ('hangzhou_test', hz['root'], hz['test']))}


@torch.no_grad()
def prepare(args):
    torch.set_num_threads(2)
    model, matcher = load_models(args.device)
    matcher.requires_grad_(False).eval()
    model.requires_grad_(False).eval()
    selection = splits() if args.selection is None else {
        k: (v['root'], v['rows']) for k,v in json.loads(args.selection.read_text()).items()}
    reuse = {}
    if args.reuse_cache is not None:
        old = json.loads((args.reuse_cache/'data.json').read_text())
        for split, spec in old['splits'].items():
            for i, pair in enumerate(spec['pairs']):
                reuse[pair] = args.reuse_cache/'cache'/split/f'{i:04d}.pt'
    ids = [{r['pair_id'] for r in selection[k][1]} for k in ('train', 'val', 'test')]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]), 'RGB split overlap'
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(temperature=float(matcher.tau_f),
        flow_checkpoint='weight/mhi_v2_u0.pt', downstream_checkpoint='outputs/flow_d8_d2/training/best.pt',
        context_side=8, query_side=4, target_side=8, max_parents=128,
        feature_dtype='float32', selection=str(args.selection) if args.selection else 'qrru_corners.selections(seed42)',
        splits={k:dict(root=root, pairs=[r['pair_id'] for r in rows]) for k,(root,rows) in selection.items()})
    write_json(args.output/'data.json', metadata)
    # All geometry enters only cached labels, after the inference-only extraction.
    for name in (args.splits or list(selection)):
        root, rows = selection[name]
        out = args.output/'cache'/name
        out.mkdir(parents=True, exist_ok=True)
        dataset = GeometryPairDataset(root, rows)
        for i, row in enumerate(rows):
            destination = out/f'{i:04d}.pt'
            source = reuse.get(row['pair_id'])
            if source is not None and source.exists():
                if not destination.exists():
                    os.link(source, destination)
                continue
            item = dataset[i]
            shared = model(item['images'].to(args.device))
            windows = extract_windows(shared, matcher)
            gt = item['geometry'].to(args.device)
            if name != 'hangzhou_test' and not gt.exact:
                raise ValueError('RGB training/evaluation requires exact H')
            target, known, visible = gt.query((windows['source']+.5)/256-1)
            record = {**windows, 'truth': (target+1)*256-.5, 'labeled': known&visible,
                      'pair_id': item['pair_id'][0]}
            torch.save({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in record.items()}, out/f'{i:04d}.pt')
            if (i+1)%16==0 or i==len(rows)-1:
                print(f'cache {name} {i+1}/{len(rows)}', flush=True)
    print('PREPARE DONE', flush=True)


def exact_labels(record):
    """Four-neighbour soft targets; ignore queries outside the actual support."""
    c, truth = record['candidates'], record['truth']
    valid = record['valid_b']
    lo = c.masked_fill(~valid[...,None], torch.inf).amin(1)
    hi = c.masked_fill(~valid[...,None], -torch.inf).amax(1)
    covered = record['labeled'] & (truth>=lo[:,None]).all(-1) & (truth<=hi[:,None]).all(-1)
    delta = (c[:,None]-truth[:,:,None]).abs()
    weights = (1-delta/2).clamp_min(0).prod(-1)*valid[:,None]
    usable = covered & (weights.sum(-1)>0)
    weights = weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
    return weights, usable


def loss_fn(result, record):
    weights, usable = exact_labels(record)
    if not usable.any():
        return result['logp'].sum()*0, 0
    # All 64 candidates compete. Dual normalization also learns competition across queries.
    loss = -(weights[usable]*result['logp'][usable]).sum(-1).mean()
    return loss, int(usable.sum())


def counts(record, result):
    truth, labeled = record['truth'], record['labeled']
    error = (result['points_b']-truth).norm(dim=-1)
    flow_error = (record['flow']-truth).norm(dim=-1)
    good, fg = labeled&(error<=3), labeled&(flow_error<=3)
    distance = (record['candidates'][:,None]-truth[:,:,None]).norm(dim=-1)
    positives = (distance<=3)&record['valid_b'][:,None]
    covered = labeled&positives.any(-1)
    lp = result['logp']
    best = lp.masked_fill(~positives,-torch.inf).amax(-1)
    rank = (lp>best[...,None]).sum(-1)+1
    out = dict(n=error.numel(), visible=int(labeled.sum()), error_sum=float(error[labeled].sum()),
        flow_error_sum=float(flow_error[labeled].sum()), flow_correct3=int(fg.sum()),
        damaged=int((fg&~good).sum()), rescued=int((~fg&good).sum()), covered3=int(covered.sum()))
    out.update({f'correct{k}':int((labeled&(error<=k)).sum()) for k in (1,3,5,10)})
    out.update({f'rank{k}':int(((rank<=k)&covered).sum()) for k in (1,5,10)})
    return out


def aggregate(rows):
    total={k:sum(row[k] for row in rows) for k in rows[0] if k!='pair_id'}
    n, v, covered = max(1,total['n']), max(1,total['visible']), max(1,total['covered3'])
    return dict(**total, p3=total['correct3']/n, epe=total['error_sum']/v,
                flow_p3=total['flow_correct3']/n, flow_epe=total['flow_error_sum']/v,
                positive_rank1=total['rank1']/covered, positive_rank5=total['rank5']/covered)


@torch.no_grad()
def evaluate(adapter, files, temperature, device):
    adapter.eval(); rows=[]
    for file in files:
        record=load_record(file,device)
        result=match_windows(adapter,record,temperature)
        rows.append(dict(pair_id=record['pair_id'],**counts(record,result)))
    return dict(summary=aggregate(rows), pairs=rows)


def train(args):
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    started = time.perf_counter()
    data=json.loads((args.output/'data.json').read_text())
    temperature=data['temperature']
    files={k:sorted((args.output/'cache'/k).glob('*.pt')) for k in data['splits']}
    for name in files:
        if len(files[name])!=len(data['splits'][name]['pairs']):
            raise RuntimeError(f'Incomplete cache: {name}')
    out=args.output/(args.run_name or args.mode)
    out.mkdir(parents=True,exist_ok=False)
    adapter=LocalPairModulation(args.mode).to(args.device)
    training_cache={f:load_record(f,args.device) for f in files['train']} if args.preload_train else None
    config=dict(mode=args.mode,seed=args.seed,epochs=args.epochs,patience=args.patience,
        learning_rate=1e-3,min_learning_rate=1e-5,weight_decay=1e-4,gradient_clip=1.,
        windows_per_update=64,updates_per_epoch=min(args.samples_per_epoch or len(files['train']),len(files['train'])),
        training_pairs=len(files['train']),preload_train=args.preload_train,
        loss='exact-H four-neighbour soft-label dual classification only',
        min_delta=.001,selection='RGB validation all-query P@3; initial weights eligible',
        frozen='DINO/MVT/Flow/VGG/DeDoDe/temperature; QRRU unused',
        parameters=sum(p.numel() for p in adapter.parameters()),data=str(args.output/'data.json'))
    write_json(out/'config.json',config)
    baseline=evaluate(adapter,files['val'],temperature,args.device)
    write_json(out/'initial_val.json',baseline)
    best=baseline['summary']['p3'];best_epoch=0;stale=0
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=config['min_learning_rate'])
    def save(name,epoch):
        torch.save(dict(adapter=adapter.state_dict(),epoch=epoch,config=config,temperature=temperature),out/name)
    save('best.pt',0); history=[]
    print(json.dumps(dict(mode=args.mode,epoch=0,**baseline['summary'])),flush=True)
    if args.mode!='none':
        for epoch in range(1,args.epochs+1):
            adapter.train();order=list(files['train']);random.Random(args.seed+epoch).shuffle(order)
            order=order[:config['updates_per_epoch']]
            generator=torch.Generator(device=args.device).manual_seed(args.seed*1000+epoch)
            losses=[];usable_queries=0;updates=0
            for file in order:
                record=training_cache[file] if training_cache is not None else load_record(file,args.device)
                n=len(record['a']);take=torch.randperm(n,device=args.device,generator=generator)[:64]
                record={k:v[take] if isinstance(v,torch.Tensor) else v for k,v in record.items()}
                result=match_windows(adapter,record,temperature);loss,usable=loss_fn(result,record)
                if not usable:continue
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();losses.append(float(loss.detach()));usable_queries+=usable;updates+=1
            val=evaluate(adapter,files['val'],temperature,args.device)
            row=dict(epoch=epoch,loss=sum(losses)/max(1,len(losses)),lr=optimizer.param_groups[0]['lr'],
                     usable_queries=usable_queries,updates=updates,**val['summary'])
            history.append(row);write_json(out/'history.json',history);write_json(out/f'val_{epoch:03d}.json',val)
            if row['p3']>best+config['min_delta']:
                best=row['p3'];best_epoch=epoch;stale=0;save('best.pt',epoch)
            else:stale+=1
            save('last.pt',epoch);scheduler.step();print(json.dumps(dict(mode=args.mode,**row)),flush=True)
            if stale>=args.patience:break
    adapter.load_state_dict(torch.load(out/'best.pt',map_location=args.device,weights_only=False)['adapter'])
    for split in (name for name in files if name != 'train'):
        write_json(out/f'best_{split}.json',evaluate(adapter,files[split],temperature,args.device))
    write_json(out/'done.json',dict(best_epoch=best_epoch,best_val_p3=best,epochs_run=len(history),updates_run=sum(row['updates'] for row in history),
        elapsed_seconds=time.perf_counter()-started,peak_cuda_mb=torch.cuda.max_memory_allocated(args.device)/2**20))
    print('TRAIN DONE '+args.mode,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train']);p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--splits',nargs='+')
    p.add_argument('--selection',type=Path);p.add_argument('--reuse-cache',type=Path)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--run-name');p.add_argument('--samples-per-epoch',type=int);p.add_argument('--preload-train',action='store_true')
    p.add_argument('--mode',choices=['none','self','pair'],default='pair');p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=5)
    args=p.parse_args();prepare(args) if args.action=='prepare' else train(args)

if __name__=='__main__':main()
