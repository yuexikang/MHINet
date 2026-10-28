"""A0–A4 controlled delta study. Only small residual MLPs train; Pair stays frozen."""
import argparse
import json
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.qrru_corners.experiment import load_models, singleton
from mhinet.v2.d2_modulation.model import LocalPairModulation, load_adapter, extract_windows, match_windows
from mhinet.v2.d2_modulation.experiment import loss_fn
from mhinet.v2.joint_single.loss import global_d8
from .model import GlobalPairModulation
from .residual import ResidualCGMDP, channel_parameters, parameter_stats
from .research_metrics import resample, labels, metrics, aggregate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def averages(rows):
    return {s: {k:sum(r[s][k] for r in rows)/len(rows) for k in rows[0][s]} for s in rows[0]}


def datasets(config, smoke):
    selection = json.loads(Path(config['selection']).read_text())
    result = {k: GeometryPairDataset(v['root'], v['rows']) for k,v in selection.items()
              if k in ('train','val','hangzhou_test')}
    # The 64-pair and 192-pair lists use different roots and relative path conventions.
    result['rgb_test'] = ConcatDataset([GeometryPairDataset(selection[k]['root'],selection[k]['rows'])
                                      for k in ('test','extra_test')])
    if smoke:result={k:Subset(v,range(min(len(v),4 if k=='train' else 2))) for k,v in result.items()}
    return result


@torch.no_grad()
def prepare(base, ds, config, root, device, smoke):
    if (root/'calibration.json').exists(): return json.loads((root/'calibration.json').read_text())
    model = ResidualCGMDP(base, [], {}, config).to(device)
    moments = {s:[] for s in ('8','4','2')}; norms = {s:[] for s in moments}
    legacy, old_stats = {}, {m:[] for m in ('cumulative','delta')}
    archive = dict(training_pairs=254, note='Historical structural references only; unequal budget to the new 1024-pair experiment.', groups={})
    for mode in ('none','cumulative','delta'):
        folder = Path('outputs/cgmdp_global_modulation')/mode
        archive['groups'][mode] = {'done':json.loads((folder/'done.json').read_text()),
                                   **{key:json.loads((folder/f'{key}.json').read_text())['summary'] for key in ('test','hangzhou_test')}}
        if mode != 'none':
            state = torch.load(folder/'best.pt', map_location=device, weights_only=False)
            modules = torch.nn.ModuleDict({s:GlobalPairModulation('pair') for s in moments}).to(device)
            modules.load_state_dict(state['modulators']); legacy[mode] = modules.eval()
    for i in range(min(len(ds['train']), config['calibration_pairs'])):
        shared, deltas = model.encode(ds['train'][i]['images'].to(device))
        for s in moments:
            moments[s].append(deltas[s].float().square().mean((0,2,3)))
            norms[s].append(float(deltas[s].float().norm(dim=1).mean()))
        for mode, modules in legacy.items():
            description = None; stats = {}
            for s, delta in deltas.items():
                if description is None: description = delta; continue
                description = F.interpolate(description, size=delta.shape[-2:], mode='bilinear', align_corners=False)
                f = delta if mode=='delta' else description+delta
                gamma,beta = channel_parameters(modules[s],f)
                stats[s] = parameter_stats(f,gamma,beta)
                changed = modules[s](f)
                description = description+changed if mode=='delta' else changed
            old_stats[mode].append(stats)
    calibration = {s:dict(channel_rms=torch.stack(v).mean(0).sqrt().cpu().tolist(),
                          mean_feature_norm=sum(norms[s])/len(norms[s]), pairs=len(v)) for s,v in moments.items()}
    write(root/'calibration.json',calibration)
    archive['train32_modulation_stats'] = {m:averages(rows) for m,rows in old_stats.items()}
    write(root/'legacy_archive.json',archive)
    if not smoke: write(Path('reports/cgmdp_residual_research/legacy_archive.json'),archive)
    print(json.dumps(dict(event='calibration',mean_norm={s:c['mean_feature_norm'] for s,c in calibration.items()}, old_stats=archive['train32_modulation_stats'])),flush=True)
    return calibration


@torch.no_grad()
def evaluate(model, matcher, adapter, dataset, device):
    model.eval(); rows={k:[] for k in ('fixed','rematched')}; stats=[]; ids=[]; selections=[]
    for item in dataset:
        shared,deltas=model.encode(item['images'].to(device))
        changed,baseline,_,diagnostics=model.decode(shared,deltas);stats.append(diagnostics)
        gt=item['geometry'].to(device)
        original=labels(extract_windows(baseline,matcher),gt)
        original_result=match_windows(adapter,original,matcher.tau_f)
        fixed=labels(resample(changed,original),gt)
        rematched=labels(extract_windows(changed,matcher),gt)
        def parent_ids(record):
            xy=record['source'].mean(1)
            return ((xy[:,1]//8).long()*64+(xy[:,0]//8).long()).cpu().tolist()
        before,after=parent_ids(original),parent_ids(rematched)
        selections.append(dict(baseline=before,rematched=after,added=sorted(set(after)-set(before)),lost=sorted(set(before)-set(after))))
        for mode,record,geometry_shared in (('fixed',fixed,baseline),('rematched',rematched,changed)):
            result=match_windows(adapter,record,matcher.tau_f)
            rows[mode].append(metrics(geometry_shared,matcher,adapter,record,result,gt,original,original_result))
        ids.append(item['pair_id'][0])
    return dict(summary={k:aggregate(v) for k,v in rows.items()}, modulation=averages(stats), pair_ids=ids, parent_selection=selections, pairs=rows)


def training_step(model,matcher,adapter,item,config,device):
    shared,deltas=model.encode(item['images'].to(device))
    changed,baseline,drift,diagnostics=model.decode(shared,deltas)
    record=extract_windows(changed,matcher)
    take=torch.randperm(len(record['source']),device=device)[:config['train_windows']]
    record={k:v[take] for k,v in record.items()}
    gt=item['geometry'].to(device)
    record=labels(resample(changed,record),gt)
    result=match_windows(adapter,record,matcher.tau_f)
    l2,usable=loss_fn(result,record)
    # Global D8 supervision does not depend on the discrete/no_grad parent selector.
    l8=global_d8(changed,matcher,gt)
    total=l2+config['d8_weight']*l8+config['drift_weight']*drift
    return total,dict(d2=float(l2.detach()),d8=float(l8.detach()),drift=float(drift.detach()),labels=usable),diagnostics


def run(group,base,matcher,ds,calibration,config,root,device,smoke):
    out=root/group
    if (out/'done.json').exists():
        print(f'{group}: already complete',flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(config['seed']);random.seed(config['seed'])
    model=ResidualCGMDP(base,config['groups'][group],calibration,config).to(device)
    adapter=(load_adapter(config['pair_checkpoint'],device) if group in config['local_pair_groups']
             else LocalPairModulation('none').to(device).requires_grad_(False).eval())
    has_parameters=bool(model.modulators)
    optimizer=torch.optim.AdamW(model.modulators.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']) if has_parameters else None
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,config['epochs'],eta_min=config['min_lr']) if has_parameters else None
    latest=out/'latest.pt';start=0;best=-1
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=False)
        model.modulators.load_state_dict(state['modulators'])
        if has_parameters:
            optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start,best=state['epoch'],state['best']
    else:
        # Zero-init identity must also preserve accumulated descriptors at all output scales.
        with torch.no_grad():
            item=ds['val'][0];shared,deltas=model.encode(item['images'].to(device))
            changed,original,drift,_=model.decode(shared,deltas)
            for s in (8,4,2):torch.testing.assert_close(changed['pyramid'][s],original['pyramid'][s],atol=0,rtol=0)
            native=base(item['images'].to(device),pyramid_scales=(8,4,2))
            for s in (8,4,2):torch.testing.assert_close(original['pyramid'][s],native['pyramid'][s],atol=0,rtol=0)
        del shared,deltas,changed,original,native,drift
    def save(name,epoch):
        state=dict(group=group,epoch=epoch,best=best,config=config,modulators=model.modulators.state_dict(),
                   pair_checkpoint=config['pair_checkpoint'] if group in config['local_pair_groups'] else None,
                   optimizer=optimizer.state_dict() if optimizer else None,scheduler=scheduler.state_dict() if scheduler else None)
        torch.save(state,out/name)
    if start==0:
        initial=evaluate(model,matcher,adapter,ds['val'],device);write(out/'epoch_000.json',initial)
        best=initial['summary']['rematched']['d2']['p3'];save('best.pt',0)
        print(json.dumps(dict(event='initial',group=group,p3=best,parameters=sum(p.numel() for p in model.modulators.parameters()))),flush=True)
    epochs=1 if smoke else config['epochs']
    for epoch in range(start+1,(epochs if has_parameters else 0)+1):
        model.train();torch.manual_seed(config['seed']+epoch)
        loader=DataLoader(ds['train'],batch_size=1,shuffle=True,num_workers=0 if smoke else 2,collate_fn=singleton,
                          generator=torch.Generator().manual_seed(config['seed']+epoch))
        terms=[];diagnostics=[];losses=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
        for i,item in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss,record,stats=training_step(model,matcher,adapter,item,config,device)
            if not torch.isfinite(loss):raise FloatingPointError(record)
            loss.backward()
            if smoke:
                grads={s:float(m.condition[-1].weight.grad.abs().sum()) for s,m in model.modulators.items()}
                assert all(v>0 for v in grads.values()),grads
                assert all(p.grad is None for p in base.parameters())
                assert all(p.grad is None for p in adapter.parameters())
                print(json.dumps(dict(event='gradient',group=group,step=i,gradients=grads)),flush=True)
            torch.nn.utils.clip_grad_norm_(model.modulators.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();losses.append(float(loss.detach()));terms.append(record);diagnostics.append(stats)
            if i==0 or (i+1)%128==0:print(json.dumps(dict(event='train',group=group,epoch=epoch,pairs=i+1,loss=sum(losses)/len(losses),peak_gib=torch.cuda.max_memory_allocated(device)/2**30)),flush=True)
        scheduler.step()
        validation=evaluate(model,matcher,adapter,ds['val'],device);write(out/f'epoch_{epoch:03d}.json',validation)
        score=validation['summary']['rematched']['d2']['p3']
        if score>best:best=score;save('best.pt',epoch)
        save('latest.pt',epoch)
        record=dict(epoch=epoch,loss=sum(losses)/len(losses),terms={k:sum(t[k] for t in terms)/len(terms) for k in terms[0]},
                    modulation=averages(diagnostics),validation=validation['summary'],lr=scheduler.get_last_lr()[0],
                    seconds=time.perf_counter()-started,peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(event='epoch',group=group,epoch=epoch,p3=score,best=best,seconds=record['seconds'])),flush=True)
    # Validation only selects weights; held-out tests are read after all training updates.
    state=torch.load(out/'best.pt',map_location=device,weights_only=False);model.modulators.load_state_dict(state['modulators'])
    for split in ('rgb_test','hangzhou_test'):
        result=evaluate(model,matcher,adapter,ds[split],device);write(out/f'{split}.json',result)
        print(json.dumps(dict(event='test',group=group,split=split,summary=result['summary'])),flush=True)
    write(out/'done.json',dict(group=group,best_epoch=state['epoch'],epochs=epochs if has_parameters else 0,
                              frozen_reference=not has_parameters,early_stopping=False))
    del model,adapter,optimizer,scheduler;torch.cuda.empty_cache()


def publish(root):
    """Publish compact completed results; detailed per-pair records stay in outputs."""
    summary={}
    for group in ('A0','A1','A2','A3','A4'):
        folder=root/group
        if not (folder/'done.json').exists():continue
        summary[group]={'selection':json.loads((folder/'done.json').read_text())}
        for split in ('rgb_test','hangzhou_test'):
            data=json.loads((folder/f'{split}.json').read_text())
            summary[group][split]=dict(summary=data['summary'],modulation=data['modulation'])
    destination=Path('reports/cgmdp_residual_research')
    write(destination/'results.json',summary)
    lines=['# 受控Delta研究结果','', 'A0/A1为冻结参照；A2–A4采用相同10轮增量训练预算。杭州为近似H参考一致性；单seed结果不构成统计显著性证明。', '',
           '|组别|数据|窗口|D8覆盖|D8 P@3|D2覆盖|D2 Rank@1|P@1|P@3|EPE|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group, record in summary.items():
        for split in ('rgb_test','hangzhou_test'):
            for mode,s in record[split]['summary'].items():
                a,b=s['d8'],s['d2']
                lines.append(f"|{group}|{split}|{mode}|{100*a['coverage']:.2f}%|{100*a['p3']:.2f}%|{100*b['coverage']:.2f}%|{100*b['rank1_rate']:.2f}%|{100*b['p1']:.2f}%|{100*b['p3']:.2f}%|{b['epe']:.3f}|")
    (destination/'results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/mhi_v2/cgmdp_residual_research.json'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--group',choices=['all','A0','A1','A2','A3','A4'],default='all')
    p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text())
    torch.set_num_threads(2);torch.cuda.set_device(args.device)
    root=Path(config['output']+('_smoke' if args.smoke else ''));root.mkdir(parents=True,exist_ok=True)
    ds=datasets(config,args.smoke);write(root/'config.json',config);write(root/'data.json',{k:len(v) for k,v in ds.items()})
    base,matcher=load_models(args.device);matcher.requires_grad_(False).eval()
    calibration=prepare(base,ds,config,root,args.device,args.smoke)
    if args.prepare_only:return
    for group in (list(config['groups']) if args.group=='all' else [args.group]):
        run(group,base,matcher,ds,calibration,config,root,args.device,args.smoke)
        if not args.smoke:publish(root)
    print('RESEARCH RUN COMPLETE',flush=True)


if __name__=='__main__':main()
