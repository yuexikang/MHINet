"""Small paired ablation of full-map CGMDP modulation, independent of full-RGB training."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from mhinet.v2.qrru_corners.experiment import load_models, singleton
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.d2_modulation.model import LocalPairModulation, extract_windows, match_windows, center_indices
from mhinet.v2.d2_modulation.experiment import splits, loss_fn, counts, aggregate
from mhinet.v2.joint_single.path import sample
from .model import ModulatedCGMDP


def windows(shared, matcher, limit=None):
    # Choose windows with the same existing inference path, then resample LIVE D2 maps.
    record = extract_windows(shared, matcher)
    if limit is not None:
        take = torch.randperm(len(record['a']), device=record['a'].device)[:limit]
        record = {k: v[take] for k, v in record.items()}
    axis = (torch.arange(8, device=record['a'].device)-3.5)*2
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    delta = torch.stack((x, y), -1).reshape(1, 64, 2)
    ca = record['source'].mean(1, keepdim=True)+delta
    d2 = shared['pyramid'][2][0].float()
    record['a'], record['b'] = sample(d2[0], ca), sample(d2[1], record['candidates'])
    return record


def prediction(model, matcher, identity, item, device, limit=None):
    shared = model(item['images'].to(device))
    record = windows(shared, matcher, limit)
    result = match_windows(identity, record, matcher.tau_f)
    gt = item['geometry'].to(device)
    truth, known, visible = gt.query((record['source']+.5)/256-1)
    record.update(truth=(truth+1)*256-.5, labeled=known & visible)
    return result, record


@torch.no_grad()
def evaluate(model, matcher, identity, dataset, device):
    model.eval(); rows = []
    for item in dataset:
        result, record = prediction(model, matcher, identity, item, device)
        rows.append(dict(pair_id=item['pair_id'][0], **counts(record, result)))
    return dict(summary=aggregate(rows), pairs=rows)


def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def smoke(base, matcher, identity, dataset, device, out):
    torch.manual_seed(42)
    images = dataset[0]['images'].to(device)
    reports = []
    with torch.no_grad(): baseline = base(images)
    for mode in ('none', 'cumulative', 'delta'):
        model = ModulatedCGMDP(base, mode).to(device)
        with torch.no_grad():
            shared = model(images)
            for scale in (8, 4, 2): torch.testing.assert_close(shared['pyramid'][scale], baseline['pyramid'][scale], atol=0, rtol=0)
            torch.testing.assert_close(shared['flow16_norm'], baseline['flow16_norm'], atol=0, rtol=0)
        if mode == 'none': continue
        # Perturb only D8: D4/D2 must see the upsampled changed descriptor sum.
        with torch.no_grad():
            model.modulators['8'].condition[-1].bias[256:].fill_(.01)
            changed = model(images)
            propagation = {str(s): float((changed['pyramid'][s]-baseline['pyramid'][s]).abs().mean()) for s in (8, 4, 2)}
            assert all(v > 0 for v in propagation.values()), propagation
            model.modulators['8'].condition[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.modulators.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats(device)
        steps = []
        for step in range(2):
            model.train(); optimizer.zero_grad(set_to_none=True)
            result, record = prediction(model, matcher, identity, dataset[step], device, 64)
            loss, usable = loss_fn(result, record)
            loss.backward()
            gradients = {s: {str(i): float(layer.weight.grad.abs().sum()) for i, layer in enumerate(m.condition) if isinstance(layer,torch.nn.Linear)} for s,m in model.modulators.items()}
            assert all(v['2'] > 0 for v in gradients.values()), gradients
            if step == 1: assert all(v['0'] > 0 for v in gradients.values()), gradients
            assert all(p.grad is None for p in base.parameters())
            torch.nn.utils.clip_grad_norm_(model.modulators.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            steps.append(dict(loss=float(loss.detach()), labels=usable, gradients=gradients))
        reports.append(dict(mode=mode, identity='exact', propagation=propagation, steps=steps,
                            peak_gib=torch.cuda.max_memory_allocated(device)/2**30))
        del shared, changed, model, optimizer, result, record, loss
    write(out/'smoke.json', reports)
    print(json.dumps(reports), flush=True)


def run(mode, base, matcher, identity, ds, config, device, root):
    out = root/mode; out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config['seed'])
    model = ModulatedCGMDP(base, mode).to(device)
    initial = evaluate(model, matcher, identity, ds['val'], device)
    write(out/'epoch_000.json', initial)
    best = anchor = initial['summary']['p3']; bad = 0
    optimizer = torch.optim.AdamW(model.modulators.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['min_lr'])
    def save(name, epoch):
        torch.save(dict(modulators=model.modulators.state_dict(), mode=mode, epoch=epoch, best=best,
                        config=config, optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), out/name)
    save('best.pt', 0)
    epoch = 0
    for epoch in range(1, (0 if mode == 'none' else config['epochs'])+1):
        model.train(); total=0.; start=time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        data = DataLoader(ds['train'], batch_size=1, shuffle=True, num_workers=2, collate_fn=singleton,
                          generator=torch.Generator().manual_seed(config['seed']+epoch))
        for item in data:
            optimizer.zero_grad(set_to_none=True)
            result, record = prediction(model, matcher, identity, item, device, config['train_windows'])
            loss, _ = loss_fn(result, record); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.modulators.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach())
        scheduler.step()
        val = evaluate(model, matcher, identity, ds['val'], device)
        write(out/f'epoch_{epoch:03d}.json', val)
        score = val['summary']['p3']
        if score > best: best = score; save('best.pt', epoch)
        if score > anchor+config['early_stop_min_delta']: anchor, bad = score, 0
        else: bad += 1
        save('last.pt', epoch)
        row = dict(epoch=epoch, train_loss=total/len(ds['train']), val=val['summary'], best=best, bad_epochs=bad,
                   seconds=time.perf_counter()-start, peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(mode=mode, **row)), flush=True)
        if epoch >= config['early_stop_min_epochs'] and bad >= config['early_stop_patience']: break
    selected = torch.load(out/'best.pt', map_location=device, weights_only=False)
    model.modulators.load_state_dict(selected['modulators'])
    for split in ('test', 'hangzhou_test'): write(out/f'{split}.json', evaluate(model, matcher, identity, ds[split], device))
    write(out/'done.json', dict(best_epoch=selected['epoch'], last_epoch=epoch, best=best))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--mode',choices=['all','none','cumulative','delta'],default='all')
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[3]/'configs/mhi_v2/cgmdp_global_modulation.json')
    args=p.parse_args(); torch.set_num_threads(2)
    config=json.loads(args.config.read_text()); root=Path(config['output']); root.mkdir(parents=True,exist_ok=True)
    ds={k: GeometryPairDataset(path,rows) for k,(path,rows) in splits().items()}
    write(root/'config.json',config)
    base,matcher=load_models(args.device); matcher.requires_grad_(False).eval()
    identity=LocalPairModulation('none').to(args.device).requires_grad_(False).eval()
    if args.smoke: smoke(base,matcher,identity,ds['train'],args.device,root); return
    for mode in (config['modes'] if args.mode=='all' else [args.mode]):
        run(mode,base,matcher,identity,ds,config,args.device,root)
    print('COMPLETE',flush=True)


if __name__ == '__main__': main()
