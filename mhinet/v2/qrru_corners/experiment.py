"""Four QRRU-only runs; frozen Flow/D8/D2, two domains, two supervision choices."""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.qrru import QRRUV2
from mhinet.v2.flow_d8_d2.run import load
from mhinet.v2.flow_d8_d2.matching import infer
from mhinet.downstream.supervision import offsets, to_uv, to_norm

ROOT = Path('outputs/qrru_corners')
DOWNSTREAM = 'outputs/flow_d8_d2/training/best.pt'


def subset(rows, count, group):
    rng = random.Random(42)
    buckets = {}
    for row in rows:
        buckets.setdefault(row[group], []).append(row)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < count:
        for key in sorted(buckets):
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
    return selected


def selections():
    source = json.loads(Path('outputs/d16_final_round/selection.json').read_text())
    hz = json.loads(Path('outputs/hangzhou_cleaned_restart/selection.json').read_text())
    return dict(
        hangzhou=dict(root=source['roots']['hz'], train=hz['train'], val=hz['val'],
                      test=subset(source['hz_test'], 64, 'G')),
        synthetic=dict(root=source['roots']['rgb_train'], test_root=source['roots']['rgb_test'],
                       train=subset(source['rgb_train'], 254, 'source_dataset'),
                       val=subset(source['rgb_dev'], 64, 'source_dataset'),
                       test=subset(source['rgb_test'], 64, 'source_dataset')))


def singleton(items):
    return items[0]


def load_models(device):
    model, matcher = load(device)
    state = torch.load(DOWNSTREAM, map_location='cpu', weights_only=False, mmap=True)
    model.vgg.load_state_dict(state['vgg'])
    model.dedode.load_state_dict(state['dedode'])
    matcher.load_state_dict({**matcher.state_dict(), **state['temperatures']})
    # Replace the old QRRU (including its no-QRRU test hook) with identical fresh weights.
    torch.manual_seed(42)
    matcher.qrru = QRRUV2(channels=256, iterations=4, radius=1.5).to(device)
    model.requires_grad_(False).eval()
    matcher.requires_grad_(False).eval()
    matcher.qrru.requires_grad_(True)
    return model, matcher


@torch.no_grad()
def targets(a, b, gt):
    """Query all four H reference controls; weak H is allowed only in this experiment."""
    corner_a = a[:, None, None] + offsets(a.device, True)
    corner_b, known, visible = gt.query(to_norm(corner_a, (256, 256)))
    flow = to_uv(corner_b, (256, 256)) - (b[:, None, None] + offsets(a.device, True))
    center, ck, cv = gt.query(to_norm(a, (256, 256)))
    return flow.permute(0, 3, 1, 2), known & visible, to_uv(center, (256, 256)), ck & cv


@torch.no_grad()
def training_queries(prediction, gt, generator):
    a = (prediction['points_a'][:512] + .5) / 2 - .5
    b = (prediction['points_b'][:512] + .5) / 2 - .5
    controls, valid, truth, visible = targets(a, b, gt)
    eligible = visible & ((b - truth).norm(dim=-1) <= 6)
    eligible &= ((controls.abs() <= 6) | ~valid[:, None]).flatten(1).all(1)
    keep = eligible.nonzero().flatten()
    keep = keep[torch.randperm(len(keep), generator=generator, device=a.device)[:64]]
    a, b = a[keep], b[keep]
    source = gt.sources((256, 256), a.device)
    take = torch.randperm(len(source), generator=generator, device=a.device)[:128-len(a)]
    source = source[take]
    truth, _, _ = gt.query(source)
    noise = torch.randn(source.shape, generator=generator, device=a.device).mul(.8).clamp(-2.4, 2.4)
    return torch.cat((a, to_uv(source, (256, 256)))), torch.cat((b, to_uv(truth, (256, 256)) + noise)), len(a)


def loss_fn(qrru, d2, a, b, gt, corner_weight):
    target, valid, center, center_valid = targets(a, b, gt)
    result = qrru(d2[0], d2[1], a, b)
    tolerance = 0 if gt.exact else gt.tolerance_px / 2
    def penalty(error, mask):
        selected = error[mask]
        residual = (selected.abs() - tolerance).clamp_min(0)
        return (torch.sqrt(residual.square() + 1e-6) - .001).mean() if selected.numel() else selected.sum()
    center_loss = result['centers'].sum() * 0
    corner_loss = result['flows'].sum() * 0
    for step in range(4):
        weight = .8 ** (3-step)
        center_loss += weight * penalty(result['centers'][step] - center, center_valid)
        corner_loss += weight * penalty((result['flows'][step] - target).permute(0, 2, 3, 1), valid)
    return center_loss + corner_weight * corner_loss, dict(center=float(center_loss.detach()), corner=float(corner_loss.detach()))


@torch.no_grad()
def evaluate(model, matcher, dataset, device):
    matcher.qrru.eval()
    records = []
    for batch in dataset:
        gt = batch['geometry'].to(device)
        shared = model(batch['images'].to(device))
        prediction = infer(shared, matcher, 'local')
        a = (prediction['points_a'] + .5) / 2 - .5
        b = (prediction['points_b'] + .5) / 2 - .5
        d2 = shared['pyramid'][2][0].float()
        target, corner_valid, truth, visible = targets(a, b, gt)
        projected = matcher.qrru.project(d2)
        pieces = [matcher.qrru(projected[0], projected[1], a[start:start+512], b[start:start+512], projected=True)
                  for start in range(0, len(a), 512)]
        centers = torch.cat([x['centers'] for x in pieces], 1) if pieces else b.new_empty((4, 0, 2))
        controls = torch.cat([x['flows'] for x in pieces], 1) if pieces else b.new_empty((4, 0, 2, 2, 2))
        trajectory = torch.cat((b[None], centers))
        error = (trajectory - truth[None]).norm(dim=-1) * 2
        finite_inside = torch.isfinite(trajectory).all(-1) & (trajectory >= -.5).all(-1) & (trajectory <= 255.5).all(-1)
        good3 = visible[None] & (error <= 3) & finite_inside
        correct = {f'correct{t}': (visible[None] & (error <= t) & finite_inside).sum(-1).tolist() for t in (1, 3, 5)}
        corner_error = (controls - target[None]).norm(dim=2) * 2
        records.append(dict(pair_id=batch['pair_id'][0], group=batch['group'], n=len(a), visible=int(visible.sum()),
            error_sum=error[:, visible].sum(-1).tolist(), **correct,
            damaged3=(good3[0:1] & ~good3).sum(-1).tolist(),
            rescued3=(~good3[0:1] & good3).sum(-1).tolist(),
            improved=((error < error[0:1]) & visible[None]).sum(-1).tolist(),
            worsened=((error > error[0:1]) & visible[None]).sum(-1).tolist(),
            corner_error_sum=corner_error[:, corner_valid].sum(-1).tolist(), corner_n=int(corner_valid.sum())))
    total = lambda key: np.asarray([r[key] for r in records]).sum(0)
    n, visible = max(1, int(total('n'))), max(1, int(total('visible')))
    return dict(p1=(total('correct1')/n).tolist(), p3=(total('correct3')/n).tolist(),
        p5=(total('correct5')/n).tolist(), epe=(total('error_sum')/visible).tolist(),
        damaged3=(total('damaged3')/max(1, int(total('correct3')[0]))).tolist(),
        rescued3=(total('rescued3')/max(1, n-int(total('correct3')[0]))).tolist(),
        corner_epe=(total('corner_error_sum')/max(1, int(total('corner_n')))).tolist(), pairs=records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', choices=['hangzhou', 'synthetic'], required=True)
    p.add_argument('--supervision', choices=['center', 'corners'], required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    selected = selections()
    out = ROOT / (args.domain + '_' + args.supervision + ('_smoke' if args.smoke else ''))
    out.mkdir(parents=True, exist_ok=False)
    config = dict(domain=args.domain, supervision=args.supervision, corner_weight=.1 if args.supervision=='corners' else 0,
        seed=42, epochs=30, learning_rate=8e-5, min_lr=1e-6, accumulation=4,
        initialization='fresh QRRUV2 seed42, zero last layer', downstream=DOWNSTREAM,
        frozen=['DINO','MVT','Flow','Matchability','VGG','D8/D2','temperatures'],
        selection='outputs/d16_final_round/selection.json', budget='30 epochs for all four; select best own-domain validation P@3',
        train_ids=[r['pair_id'] for r in selected[args.domain]['train']],
        val_ids=[r['pair_id'] for r in selected[args.domain]['val']])
    (out/'config.json').write_text(json.dumps(config, indent=2))
    ds = selected[args.domain]
    train = GeometryPairDataset(ds['root'], ds['train'])
    val = GeometryPairDataset(ds['root'], ds['val'])
    model, matcher = load_models(args.device)
    params = list(matcher.qrru.parameters())
    optimizer = torch.optim.AdamW(params, lr=8e-5, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=1e-6)
    def save(name, epoch, score):
        torch.save(dict(qrru=matcher.qrru.state_dict(), epoch=epoch, score=score, config=config), out/name)
    initial = evaluate(model, matcher, val, args.device) if not args.smoke else dict(p3=[0]*5)
    (out/'epoch_000.json').write_text(json.dumps(initial))
    best = initial['p3'][-1]
    save('best.pt', 0, best)
    print(json.dumps(dict(event='initial',p3=initial['p3'])), flush=True)
    for epoch in range(1, 31):
        matcher.qrru.train()
        generator = torch.Generator(device=args.device).manual_seed(42000+epoch)
        loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=2, collate_fn=singleton,
                            generator=torch.Generator().manual_seed(42+epoch))
        optimizer.zero_grad(set_to_none=True)
        losses = []; predicted_count = 0; started = time.perf_counter()
        for index, batch in enumerate(loader):
            gt = batch['geometry'].to(args.device)
            with torch.no_grad():
                shared = model(batch['images'].to(args.device))
                prediction = infer(shared, matcher, 'local')
                a, b, predicted = training_queries(prediction, gt, generator)
            loss, terms = loss_fn(matcher.qrru, shared['pyramid'][2][0].float(), a, b, gt, config['corner_weight'])
            count = min(4, len(train)-(index//4)*4)
            (loss/count).backward()
            losses.append(float(loss.detach())); predicted_count += predicted
            if (index+1)%4==0 or index+1==len(train):
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if args.smoke and index==7:
                assert all(p.grad is None for p in model.parameters())
                print(json.dumps(dict(event='smoke_passed',losses=losses,**terms)),flush=True)
                return
        scheduler.step()
        result = evaluate(model, matcher, val, args.device)
        (out/f'epoch_{epoch:03d}.json').write_text(json.dumps(result))
        if result['p3'][-1] > best:
            best = result['p3'][-1]
            save('best.pt', epoch, best)
        save('last.pt', epoch, result['p3'][-1])
        record = dict(epoch=epoch,loss=float(np.mean(losses)),predicted_per_pair=predicted_count/len(train),
            seconds=time.perf_counter()-started,best_p3=best,**{k:v for k,v in result.items() if k!='pairs'})
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    checkpoint = torch.load(out/'best.pt',map_location=args.device,weights_only=False)
    matcher.qrru.load_state_dict(checkpoint['qrru'])
    for domain, spec in selected.items():
        test = GeometryPairDataset(spec.get('test_root',spec['root']),spec['test'])
        result = evaluate(model,matcher,test,args.device)
        (out/f'test_{domain}.json').write_text(json.dumps(result))
    (out/'status.json').write_text(json.dumps(dict(status='complete',best_epoch=checkpoint['epoch'],best_p3=best)))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
