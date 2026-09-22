"""Cascaded evaluation; GT is used only after prediction for reporting."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image,ImageDraw
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.dataio.data import load_rgb_bicubic
from .training import SemidenseSystem
from .semidense import SemidenseMatcher,SemidenseConfig
from .metrics import dense_metrics


def coverage(points,size,bins=16):
    if not len(points):return 0.
    xy=np.floor(np.asarray(points)/np.asarray(size)*bins).astype(int)
    good=((xy>=0)&(xy<bins)).all(1)
    return len(np.unique(xy[good],axis=0))/(bins*bins)


def draw_matches(root,row,result,path):
    images=[Image.open(root/row['image_'+s]).convert('RGB') for s in ('A','B')]
    canvas=Image.new('RGB',(sum(i.width for i in images),max(i.height for i in images)))
    canvas.paste(images[0],(0,0));canvas.paste(images[1],(images[0].width,0))
    draw=ImageDraw.Draw(canvas)
    for a,b in zip(result['points_a'][:100],result['points_b'][:100]):
        x,y=map(float,a);u,v=map(float,b)
        draw.line((x,y,u+images[0].width,v),fill=(70,220,100),width=1)
    canvas.thumbnail((1600,900));canvas.save(path)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--tier',type=int,default=1,choices=(1,2,3))
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args(argv)
    if args.limit<0:raise ValueError('Negative limit')
    if args.output.exists():raise FileExistsError(args.output)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    meta=state['metadata']
    if meta.get('task')!='semidense_qrru_v1':raise ValueError('Not a second-stage checkpoint')
    runtime=replace(RuntimePaths.from_json(meta['config']['runtime']),device=args.device)
    if sha256_file(runtime.dino_checkpoint)!=meta['dino_sha256']:raise ValueError('DINO identity mismatch')
    shared,_=build_shared_network(runtime,lora=False)
    system=SemidenseSystem(shared,SemidenseMatcher(SemidenseConfig(**meta['matcher']))).to(args.device)
    system.load_state_dict(state['model'],strict=True);system.eval()
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    rows=[x for x in rows if x['tier']==args.tier]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('Empty evaluation')
    args.output.mkdir(parents=True)
    registration=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256_file(args.checkpoint),
        manifest=str(args.manifest.resolve()),manifest_sha256=sha256_file(args.manifest),tier=args.tier,
        scope='smoke' if args.limit else 'full_tier',protocol='semidense_qrru_v1')
    (args.output/'registration.json').write_text(json.dumps(registration,indent=2))
    root=args.manifest.parent;records=[]
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize()
    with torch.no_grad(),(args.output/'pairs.jsonl').open('w') as stream:
        if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        for index,row in enumerate(rows):
            images=torch.stack([load_rgb_bicubic(root/row['image_'+s]) for s in ('A','B')])[None].to(args.device)
            # First sample warm-up is excluded from measured latency.
            if index==0:system.matcher.infer(system(images),[(tuple(row['size_A']),tuple(row['size_B']))]);sync()
            started=time.perf_counter()
            result=system.matcher.infer(system(images),[(tuple(row['size_A']),tuple(row['size_B']))])[0]
            sync();elapsed=time.perf_counter()-started
            a=result['points_a'].cpu().numpy();b=result['points_b'].cpu().numpy()
            metrics=dense_metrics(row,root,a,b)
            record=dict(pair_id=row['pair_id'],milliseconds=elapsed*1000,failure_reason=result['failure_reason'],
                final=metrics,coverage_a=coverage(a,row['size_A']),coverage_b=coverage(b,row['size_B']),
                qrru_outside_samples=result.get('qrru_outside_samples',0),rejected_qrru=result.get('rejected_qrru',0))
            if 'fine_all_a' in result:
                record['fine']=dense_metrics(row,root,result['fine_all_a'].cpu().numpy(),result['fine_all_b'].cpu().numpy())
                record['coarse']=dense_metrics(row,root,result['coarse_a'].cpu().numpy(),result['coarse_b'].cpu().numpy())
            stream.write(json.dumps(record)+'\n');stream.flush();records.append(record)
            if index<3:draw_matches(root,row,result,args.output/f'pair_{index:04d}.png')
    report=dict(**registration,pairs=len(records),failure_rate=float(np.mean([r['failure_reason']!='none' for r in records])),
        mean_ms=float(np.mean([r['milliseconds'] for r in records])),
        precision={key:float(np.mean([r['final']['thresholds'][key]['precision'] for r in records])) for key in ('1.0','3.0','5.0','10.0')},
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
        note='Native target pixels; first-pair warmup excluded; no fitted-H accuracy claim.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    return 0


if __name__=='__main__':raise SystemExit(main())
