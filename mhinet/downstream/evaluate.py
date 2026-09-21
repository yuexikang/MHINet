"""Optional LoMa downstream evaluation; zero-shot P3/P6, never uses GT for matching."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm
from mhinet.config import RuntimePaths,sha256_file,PROJECT_ROOT
from mhinet.pretraining.model import build_shared_network
from mhinet.dataio.data import load_rgb_bicubic
from .dense import HGuidedDenseDownstream
from .metrics import dense_metrics


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tier',type=int,choices=(1,2,3),required=True)
    p.add_argument('--result-id',choices=('C3-P3','C3-P6'),default='C3-P3')
    p.add_argument('--limit',type=int,default=0,help='Explicit smoke subset; 0 means all selected-tier pairs')
    args=p.parse_args(argv)
    if args.limit<0:raise ValueError('limit must be nonnegative')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    runtime=replace(RuntimePaths.from_json(state['metadata']['config']['runtime']),device='cuda:0')
    registration=dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=sha256_file(args.checkpoint),
        manifest=str(args.manifest.resolve()),manifest_sha256=sha256_file(args.manifest),
        tier=args.tier,split=args.manifest.parent.name,result_id=args.result_id,limit=args.limit,scope='smoke' if args.limit else 'full_tier',
        source_registry_sha256=sha256_file(PROJECT_ROOT/'artifacts/loma_downstream_source.json'))
    (args.output/'registration.json').write_text(json.dumps(registration,indent=2))
    model,_=build_shared_network(runtime,lora=state['metadata']['config']['lora'])
    model.load_state_dict(state['model'],strict=True);model.eval();matcher=HGuidedDenseDownstream(args.result_id)
    rows=[json.loads(l) for l in args.manifest.open() if l.strip()];rows=[r for r in rows if r['tier']==args.tier]
    if args.limit:rows=rows[:args.limit]
    if not rows:raise ValueError('Empty evaluation selection')
    stats=[];torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),(args.output/'pairs.jsonl').open('w') as f:
        for r in tqdm(rows,desc='LoMa '+args.result_id):
            root=args.manifest.parent
            images=torch.stack([load_rgb_bicubic(root/r['image_'+s]) for s in ('A','B')])[None].to(runtime.device)
            torch.cuda.synchronize();start=time.perf_counter()
            shared=model(images);result=matcher(shared,(tuple(r['size_A']),tuple(r['size_B'])))
            torch.cuda.synchronize();milliseconds=(time.perf_counter()-start)*1000
            record=dict(pair_id=r['pair_id'],runtime_ms=milliseconds,call_counts=shared['call_counts'],
                failure_reason=result['failure_reason'],**dense_metrics(r,root,result['points_a'].cpu().numpy(),result['points_b'].cpu().numpy()))
            f.write(json.dumps(record)+'\n');f.flush();stats.append(record)
    report=dict(**registration,pairs=len(stats),failure_rate=sum(r['failure_reason']!='none' for r in stats)/len(stats),
        mean_matches=float(np.mean([r['matches'] for r in stats])),
        mean_precision={t:float(np.mean([r['thresholds'][t]['precision'] for r in stats])) for t in stats[0]['thresholds']},
        success_rate_20_correct_at_5px=float(np.mean([r['success_20_correct_at_5px'] for r in stats])),
        mean_runtime_ms=float(np.mean([r['runtime_ms'] for r in stats])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        note='Native target pixels. No fitted-H metric yet; RMSE in pair rows is legacy correct-match-conditioned, not all-match accuracy. Warmup included.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return 0


if __name__=='__main__':raise SystemExit(main())
