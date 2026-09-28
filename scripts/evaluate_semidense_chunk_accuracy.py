"""Paired full-val comparison of inference chunks; never edits checkpoint/default config."""
import argparse,json,sys,time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.downstream.training import SemidenseSystem
from mhinet.downstream.semidense import SemidenseMatcher,SemidenseConfig
from mhinet.downstream.metrics import dense_metrics
from mhinet.dataio.data import load_rgb_bicubic
from scripts.semidense_timing import timed_inference

CHUNKS=(32,128,256,512,1024)


def compare_points(base_a,base_b,a,b):
    """Compare source sets separately from output order; align unique common sources."""
    base_a,base_b,a,b=[np.asarray(x,dtype=np.float32).reshape(-1,2) for x in (base_a,base_b,a,b)]
    dtype=np.dtype([('x',np.float32),('y',np.float32)])
    def keys(x):return np.ascontiguousarray(x).view(dtype).reshape(-1)
    ka,kb=keys(base_a),keys(a)
    common,ia,ib=np.intersect1d(ka,kb,return_indices=True)
    ua,ub=np.unique(ka),np.unique(kb)
    duplicates=(len(ua)!=len(ka) or len(ub)!=len(kb))
    distances=np.linalg.norm(base_b[ia]-b[ib],axis=1) if len(common) and not duplicates else np.array([])
    return dict(source_order_equal=np.array_equal(base_a,a),source_sets_equal=np.array_equal(ua,ub),
        targets_order_equal=np.array_equal(base_b,b),baseline_points=len(base_a),candidate_points=len(a),
        added_sources=len(ub)-len(common),dropped_sources=len(ua)-len(common),common_sources=len(common),
        duplicate_sources=duplicates,compared_targets=len(distances),
        common_target_mean_delta_px=float(distances.mean()) if len(distances) else None,
        common_target_max_delta_px=float(distances.max()) if len(distances) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=2)
    args=p.parse_args()
    if not 0<=args.shard<args.shards:raise ValueError('Invalid shard')
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.manual_seed(0)
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    rows=[r for r in rows if r['tier']==3];assert len(rows)==3848 and all(r['split']=='val' for r in rows)
    selected=list(enumerate(rows))[args.shard::args.shards]
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True);meta=state['metadata']
    assert state['progress']['optimizer_step']==meta['total_steps']
    runtime=replace(RuntimePaths.from_json(meta['config']['runtime']),device='cuda:0')
    assert sha256_file(runtime.dino_checkpoint)==meta['dino_sha256']
    shared,_=build_shared_network(runtime,lora=False)
    model=SemidenseSystem(shared,SemidenseMatcher(SemidenseConfig(**meta['matcher']))).cuda();model.load_state_dict(state['model'],strict=True);model.eval();del state
    args.output.mkdir(parents=True)
    registration=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256_file(args.checkpoint),manifest=str(args.manifest),manifest_sha256=sha256_file(args.manifest),tier=3,split='val',shard=args.shard,shards=args.shards,pairs=len(selected),chunks=CHUNKS,script_sha256=sha256_file(Path(__file__)),baseline_tag='baseline/pre-chunk-accuracy-20260928')
    (args.output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    root=args.manifest.parent
    with torch.no_grad(),(args.output/'pairs.jsonl').open('w') as stream:
        for position,(index,row) in enumerate(selected):
            images=torch.stack([load_rgb_bicubic(root/row['image_'+side]) for side in 'AB'])[None].cuda()
            sizes=[(tuple(row['size_A']),tuple(row['size_B']))]
            if position==0:
                for chunk in CHUNKS:
                    model.matcher.config=replace(model.matcher.config,window_chunk=chunk)
                    model.matcher.infer(model(images),sizes)
                torch.cuda.synchronize()
            # Baseline first; rotate the other chunk sizes to reduce fixed order bias.
            alternatives=list(CHUNKS[1:]);shift=index%len(alternatives);order=[32]+alternatives[shift:]+alternatives[:shift]
            variants={};base_a=base_b=None
            for chunk in order:
                model.matcher.config=replace(model.matcher.config,window_chunk=chunk)
                torch.cuda.reset_peak_memory_stats()
                results,timing=timed_inference(model,images,sizes);result=results[0]
                a=result['points_a'].cpu().numpy();b=result['points_b'].cpu().numpy()
                if chunk==32:base_a,base_b=a.copy(),b.copy()
                metrics={'final':dense_metrics(row,root,a,b)}
                for stage,ak,bk in [('coarse','coarse_a','coarse_b'),('fine','fine_all_a','fine_all_b')]:
                    if ak in result:metrics[stage]=dense_metrics(row,root,result[ak].cpu().numpy(),result[bk].cpu().numpy())
                variants[str(chunk)]=dict(timing=timing,metrics=metrics,comparison=compare_points(base_a,base_b,a,b),failure_reason=result['failure_reason'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
            record=dict(index=index,pair_id=row['pair_id'],variants=variants)
            stream.write(json.dumps(record)+'\n');stream.flush()
            if (position+1)%10==0:print(json.dumps(dict(shard=args.shard,completed=position+1,total=len(selected))),flush=True)
    (args.output/'completed.json').write_text(json.dumps(dict(pairs=len(selected),status='completed'))+'\n')


if __name__=='__main__':main()
