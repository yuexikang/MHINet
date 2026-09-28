"""Bounded inference-only chunk audit; does not modify weights or registered test outputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import replace
import json,time,torch,argparse,numpy as np
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.downstream.training import SemidenseSystem
from mhinet.downstream.semidense import SemidenseMatcher,SemidenseConfig
from mhinet.downstream.metrics import dense_metrics
from mhinet.dataio.data import load_rgb_bicubic
from scripts.semidense_timing import timed_inference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-ordering',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    source=root/'outputs/semidense_stable_v2_tier3_lr_c_seed0/latest.pt'
    manifest=Path('/home/disk1/Data/datasets/GoogleEarth_test_single_tier3_v1/test/pairs.jsonl')
    target=root/'artifacts/semidense_chunk_benchmark_20260928.json'
    if args.verify_ordering:target=target.with_name('semidense_chunk_ordering_20260928.json')
    if target.exists():raise FileExistsError(target)
    torch.set_num_threads(2);torch.manual_seed(0)
    state=torch.load(source,map_location='cpu',weights_only=True,mmap=True);meta=state['metadata']
    runtime=replace(RuntimePaths.from_json(meta['config']['runtime']),device='cuda:0')
    shared,_=build_shared_network(runtime,lora=False)
    model=SemidenseSystem(shared,SemidenseMatcher(SemidenseConfig(**meta['matcher']))).to(runtime.device)
    model.load_state_dict(state['model'],strict=True);model.eval();del state
    rows=[json.loads(x) for x in manifest.read_text().splitlines()];records=[]
    report=dict(checkpoint_sha256=sha256_file(source),manifest_sha256=sha256_file(manifest),gpu=torch.cuda.get_device_name(),pair_indices=[0,666,1333],repeats=1 if args.verify_ordering else 2,records=records,scope='Three-pair chunk-only performance diagnostic; not full validation or a changed production configuration')
    with torch.no_grad():
        for idx in report['pair_indices']:
            row=rows[idx];images=torch.stack([load_rgb_bicubic(manifest.parent/row['image_'+s]) for s in 'AB'])[None].cuda();sizes=[(tuple(row['size_A']),tuple(row['size_B']))]
            baseline=None
            for chunk in [32,128,256,512,1024]:
                model.matcher.config=replace(model.matcher.config,inference_window_chunk=chunk)
                model.matcher.infer(model(images),sizes);torch.cuda.synchronize()
                for repeat in range(report['repeats']):
                    torch.cuda.reset_peak_memory_stats()
                    result,timing=timed_inference(model,images,sizes);r=result[0]
                    a,b=r['points_a'].cpu(),r['points_b'].cpu();score=r['confidence'].cpu()
                    if baseline is None:baseline=(a.clone(),b.clone(),score.clone())
                    same_a=torch.equal(a,baseline[0]);same_shape=b.shape==baseline[1].shape
                    order=np.lexsort((a.numpy()[:,1],a.numpy()[:,0]));base_order=np.lexsort((baseline[0].numpy()[:,1],baseline[0].numpy()[:,0]))
                    set_equal=torch.equal(a[order],baseline[0][base_order]);unique=len(torch.unique(a,dim=0))==len(a)
                    metrics=dense_metrics(row,manifest.parent,a.numpy(),b.numpy())
                    record=dict(source_sets_equal=set_equal,source_points_unique=unique,max_target_delta_sorted_px=float((b[order]-baseline[1][base_order]).abs().max()) if set_equal and unique and len(a) else None,pair_id=row['pair_id'],chunk=chunk,repeat=repeat,wall_ms=timing['wall_total_ms'],module_ms=timing['module_ms'],peak_allocated_bytes=torch.cuda.max_memory_allocated(),matches=len(a),coarse_matches=len(r['coarse_a']),source_coordinates_bitwise_equal=same_a,target_coordinates_bitwise_equal=torch.equal(b,baseline[1]),max_target_delta_px=float((b-baseline[1]).abs().max()) if same_a and same_shape and b.numel() else None,confidence_bitwise_equal=torch.equal(score,baseline[2]),metrics=metrics)
                    records.append(record);target.write_text(json.dumps(report,indent=2)+'\n')
                    print(json.dumps({k:record[k] for k in ['pair_id','chunk','repeat','wall_ms','max_target_delta_px']}),flush=True)
    report['completed']=True;target.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
