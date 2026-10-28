"""No-training release regression: reference tensors and full RGB/Hangzhou tests."""
import argparse
import json
from pathlib import Path
import time

import torch
from ..pipeline import load_mhi_v2, DEFAULT_CONFIG
from ..data import GeometryPairDataset
from .metrics import measure, totals


def dataset(root, manifest):
    return GeometryPairDataset(root, [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--assets-root', type=Path, required=True)
    p.add_argument('--rgb-root', type=Path, required=True)
    p.add_argument('--hangzhou-root', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--reference', type=Path)
    p.add_argument('--deterministic-reference', type=Path)
    p.add_argument('--historical-results', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', choices=['best','latest'],default='best')
    p.add_argument('--full', action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    model=load_mhi_v2(a.config,a.device,runtime=json.loads(a.runtime.read_text()),assets_root=a.assets_root,variant=a.variant)
    ds=dict(test=dataset(a.rgb_root/'test',a.rgb_root/'test/pairs.jsonl'),
            hangzhou_test_real=dataset(a.hangzhou_root,a.hangzhou_root/'manifests/test_real.jsonl'))
    report=dict(release=model.release,device=a.device,torch=torch.__version__,gpu=torch.cuda.get_device_name(a.device),
                tf32_matmul=False,tf32_cudnn=True,reference=[],tests={})
    for reference in [r for r in (a.reference, a.deterministic_reference) if r]:
        index=json.loads((reference/'index.json').read_text())
        deterministic=index.get('deterministic',False)
        torch.use_deterministic_algorithms(deterministic)
        for record in index['rows']:
            item=ds[record['split']][record['index']]
            if item['pair_id'][0]!=record['pair_id']:raise ValueError('Reference dataset ordering changed')
            actual=model.match(item['images'],diagnostics=True);trace=actual['diagnostics'];w=trace['windows']
            fields={k:actual[k] for k in ('points_a','points_b','confidence')}
            fields.update({k:trace[k] for k in ('flow16_norm','flow16_target_norm','parents','parent_ids','centers','candidate_logp','candidate_valid')})
            fields.update({f'window_{k}':w[k] for k in ('a','b','valid_a','valid_b','source','candidates','score8')})
            expected=torch.load(reference/record['file'],map_location='cpu',weights_only=True)
            errors={}
            order_equal=torch.equal(fields['parent_ids'].cpu(),expected['parent_ids'])
            if not order_equal:
                if deterministic: raise AssertionError('Deterministic parent ordering changed')
                ids=fields['parent_ids'].cpu();other_ids=expected['parent_ids']
                assert torch.equal(ids.sort().values,other_ids.sort().values), 'Parent set changed'
                # Align entire parents for near-tied sparse CUDA reduction ordering.
                for tensors in (fields,expected):
                    perm=tensors['parent_ids'].argsort()
                    for key,value in list(tensors.items()):
                        if key in ('points_a','points_b','confidence'):
                            tensors[key]=value.reshape(len(perm),16,*value.shape[1:])[perm].reshape(value.shape)
                        elif not key.startswith('flow16_'):
                            tensors[key]=value[perm]
            for name,value in fields.items():
                value=value.cpu();other=expected[name]
                torch.testing.assert_close(value,other,rtol=0 if deterministic else 1e-5,atol=0 if deterministic else 1e-5)
                errors[name]=float((value.float()-other.float()).abs().max()) if value.numel() else 0.
            counts,_,_=measure(actual['points_a'],actual['points_b'],item['geometry'].to(a.device))
            for k in ('n','visible','correct1','correct3','correct5','correct10'):
                if counts[k]!=record['counts'][k]:raise AssertionError((record['pair_id'],k,counts[k],record['counts'][k]))
            assert trace['call_counts']['homography_fitter']==0
            assert len(actual['confidence'])<=2048
            assert actual['valid'].shape==actual['confidence'].shape==actual['parent'].shape
            report['reference'].append(dict(pair_id=record['pair_id'],deterministic=deterministic,parent_order_equal=order_equal,max_absolute_errors=errors,counts=counts))
        print(f'REFERENCE EQUIVALENCE PASSED: {len(index["rows"])} pairs; deterministic={deterministic}',flush=True)
    torch.use_deterministic_algorithms(False)
    (a.output/'regression.json').write_text(json.dumps(report,indent=2))
    if a.full:
        historical=json.loads(a.historical_results.read_text()) if a.historical_results else None
        for split,data in ds.items():
            rows=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats(a.device)
            for i in range(len(data)):
                item=data[i];result=model.match(item['images'])
                counts,_,_=measure(result['points_a'],result['points_b'],item['geometry'].to(a.device))
                rows.append(dict(pair_id=item['pair_id'][0],**counts))
                if (i+1)%128==0:print(json.dumps(dict(split=split,pairs=i+1,total=len(data),seconds=time.perf_counter()-started)),flush=True)
            summary=totals(rows);elapsed=time.perf_counter()-started
            comparison={}
            if historical:
                previous=historical['tests'][f'd2_{a.variant}_test_{split}']
                # Archived report stores test variants under summary.
                previous=previous.get('summary',previous)['local_flow']
                comparison={k:summary[k]-previous[k] for k in summary if k in previous}
            report['tests'][split]=dict(pairs=len(rows),seconds=elapsed,peak_gib=torch.cuda.max_memory_allocated(a.device)/2**30,summary=summary,difference_from_original=comparison)
            (a.output/f'{split}_pairs.json').write_text(json.dumps(rows,indent=2))
            (a.output/'regression.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report['tests'][split]),flush=True)
    print('RELEASE REGRESSION COMPLETE',flush=True)


if __name__=='__main__':main()
