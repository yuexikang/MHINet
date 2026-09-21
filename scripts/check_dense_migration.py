"""Compare original LoMa orchestration to migrated downstream on identical features."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.dataio.data import load_rgb_bicubic
from mhinet.downstream.dense import HGuidedDenseDownstream
from mhinet.downstream.metrics import dense_metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    path=Path('outputs/shared_descriptor_frozen_tier3_seed0/latest.pt')
    state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    runtime=replace(RuntimePaths.from_json(state['metadata']['config']['runtime']),device='cuda:0')
    model,_=build_shared_network(runtime,lora=False);model.load_state_dict(state['model'],strict=True);model.eval()
    from experiments.stage1_dedode_pyramid_hroi_v1.model import Stage1DeDoDePyramidMatcher
    from experiments.stage1_dedode_pyramid_hroi_v1.configs import registered_config
    import importlib.util
    spec=importlib.util.spec_from_file_location('legacy_dense_metrics',runtime.loma_root/'scripts/evaluate_googleearth_scale_pairs.py')
    metrics_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics_module)
    root=Path('/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2/val')
    row=json.loads(next((root/'pairs.jsonl').open()))
    images=torch.stack([load_rgb_bicubic(root/row['image_'+s]) for s in ('A','B')])[None].cuda()
    results={}
    with torch.no_grad():
        shared=model(images)
        legacy=Stage1DeDoDePyramidMatcher.__new__(Stage1DeDoDePyramidMatcher)
        torch.nn.Module.__init__(legacy)
        # Reuse identical GHIM/CGMDP outputs: this isolates downstream migration.
        legacy.extract_pyramid=lambda *a,**kw:(shared['pair_descriptors'],shared['context'],
            SimpleNamespace(d8=shared['pyramid'][8][0],d2=shared['pyramid'][2][0]),shared['stage1'])
        for mode in ('C3-P3','C3-P6'):
            old=legacy._match(root/row['image_A'],root/row['image_B'],registered_config(mode),oracle_h=None,oracle_paths=None)
            new=HGuidedDenseDownstream(mode)(shared,(tuple(row['size_A']),tuple(row['size_B'])))
            old_pairs=np.c_[old.points_a,old.points_b]
            new_pairs=np.c_[new['points_a'].cpu().numpy(),new['points_b'].cpu().numpy()]
            old_order=np.lexsort(old_pairs.T[::-1]);new_order=np.lexsort(new_pairs.T[::-1])
            diffs={}
            for field in ('points_a','points_b','confidence'):
                x=getattr(old,field)[old_order];y=new[field].cpu().numpy()[new_order]
                np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)
                diffs[field]=float(np.max(np.abs(x-y))) if len(x) else 0.
            # Legacy grouping metadata does not enter any numeric accuracy formula.
            legacy_row={**row,'_H_A_to_B':np.array(row['H_A_to_B']),
                '_mask_A_overlap':str(root/row['mask_A_overlap']),
                '_mask_B_overlap':str(root/row['mask_B_overlap']),
                'input_pair_row_index':0,'geometric_augmentation':{'profile':'migration'},
                'size_ratio':1.25,'overlap_ratio':.7}
            old_metrics,_,_=metrics_module.pair_metrics(legacy_row,old.points_a,old.points_b,0.,(1.,3.,5.,10.),5.,20,10.)
            new_metrics=dense_metrics(row,root,old.points_a,old.points_b)
            for t in (1.,3.,5.,10.):
                m=new_metrics['thresholds'][str(t)]
                assert old_metrics[f'NCM@{int(t)}px']==m['NCM']
                assert old_metrics[f'Pre@{int(t)}px']==m['precision']
                assert old_metrics[f'OverlapPre@{int(t)}px']==m['overlap_precision']
            assert old_metrics['RMSE']==new_metrics['legacy_RMSE_correct5_or_failure10']
            assert old_metrics['SR']==new_metrics['success_20_correct_at_5px']
            results[mode]=dict(matches=len(old.points_a),comparison='canonical coordinate order; atol1e-7 rtol1e-6',max_abs_differences=diffs,legacy_metrics_exact=True)
    report=dict(status='passed',scope='original downstream orchestration vs migrated on identical shared tensors; not upstream legacy-weight parity',
        pair_id=row['pair_id'],checkpoint_sha256=sha256_file(path),call_counts=shared['call_counts'],results=results)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
