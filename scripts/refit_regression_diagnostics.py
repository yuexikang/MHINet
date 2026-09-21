"""Counterfactual-only GT filtering of saved coarse outputs; never used in inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from mhinet.pretraining.data import SharedPairDataset
from mhinet.pretraining.loss import grid,sample
from mhinet.ops.geometry import safe_project_points
from mhinet.models.feature_provider import SafeMatchabilityWeightedHomographyFitter,make_loma_importable
from mhinet.pretraining.overlap_metrics import overlap_projection_error


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=json.loads((args.input/'report.json').read_text())
    make_loma_importable('/home/disk1/LoMa','/home/disk1/LoMa/third_party/LoRetta')
    data=SharedPairDataset(Path('/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2/val/pairs.jsonl'),tier=1)
    index={r.pair_id:i for i,r in enumerate(data.index)};fitter=SafeMatchabilityWeightedHomographyFitter()
    with torch.no_grad():
        for r in report['pairs']:
            b=data[index[r['pair_id']]];gt=b['H_gt_norm'][None];ma=b['mask_A_overlap'][None];mb=b['mask_B_overlap'][None]
            saved=np.load(args.input/f"{r['rank']:02d}"/'coarse_arrays.npz');arrays={k:torch.from_numpy(saved[k]) for k in saved.files}
            h,w=arrays['after_coarse_warp'].shape[-2:];xy=grid(h,w,'cpu');truth,valid,_=safe_project_points(gt,xy[None]);truth=truth[0]
            support=valid[0]&(truth.abs()<=1-1/784).all(1)&(sample(ma[0],xy)[:,0]>.999)&(sample(mb[0],truth)[:,0]>.999)
            training_target=valid[0]&(truth.abs()<=1).all(1)&(torch.nn.functional.interpolate(ma,(h,w),mode='nearest').flatten()>.5)
            for name in ('before','after'):
                weight=arrays[name+'_coarse_matchability'].flatten();keep=weight>.3
                r['states'][name]['outside_gt_weight_fraction']=float(weight[keep&~support].sum()/weight[keep].sum())
                r['states'][name]['selected_training_target_negative']=int((keep&~training_target).sum())
            after=arrays['after_coarse_matchability'];before=arrays['before_coarse_matchability'];warp=arrays['after_coarse_warp']
            error=((warp[0].permute(1,2,0).reshape(-1,2)-truth)*392).norm(dim=1)
            variants={'after_weights_intersect_old_gate':after*(before>.3),
                'oracle_gt_visible_only':after*support.reshape_as(after),
                'oracle_gt_visible_and_error5_only':after*(support&(error<=5)).reshape_as(after)}
            for name,weight in variants.items():
                H,ok,count=fitter(warp,weight)
                r['crossed_refits'][name]=dict(error_px=overlap_projection_error(H,gt,ma,mb,fit_valid=bool(ok[0]))['mean_px'],selected=int(count[0]))
            print(r['rank'],json.dumps(r['crossed_refits']),flush=True)
    report['diagnostic_limitations']='GT filtering is an oracle-only cause probe, not a deployable fix; top5 selected on val, not representative evaluation.'
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
