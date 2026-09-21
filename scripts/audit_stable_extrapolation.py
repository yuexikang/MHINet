"""Full bidirectional 784-pixel GT extrapolation audit, not model evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.analyze_homography_extrapolation import resized_H, sensitivity


def measure(row):
    forward=resized_H(row)
    corners=np.array([[0.,0,1],[783,0,1],[783,783,1],[0,783,1]])
    records=[]
    for matrix in (forward,np.linalg.inv(forward)):
        q=corners@matrix.T;z=q[:,2]
        if not np.isfinite(q).all() or not (np.all(z>0) or np.all(z<0)):
            raise ValueError('Invalid projection before division: '+row['pair_id'])
        xy=q[:,:2]/z[:,None]
        normal=np.linalg.norm(matrix[2,:2])
        distance=float(np.abs(z).min()/normal) if normal>0 else None
        jac=(matrix[:2,:2][None]-xy[:,:,None]*matrix[2,:2][None,None,:])/z[:,None,None]
        magnification=float(np.linalg.svd(jac,compute_uv=False).max())
        records.append(dict(max_abs_corner_px=float(np.abs(xy).max()),horizon_min_px=distance,
                            max_corner_magnification=magnification))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report={'root':str(args.root),'coordinate_protocol':'784 input pixels; half-pixel resize; both directions',
        'scope':'GT geometry, not model predictions. Noise experiment uses GT geometric overlap only, not occlusion masks.', 'splits':{}}
    for split in ('train','val'):
        path=args.root/split/'pairs.jsonl';groups=defaultdict(list);worst=[]
        for line in path.open():
            row=json.loads(line); records=measure(row)
            extent=max(r['max_abs_corner_px'] for r in records)
            distances=[r['horizon_min_px'] for r in records if r['horizon_min_px'] is not None]
            distance=min(distances) if distances else float('inf')
            mag=max(r['max_corner_magnification'] for r in records)
            groups[row['tier']].append([extent,distance,mag])
            worst.append((extent,row));worst.sort(key=lambda x:x[0],reverse=True);worst=worst[:3]
        summary={}
        for tier,values in groups.items():
            a=np.asarray(values)
            summary[tier]=dict(pairs=len(a),corner_abs_gt_10000=int((a[:,0]>10000).sum()),
                corner_abs_gt_2000=int((a[:,0]>2000).sum()),horizon_lt_10px=int((a[:,1]<10).sum()),
                max_abs_corner_px=float(a[:,0].max()),min_horizon_distance_px=float(a[:,1].min()),
                max_corner_magnification=float(a[:,2].max()),corner_abs_p50_p90_p99=np.quantile(a[:,0],[.5,.9,.99]).tolist())
        experiments=[]
        for extent,row in worst:
            reverse={**row,'size_A':row['size_B'],'size_B':row['size_A'],'H_A_to_B':row['H_B_to_A']}
            experiments.append(dict(max_abs_corner_px=extent,forward=sensitivity(row),backward=sensitivity(reverse)))
        report['splits'][split]=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),tiers=summary,
            sensitivity_worst_three=experiments)
        print(split,json.dumps(summary),flush=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':main()
