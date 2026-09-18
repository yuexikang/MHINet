"""Read-only label/metric audit; does not filter, rewrite or regenerate data."""
import argparse
import json
from pathlib import Path
import numpy as np
from mhinet.config import sha256_file


def resized_H(row):
    transforms=[]
    for name in ('size_A','size_B'):
        w,h=row[name];sx,sy=784/w,784/h
        transforms.append(np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]]))
    return transforms[1]@np.asarray(row['H_A_to_B'])@np.linalg.inv(transforms[0])


def geometry(row):
    H=resized_H(row)
    points=np.array([[0,0,1],[783,0,1],[783,783,1],[0,783,1.]])
    q=points@H.T;z=q[:,2];xy=q[:,:2]/z[:,None]
    normal=np.linalg.norm(H[2,:2])
    distances=np.abs(z)/normal if normal>1e-15 else np.full(4,np.inf)
    magnification=[]
    for p,denom in zip(xy,z):
        J=(H[:2,:2]-p[:,None]*H[2,:2])/denom
        magnification.append(np.linalg.svd(J,compute_uv=False)[0])
    composed=np.asarray(row['T_source_to_B'])@np.linalg.inv(np.asarray(row['T_source_to_A']))
    composed/=composed[2,2]
    label=np.asarray(row['H_A_to_B']);label/=label[2,2]
    return dict(pair_id=row['pair_id'],tier=row['tier'],
        gt_corner_max_abs_px=float(np.abs(xy).max()),gt_corners_xy=xy.tolist(),
        horizon_distance_min_px=float(distances.min()) if np.isfinite(distances.min()) else None,
        max_corner_local_magnification=float(max(magnification)),
        denominator_sign_change=bool(z.min()*z.max()<=0),
        composition_max_abs_error=float(np.abs(composed-label).max()),
        visible_overlap_fractions=row['visible_overlap_fractions'])


def sensitivity(row):
    """Controlled correspondence-noise experiment, not a network prediction."""
    import cv2
    H=resized_H(row)
    def project(matrix,p):
        q=np.c_[p,np.ones(len(p))]@matrix.T
        return q[:,:2]/q[:,2:]
    y,x=np.mgrid[0:784:8,0:784:8]
    src=np.c_[x.ravel(),y.ravel()].astype(float)
    dst=project(H,src)
    good=((dst>=0)&(dst<=783)).all(1)
    src,dst=src[good],dst[good]
    rng=np.random.default_rng(0)
    selected=rng.choice(len(src),128,replace=False)
    corners=np.array([[0.,0],[783,0],[783,783],[0,783]])
    values=[]
    for _ in range(100):
        fit,_=cv2.findHomography(src[selected],dst[selected]+rng.normal(0,.1,(128,2)),method=0)
        if fit is None:raise ValueError('Controlled fit unexpectedly failed')
        values.append([np.linalg.norm(project(fit,src)-dst,axis=1).mean(),
                       np.linalg.norm(project(fit,corners)-project(H,corners),axis=1).mean()])
    values=np.asarray(values)
    return dict(pair_id=row['pair_id'],seed=0,trials=100,correspondences=128,
        gaussian_noise_sigma_target_px=.1,cv2_version=cv2.__version__,
        overlap_error_median_px=float(np.median(values[:,0])),
        corner_error_median_px=float(np.median(values[:,1])),
        corner_error_p90_px=float(np.quantile(values[:,1],.9)),
        note='Controlled GT correspondence noise plus unrobust least-squares H fitting; not MHINet output, and not evidence that all network errors are harmless.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_v1/val/pairs.jsonl')
    p.add_argument('--evaluated-pairs',default='outputs/shared_descriptor_frozen_tier2_seed0/validation/step_008663/pairs.jsonl')
    p.add_argument('--output',required=True)
    p.add_argument('--sensitivity',action='store_true')
    args=p.parse_args()
    measured={r['pair_id']:r for r in map(json.loads,Path(args.evaluated_pairs).open())}
    rows=[geometry(r) for r in map(json.loads,Path(args.manifest).open())]
    for r in rows:
        m=measured.get(r['pair_id'])
        if m:
            r['H0_corner_mace_px']=m['H0_mace_px']
            r['H0_overlap_mean_px']=m['H0_overlap']['mean_px']
    summaries={}
    for tier in (1,2,3):
        group=[r for r in rows if r['tier']==tier]
        summaries[str(tier)]=dict(pairs=len(group),
            gt_extent_gt_10000=sum(r['gt_corner_max_abs_px']>10000 for r in group),
            horizon_distance_lt_10=sum(r['horizon_distance_min_px'] is not None and r['horizon_distance_min_px']<10 for r in group),
            max_composition_error=max(r['composition_max_abs_error'] for r in group),
            denominator_sign_changes=sum(r['denominator_sign_change'] for r in group))
    scored=[r for r in rows if 'H0_corner_mace_px' in r]
    bins=[]
    for lo,hi in ((0,2000),(2000,10000),(10000,float('inf'))):
        selected=[r for r in scored if lo<=r['gt_corner_max_abs_px']<hi]
        bins.append(dict(gt_extent_range=[lo,hi if np.isfinite(hi) else None],pairs=len(selected),
            corner_mace_mean=float(np.mean([r['H0_corner_mace_px'] for r in selected])) if selected else None,
            overlap_mean=float(np.mean([r['H0_overlap_mean_px'] for r in selected])) if selected else None))
    report=dict(manifest_sha256=sha256_file(args.manifest),
        evaluated_pairs_sha256=sha256_file(args.evaluated_pairs),tiers=summaries,
        tier2_extent_bins=bins,worst_tier2=sorted(scored,key=lambda r:r['H0_corner_mace_px'],reverse=True)[:12])
    if args.sensitivity:
        selected=report['worst_tier2'][0]['pair_id']
        row=next(r for r in map(json.loads,Path(args.manifest).open()) if r['pair_id']==selected)
        report['controlled_sensitivity']=sensitivity(row)
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
