"""LoMa NCM/precision protocol adapted to tier manifests, native target pixels.

Original: LoMa/scripts/evaluate_googleearth_scale_pairs.py::pair_metrics.
RMSE below is conditional on GT-correct matches, NOT estimated-H error.
"""
import numpy as np
from PIL import Image


def dense_metrics(row, root, points_a, points_b, thresholds=(1.,3.,5.,10.)):
    a=np.asarray(points_a,dtype=np.float64);b=np.asarray(points_b,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=2:raise ValueError('Expected Nx2 pairs')
    finite=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    overlap=finite.copy()
    for side,points in (('A',a),('B',b)):
        with Image.open(root/row[f'mask_{side}_overlap']) as image:mask=np.asarray(image.convert('L'))>0
        safe=np.where(np.isfinite(points),points,0)
        x,y=np.rint(safe).astype(np.int64).T
        inside=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0])&finite
        valid=np.zeros(len(a),bool);valid[inside]=mask[y[inside],x[inside]];overlap&=valid
    q=np.c_[a,np.ones(len(a))]@np.asarray(row['H_A_to_B'],dtype=np.float64).T
    valid=finite&np.isfinite(q).all(1)&(np.abs(q[:,2])>1e-8)
    error=np.full(len(a),np.inf);error[valid]=np.linalg.norm(q[valid,:2]/q[valid,2,None]-b[valid],axis=1)
    result=dict(matches=len(a),overlap_matches=int(overlap.sum()),invalid_matches=int((~valid).sum()),
                coordinate_unit='native_target_pixels',thresholds={})
    for t in thresholds:
        correct=overlap&(error<=t);n=int(correct.sum())
        result['thresholds'][str(t)]=dict(NCM=n,precision=n/max(1,len(a)),
            overlap_precision=n/max(1,int(overlap.sum())))
    correct=overlap&(error<=5);success=int(correct.sum())>=20
    result.update(success_20_correct_at_5px=success,
        legacy_RMSE_correct5_or_failure10=float(np.sqrt(np.mean(error[correct]**2))) if success else 10.,
        overlap_finite_EPE=float(error[overlap&np.isfinite(error)].mean()) if (overlap&np.isfinite(error)).any() else None)
    return result
