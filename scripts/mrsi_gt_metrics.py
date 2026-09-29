"""Native-pixel matching and registration metrics against supplied homography GT."""
import numpy as np
import cv2


def project(points,H):
    p=np.asarray(points,dtype=np.float64).reshape(-1,2)
    h=np.c_[p,np.ones(len(p))]@np.asarray(H,dtype=np.float64).T
    valid=np.isfinite(h).all(1)&(np.abs(h[:,2])>1e-10)
    result=np.full_like(p,np.nan);result[valid]=h[valid,:2]/h[valid,2:]
    return result,valid


def score_matches(a,b,size_a,size_b,H_gt):
    a=np.asarray(a,dtype=np.float64).reshape(-1,2)
    b=np.asarray(b,dtype=np.float64).reshape(-1,2)
    H_gt=np.asarray(H_gt,dtype=np.float64)
    if a.shape!=b.shape:raise ValueError('Unequal point arrays')
    if H_gt.shape!=(3,3) or not np.isfinite(H_gt).all() or np.linalg.matrix_rank(H_gt)<3:raise ValueError('Invalid GT')
    n=len(a);finite=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    truth,projectable=project(a,H_gt);scorable=finite&projectable
    inside=scorable&(a>=0).all(1)&(a<=np.asarray(size_a)-1).all(1)&(truth>=0).all(1)&(truth<=np.asarray(size_b)-1).all(1)&(b>=0).all(1)&(b<=np.asarray(size_b)-1).all(1)
    errors=np.linalg.norm(truth-b,axis=1)
    correct={str(t):int(np.count_nonzero(inside&(errors<=t))) for t in (1,3,5)}
    result=dict(matches=n,finite_matches=int(finite.sum()),scorable_matches=int(scorable.sum()),common_support_matches=int(inside.sum()),
        precision={t:count/max(1,n) for t,count in correct.items()},ncm=correct,
        matching_rmse=float(np.sqrt(np.mean(errors[scorable]**2))) if scorable.any() else None,
        matching_epe=float(errors[scorable].mean()) if scorable.any() else None,
        match_success20_at5=correct['5']>=20,registration_rmse=None,
        sr={str(t):False for t in (1,3,5)},fit_failure=None,H_estimated=None)
    if finite.sum()<4:
        result['fit_failure']='fewer_than_four_matches';return result
    cv2.setRNGSeed(0)
    # GT never participates in correspondence selection or homography fitting.
    try:H,inliers=cv2.findHomography(a[finite],b[finite],cv2.RANSAC,3.,maxIters=10000,confidence=.999)
    except cv2.error:H=None;inliers=None
    if H is None or not np.isfinite(H).all() or np.linalg.matrix_rank(H)<3:
        result['fit_failure']='invalid_homography';return result
    x,y=np.meshgrid(np.linspace(0,size_a[0]-1,20),np.linspace(0,size_a[1]-1,20))
    grid=np.stack((x.ravel(),y.ravel()),axis=1)
    target,valid=project(grid,H_gt)
    valid=valid&(target>=0).all(1)&(target<=np.asarray(size_b)-1).all(1)
    result['registration_grid_points']=int(valid.sum())
    if not valid.any():raise ValueError('GT has no visible evaluation grid')
    predicted,pvalid=project(grid[valid],H)
    if not pvalid.all():
        result['fit_failure']='invalid_grid_projection';return result
    rmse=float(np.sqrt(np.mean(np.sum((predicted-target[valid])**2,axis=1))))
    result.update(registration_rmse=rmse,sr={str(t):rmse<=t for t in (1,3,5)},
        H_estimated=H.tolist(),ransac_inliers=int(inliers.sum()))
    return result

def aggregate(records):
    def avg(key):
        values=[r['metrics'][key] for r in records if r['metrics'][key] is not None]
        return float(np.mean(values)) if values else None
    n=len(records);metrics=[r['metrics'] for r in records]
    return dict(pairs=n,precision={t:float(np.mean([m['precision'][t] for m in metrics])) for t in ('1','3','5')},
        precision_micro={t:sum(m['ncm'][t] for m in metrics)/max(1,sum(m['matches'] for m in metrics)) for t in ('1','3','5')},
        matching_rmse=avg('matching_rmse'),matching_rmse_pairs=sum(m['matching_rmse'] is not None for m in metrics),
        matching_epe=avg('matching_epe'),registration_rmse=avg('registration_rmse'),
        registration_rmse_pairs=sum(m['registration_rmse'] is not None for m in metrics),
        sr={t:float(np.mean([m['sr'][t] for m in metrics])) for t in ('1','3','5')},
        match_success20_at5=float(np.mean([m['match_success20_at5'] for m in metrics])),
        mean_ncm={t:float(np.mean([m['ncm'][t] for m in metrics])) for t in ('1','3','5')},
        mean_matches=avg('matches'),inference_failure_rate=float(np.mean([r['failure_reason']!='none' for r in records])),
        registration_failure_rate=float(np.mean([m['fit_failure'] is not None for m in metrics])),
        inference_ms_mean=float(np.mean([r['timing']['wall_total_ms'] for r in records])),
        inference_ms_p95=float(np.percentile([r['timing']['wall_total_ms'] for r in records],95)))
