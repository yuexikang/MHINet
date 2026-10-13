from pathlib import Path
import json,sys,collections
import numpy as np,cv2
sys.path.insert(0,'/home/disk1/MHINet')
from scripts.mrsi_gt_metrics import project
ROOT=Path('/home/disk1/MHINet/outputs/semidense_rgb512_h0joint_test512_20261012')
OLD=Path('/home/disk1/MHINet/outputs/jl1flight_pre_post_tests_512_20261004')
def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def dist(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)]
 return dict(n=len(a),mean=float(a.mean()),p10=float(np.quantile(a,.1)),p25=float(np.quantile(a,.25)),median=float(np.median(a)),p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else {}
def summary(records):
 return dict(n=len(records),P3=np.mean([r['metrics']['precision']['3'] for r in records])*100,P5=np.mean([r['metrics']['precision']['5'] for r in records])*100,SR3=sum(r['metrics']['sr']['3'] for r in records),reg=dist([r['metrics']['registration_rmse'] for r in records]))
all_stats={}
for dataset,manifest in [('GoogleEarth',Path('/home/disk1/Data/datasets/Test_Data/GoogleEarth/val/pairs.jsonl')),('JL1Flight',Path('/home/disk1/Data/datasets/Test_Data/JL1Flight/test/pairs.jsonl'))]:
 rows=read(manifest);records=read(ROOT/f'h0joint_{dataset}/pairs.jsonl');before=read(OLD/f'before_{dataset}/pairs.jsonl');after=read(OLD/f'after_{dataset}/pairs.jsonl')
 n=len(rows);out={};all_stats[dataset]=out
 out['reg_by_phase']={phase:summary(rs) for phase,rs in [('h0joint',records),('before',before),('after',after)]}
 out['stage_transitions']={}
 for a,b in [('coarse','fine'),('fine','final')]:
  ma=[r['metrics'] if a=='final' else r['stages'][a] for r in records];mb=[r['metrics'] if b=='final' else r['stages'][b] for r in records]
  out['stage_transitions'][f'{a}_to_{b}']=dict(SR3_gained=sum(not x['sr']['3'] and y['sr']['3'] for x,y in zip(ma,mb)),SR3_lost=sum(x['sr']['3'] and not y['sr']['3'] for x,y in zip(ma,mb)),P1_improved=sum(y['precision']['1']>x['precision']['1'] for x,y in zip(ma,mb)),P3_improved=sum(y['precision']['3']>x['precision']['3'] for x,y in zip(ma,mb)),P5_improved=sum(y['precision']['5']>x['precision']['5'] for x,y in zip(ma,mb)),reg_improved=sum(y['registration_rmse']<x['registration_rmse'] for x,y in zip(ma,mb)))
 out['new_vs_before']=dict(SR3_gained=sum(not x['metrics']['sr']['3'] and y['metrics']['sr']['3'] for x,y in zip(before,records)),SR3_lost=sum(x['metrics']['sr']['3'] and not y['metrics']['sr']['3'] for x,y in zip(before,records)),P5_improved=sum(y['metrics']['precision']['5']>x['metrics']['precision']['5'] for x,y in zip(before,records)))
 out['fail_thresholds']={str(t):sum(r['metrics']['registration_rmse']>t for r in records) for t in [3,5,10,20,100]}
 out['P5_gt80_SR3_fail']=sum(r['metrics']['precision']['5']>.8 and not r['metrics']['sr']['3'] for r in records)
 out['P5_lt20']=sum(r['metrics']['precision']['5']<.2 for r in records)
 out['support_fraction']=dist([r['metrics']['common_support_matches']/max(1,r['metrics']['matches']) for r in records])
 out['mask_P5_minus_headline_pp']=dist([r['mask_metrics']['thresholds']['5.0']['precision']*100-r['metrics']['precision']['5']*100 for r in records])
 out['NM']=dist([r['metrics']['matches'] for r in records]);out['NM_exact12000']=sum(r['metrics']['matches']==12000 for r in records)
 # Geometry only; scale is local area-equivalent scale sqrt(abs(det J)) at source center.
 scales=[]
 for row in rows:
  H=np.asarray(row['H_A_to_B']);center=np.array([[255.5,255.5],[256.5,255.5],[255.5,256.5]])
  proj,_=project(center,H);J=np.column_stack((proj[1]-proj[0],proj[2]-proj[0]));scales.append(np.sqrt(abs(np.linalg.det(J))))
 out['local_scale']=dist(scales);out['overlap_A']=dist([r['overlap_fractions'][0] for r in rows]);out['overlap_B']=dist([r['overlap_fractions'][1] for r in rows])
 for name,values in [('scale',scales),('overlap_A',[r['overlap_fractions'][0] for r in rows]),('overlap_B',[r['overlap_fractions'][1] for r in rows])]:
  bounds=np.quantile(values,[0,.25,.5,.75,1]);out[name+'_quartiles']=[]
  for i in range(4):
   ids=[j for j,v in enumerate(values) if v>=bounds[i] and (v<bounds[i+1] or i==3)]
   out[name+'_quartiles'].append(dict(range=[float(bounds[i]),float(bounds[i+1])],**summary([records[j] for j in ids])))
 # On existing final coordinates, measure displacement/coverage/error distribution. No refitting/filter sweeps.
 x,y=np.meshgrid(np.linspace(0,511,20),np.linspace(0,511,20));grid=np.stack((x.ravel(),y.ravel()),axis=1)
 spatial=[]
 for i,(row,r) in enumerate(zip(rows,records)):
  H=np.asarray(row['H_A_to_B']);truth,ok=project(grid,H);ok&=(truth>=0).all(1)&(truth<=511).all(1)
  pred,_=project(grid[ok],r['metrics']['H_estimated']);res=pred-truth[ok];translation=float(np.linalg.norm(res.mean(0)));centered=float(np.sqrt(np.mean(np.sum((res-res.mean(0))**2,axis=1))))
  with np.load(ROOT/f'h0joint_{dataset}/matches/{i:05d}.npz',allow_pickle=False) as z:a,b,c=z['points_a'],z['points_b'],z['confidence']
  ta,valid=project(a,H);valid&=(a>=0).all(1)&(a<=511).all(1)&(ta>=0).all(1)&(ta<=511).all(1)&(b>=0).all(1)&(b<=511).all(1)
  errs=np.linalg.norm(ta-b,axis=1)
  def hull_area(p):return float(cv2.contourArea(cv2.convexHull(np.asarray(p,np.float32)))) if len(p)>=3 else 0.
  visible_hull=hull_area(grid[ok]);correct5=valid&(errs<=5);correct3=valid&(errs<=3)
  cover5=hull_area(a[correct5])/visible_hull if visible_hull else 0.;cover3=hull_area(a[correct3])/visible_hull if visible_hull else 0.
  # Fixed 16x16 source bins with GT-visible centers; occupancy indicates reach, not uniformly accurate field.
  xx,yy=np.meshgrid((np.arange(16)+.5)*512/16,(np.arange(16)+.5)*512/16);centers=np.stack((xx.ravel(),yy.ravel()),axis=1);tc,vv=project(centers,H);vv&=(tc>=0).all(1)&(tc<=511).all(1)
  def occ(m):
   p=a[m];binids=np.clip((p/32).astype(int),0,15);ids=np.unique(binids[:,1]*16+binids[:,0]);return np.count_nonzero(vv[ids])/max(1,vv.sum())
  residual_self,_=project(a,r['metrics']['H_estimated']);eself=np.linalg.norm(residual_self-b,axis=1)
  spatial.append(dict(pair_id=row['pair_id'],success=r['metrics']['sr']['3'],reg=r['metrics']['registration_rmse'],translation=translation,centered=centered,inlier_ratio=r['metrics']['ransac_inliers']/len(a),self_epe=float(np.mean(eself)),hull5=cover5,hull3=cover3,occupancy5=occ(correct5),occupancy3=occ(correct3),error1_3=float(np.mean(valid&(errs>1)&(errs<=3))),error3_5=float(np.mean(valid&(errs>3)&(errs<=5))),error5_10=float(np.mean(valid&(errs>5)&(errs<=10))),error_gt10=float(np.mean(valid&(errs>10))),confidence_median=float(np.median(c)),unique_a=len(np.unique(a,axis=0))))
  if (i+1)%500==0:print(dataset,'processed',i+1,flush=True)
 out['spatial_all']={k:dist([r[k] for r in spatial]) for k in spatial[0] if k not in ['pair_id','success']}
 out['spatial_success_fail']={str(flag):{k:dist([r[k] for r in spatial if r['success']==flag]) for k in ['reg','translation','centered','inlier_ratio','self_epe','hull5','hull3','occupancy5','occupancy3','error3_5','error5_10','error_gt10']} for flag in [True,False]}
 out['unique_a_all']=all(s['unique_a']==records[i]['metrics']['matches'] for i,s in enumerate(spatial))
 out['outliers']=sorted(spatial,key=lambda s:s['reg'],reverse=True)[:10]
 if dataset=='JL1Flight':
  base=collections.defaultdict(list)
  for row,r in zip(rows,records):base[row['base_id']].append(r)
  out['base_success_count_distribution']=dict(collections.Counter(sum(x['metrics']['sr']['3'] for x in rs) for rs in base.values()))
  out['base_P5_range']=dist([max(x['metrics']['precision']['5'] for x in rs)-min(x['metrics']['precision']['5'] for x in rs) for rs in base.values()])
  out['group_geometry']={g:dict(local_scale=dist([s for s,row in zip(scales,rows) if row['transform_group']==g]),overlap_A=dist([row['overlap_fractions'][0] for row in rows if row['transform_group']==g])) for g in ['t0','t1','t2','t3','t4']}
Path('/tmp/mhinet_round0001_cpu_stats.json').write_text(json.dumps(all_stats,ensure_ascii=False,indent=2))
print('saved /tmp/mhinet_round0001_cpu_stats.json')
