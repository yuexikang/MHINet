"""Read-only worst-pair GHIM regression audit and diagnostic-only crossed refits."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from mhinet.config import RuntimePaths,sha256_file
from mhinet.pretraining.model import build_shared_network
from mhinet.pretraining.data import SharedPairDataset
from mhinet.pretraining.loss import grid,sample
from mhinet.pretraining.overlap_metrics import overlap_projection_error
from mhinet.ops.geometry import safe_project_points
from mhinet.visualization.visualization import write_iteration_overlays


def rows(path):return {r['pair_id']:r for r in map(json.loads,path.open())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    before_path=Path('outputs/shared_descriptor_frozen_tier3_seed0/latest.pt')
    after_path=Path('outputs/shared_stable_v2_tier1_adapt_seed0/latest.pt')
    old=rows(Path('outputs/shared_tier3_stable_v2_baseline_tier1/validation/step_008663/pairs.jsonl'))
    new=rows(Path('outputs/shared_stable_v2_tier1_adapt_seed0/validation/step_002000/pairs.jsonl'))
    assert old.keys()==new.keys()
    selected=sorted(old,key=lambda k:new[k]['H0_overlap']['mean_px']-old[k]['H0_overlap']['mean_px'],reverse=True)[:5]
    state=torch.load(before_path,map_location='cpu',weights_only=True,mmap=True)
    root=Path('/home/disk1/Data/datasets/GoogleEarth_quadrant_tiers_stable_v2')
    runtime=replace(RuntimePaths.from_json(state['metadata']['config']['runtime']),device='cuda:0',data_root=root)
    data=SharedPairDataset(root/'val/pairs.jsonl',tier=1)
    index={r.pair_id:i for i,r in enumerate(data.index)}
    batches={k:data[index[k]] for k in selected}
    model,_=build_shared_network(runtime,lora=False); model.eval()
    saved={k:{} for k in selected}
    with torch.no_grad():
        for name,path in [('before',before_path),('after',after_path)]:
            s=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            model.load_state_dict(s['model'],strict=True)
            for k in selected:
                result=model(batches[k]['images'][None].cuda(),pyramid_scales=())
                saved[k][name]={key:result['stage1'][key].float().cpu() for key in ('coarse_warp','coarse_matchability','H_A_to_B_norm')}
                print('extracted',name,k,flush=True)
        fitter=model.stage1_head.homography_fitter.cpu()
        report=dict(before_checkpoint_sha256=sha256_file(before_path),after_checkpoint_sha256=sha256_file(after_path),
            val_manifest_sha256=sha256_file(root/'val/pairs.jsonl'),selection='top5 increase in pair-mean H0 GT-visible-overlap error',pairs=[])
        for rank,k in enumerate(selected,1):
            batch=batches[k];folder=args.output/f'{rank:02d}';folder.mkdir()
            gt=batch['H_gt_norm'][None];ma=batch['mask_A_overlap'][None];mb=batch['mask_B_overlap'][None]
            images=batch['images'];rgb=(images.permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
            panel=Image.new('RGB',(1568,830),'black');draw=ImageDraw.Draw(panel)
            panel.paste(Image.fromarray(rgb[0]),(0,46));panel.paste(Image.fromarray(rgb[1]),(784,46));draw.text((8,8),k,fill='white');draw.text((8,25),'A | B (resized to 784)',fill='white');panel.save(folder/'original_pair.jpg')
            summary=dict(rank=rank,pair_id=k,archived_before=old[k]['H0_overlap']['mean_px'],archived_after=new[k]['H0_overlap']['mean_px'],states={},crossed_refits={})
            for name in ('before','after'):
                s=saved[k][name];warp=s['coarse_warp'];weight=s['coarse_matchability'].flatten();h,w=warp.shape[-2:]
                xy=grid(h,w,'cpu');pred=warp[0].permute(1,2,0).reshape(-1,2)
                truth,valid,_=safe_project_points(gt,xy[None]);truth=truth[0]
                support=valid[0]&(truth.abs()<=1-1/784).all(1)&(sample(ma[0],xy)[:,0]>.999)&(sample(mb[0],truth)[:,0]>.999)
                error=((pred-truth)*392).norm(dim=1);keep=weight>.3;wk=weight[keep]
                points=xy[keep].numpy();hull=cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / 4 if len(points)>=3 else 0.
                bins=((xy[keep]+1)*2).long().clamp(0,3);cells=int(torch.unique(bins[:,0]+4*bins[:,1]).numel())
                selected_support=keep&support
                good=selected_support&(error<=5)
                x,y=xy.T;u,v=pred.T;ones=torch.ones_like(x);zero=torch.zeros_like(x)
                design=torch.stack((torch.stack((x,y,ones,zero,zero,zero,-u*x,-u*y),1),torch.stack((zero,zero,zero,x,y,ones,-v*x,-v*y),1)),1).reshape(-1,8)
                ww=torch.where(keep,weight,0).clamp_min(1e-8).sqrt().repeat_interleave(2)
                normal=(design*ww[:,None]).T@(design*ww[:,None])+torch.eye(8)*1e-4
                metrics=dict(selected=int(keep.sum()),gt_visible_grid=int(support.sum()),selected_gt_visible=int(selected_support.sum()),
                    selected_outside_gt_visible=int((keep&~support).sum()),selected_gt_error_gt5=int((selected_support&(error>5)).sum()),
                    selected_gt_error_gt10=int((selected_support&(error>10)).sum()),
                    selected_gt_error_mean=float(error[selected_support].mean()) if selected_support.any() else None,
                    mean_matchability=float(weight.mean()),effective_weighted_points=float(wk.sum()**2/(wk.square().sum())) if len(wk) else 0.,
                    good5_weight_fraction=float(weight[good].sum()/wk.sum()) if len(wk) else 0.,source_hull_fraction=hull,occupied_4x4_cells=cells,
                    regularized_normal_condition=float(torch.linalg.cond(normal.double())))
                metrics['H0_overlap']=overlap_projection_error(s['H_A_to_B_norm'],gt,ma,mb)
                summary['states'][name]=metrics
                write_iteration_overlays(images,gt[0],[],[],folder/name,pair_id=k,h0=s['H_A_to_B_norm'][0])
                canvas=np.ascontiguousarray((rgb[0]*.45).astype(np.uint8))
                for i in keep.nonzero().flatten().tolist():
                    color=(180,0,255) if not support[i] else ((0,255,0) if error[i]<=5 else (255,40,0))
                    pos=tuple(np.rint((xy[i].numpy()+1)*392-.5).astype(int));cv2.circle(canvas,pos,3,color,-1)
                pic=Image.new('RGB',(784,825),'black');pic.paste(Image.fromarray(canvas),(0,41));ImageDraw.Draw(pic).text((6,8),f'{name}: green GT error<=5; red >5; purple outside GT support',fill='white');pic.save(folder/f'{name}_fitting_points.png')
            for coord_name in ('before','after'):
                for weight_name in ('before','after'):
                    H,ok,_=fitter(saved[k][coord_name]['coarse_warp'],saved[k][weight_name]['coarse_matchability'])
                    summary['crossed_refits'][coord_name+'_coords_'+weight_name+'_weights']=overlap_projection_error(H,gt,ma,mb,fit_valid=bool(ok[0]))['mean_px']
            np.savez_compressed(folder/'coarse_arrays.npz',**{name+'_'+key:value.numpy() for name,s in saved[k].items() for key,value in s.items()})
            (folder/'report.json').write_text(json.dumps(summary,indent=2));report['pairs'].append(summary)
            print('PAIR',json.dumps(summary),flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
