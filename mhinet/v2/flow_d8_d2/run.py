"""Frozen Flow → D8 → D2 evaluation; all errors are native 512-image pixels."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from mhinet.v2.u0 import load_u0
from mhinet.v2.frozen_locator import load_locator_weights
from mhinet.v2.semidense import SemidenseMatcherV2
from mhinet.v2.config import SemidenseConfigV2
from mhinet.v2.data import GeometryPairDataset
from .matching import infer

CHECKPOINT = 'outputs/hangzhou_cleaned_restart/v2_20261022/best.pt'
VARIANTS = ('global', 'local', 'local_mat01')

def dataset(domain):
    selection = json.loads(Path('outputs/d16_final_round/selection.json').read_text())
    if domain == 'hangzhou':
        return GeometryPairDataset('/home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4', selection['hz_dev'])
    rows = [json.loads(line) for line in Path('outputs/pcf01/pairs.jsonl').read_text().splitlines()]
    rows = [dict(r, image_A=r['image_a'], image_B=r['image_b'], size_A=[512,512], size_B=[512,512], G=r['kind'])
            for r in rows if r['split']=='eval' and r['kind'] in ('full','partial')]
    return GeometryPairDataset('/', rows)

def load(device):
    model = load_u0(device=device)
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False, mmap=True)
    missing, unexpected = model.load_state_dict({k[7:]:v for k,v in state['model'].items() if k.startswith('shared.')}, strict=False)
    assert not unexpected and all(k.startswith('descriptor.encoder.frozen_dinov3.model.') for k in missing), (missing,unexpected)
    load_locator_weights(model, 'weight/mhi_v2_u0.pt')
    matcher = SemidenseMatcherV2(SemidenseConfigV2(**state['matcher'])).to(device)
    matcher.load_state_dict({k[8:]:v for k,v in state['model'].items() if k.startswith('matcher.')})
    def prohibit_qrru(*args):
        raise RuntimeError('QRRU must not run in this experiment')
    matcher.qrru.register_forward_pre_hook(prohibit_qrru)
    return model.requires_grad_(False).eval(), matcher.requires_grad_(False).eval()

def measure(a,b,gt):
    target,known,visible = gt.query((a+.5)/256-1)
    good = known & visible
    error = ((target+1)*256-.5-b).norm(dim=-1)
    result = dict(n=len(a), visible=int(good.sum()), error_sum=float(error[good].sum()))
    result.update({f'correct{t}':int((good & (error<=t)).sum()) for t in (1,3,5,10)})
    return result, error, good

def sample(field, points):
    return F.grid_sample(field.float(), ((points+.5)/256-1)[None,None], align_corners=False, padding_mode='border')[0,:,0].T

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=['hangzhou','synthetic'],required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path,default=Path('outputs/flow_d8_d2'))
    args=p.parse_args();torch.set_num_threads(2)
    ds=dataset(args.domain);model,matcher=load(args.device)
    if args.checkpoint:
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model.vgg.load_state_dict(state['vgg']);model.dedode.load_state_dict(state['dedode'])
        matcher.load_state_dict({**matcher.state_dict(),**state['temperatures']})
    out=args.output/args.domain;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    def tick():
        torch.cuda.synchronize(args.device);return time.perf_counter()
    for i in range(min(len(ds),args.limit or len(ds))):
        item=ds[i];images=item['images'].to(args.device);gt=item['geometry'].to(args.device)
        if i==0:
            warm=model(images,pyramid_scales=(8,4,2))
            for v in VARIANTS: infer(warm,matcher,v)
            baseline=matcher.infer(warm,refine=False)[0];ours=infer(warm,matcher,'global')
            for key in ('points_a','points_b','confidence'):
                torch.testing.assert_close(ours[key],baseline[key],rtol=1e-4,atol=1e-4)
            print('Global path matches existing inference; QRRU guard active.',flush=True)
            del warm,baseline,ours
        start=tick();shared=model(images,pyramid_scales=(8,4,2));feature_ms=(tick()-start)*1000
        yy,xx=torch.meshgrid(torch.arange(32,device=args.device),torch.arange(32,device=args.device),indexing='ij')
        a=(torch.stack((xx,yy),-1).reshape(-1,2)+.5)*16-.5
        b=(shared['flow16_target_norm'][0].flatten(1).T+1)*256-.5
        base,_,_=measure(a,b,gt)
        record=dict(pair_id=item['pair_id'][0],group=item['group'],feature_ms=feature_ms,flow=base,variants={})
        arrays=dict(flow_a=a.cpu().numpy(),flow_b=b.cpu().numpy(),image_a=images[0,0].cpu().numpy(),image_b=images[0,1].cpu().numpy())
        for v in VARIANTS:
            start=tick();r=infer(shared,matcher,v);elapsed=(tick()-start)*1000
            coarse,_,_=measure(r['coarse_a'],r['coarse_b'],gt)
            fine,err,vis=measure(r['points_a'],r['points_b'],gt)
            initial=r['points_a']+sample(shared['flow16_norm'],r['points_a'])*256
            usable=sample(shared['flow16_valid'].float(),r['points_a']).flatten()>=1-1e-6
            before,be,bv=measure(r['points_a'][usable],initial[usable],gt)
            after,ae,av=measure(r['points_a'][usable],r['points_b'][usable],gt)
            yy,xx=torch.meshgrid(torch.arange(64,device=args.device),torch.arange(64,device=args.device),indexing='ij')
            source=(torch.stack((xx,yy),-1).reshape(-1,2)+.5)*8-.5
            target,known,visible=gt.query((source+.5)/256-1)
            truth=((target+1)*32-.5).round()
            in_window=((truth-r['prior'].target_uv.round()).abs()<=4).all(-1)
            retained=torch.ones_like(visible) if v=='global' else r['prior'].valid
            covered=visible & retained & (torch.ones_like(visible) if v=='global' else in_window)
            d8target,_,d8vis=gt.query((r['coarse_a']+.5)/256-1)
            parent_correct=(((d8target+1)*256-.5-r['coarse_b']).abs()<=4).all(-1)&d8vis
            record['variants'][v]=dict(d8=coarse,d2=fine,paired_flow=before,paired_d2=after,
                improved=int(((ae<be)&av&bv).sum()),worsened=int(((ae>be)&av&bv).sum()),
                match_ms=elapsed,source_queries=r['source_queries'],edges=r['edges'],fine_before_cap=r['fine_before_cap'],
                visible_queries=int(visible.sum()),retained_visible=int((visible&retained).sum()),search_covered=int(covered.sum()),
                correct_d8_parent=int(parent_correct.sum()))
            arrays.update({v+'_'+k:r[k].cpu().numpy() for k in ('coarse_a','coarse_b','points_a','points_b')})
            arrays[v+'_error']=err.cpu().numpy();arrays[v+'_visible']=vis.cpu().numpy()
        np.savez_compressed(out/f'{i:03d}.npz',**arrays)
        rows.append(record)
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print(f'{args.domain} {i+1}/{len(ds)} {record["pair_id"]}',flush=True)
    print('DONE',flush=True)

if __name__=='__main__': main()
