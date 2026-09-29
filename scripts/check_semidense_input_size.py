"""Bounded E-weight regression and 512 forward/backward check; no optimizer update."""
import argparse,json,os,sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from mhinet.config import RuntimePaths,sha256_file
from mhinet.downstream.semidense import SemidenseConfig,SemidenseMatcher
from mhinet.downstream.training import SemidenseSystem
from mhinet.pretraining.model import build_shared_network
from mhinet.pretraining.data import SharedPairDataset
from mhinet.dataio.data import load_rgb_bicubic
from scripts.semidense_timing import timed_inference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);torch.manual_seed(0)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True);meta=state['metadata']
    runtime=replace(RuntimePaths.from_json(meta['config']['runtime']),device='cuda:0')
    config=SemidenseConfig(**meta['matcher']);assert config.input_size==784
    shared,_=build_shared_network(runtime,lora=False)
    model=SemidenseSystem(shared,SemidenseMatcher(config)).cuda();model.load_state_dict(state['model'],strict=True);del state
    rows=[json.loads(x) for x in Path('artifacts/mrsi_expanded_e_manifest.jsonl').read_text().splitlines()]
    args.output.mkdir(parents=True)
    records=[];model.eval()
    with torch.no_grad():
        for index in (0,100,200):
            row=rows[index];sizes=[tuple(Image.open(row['image'+str(i)]).size) for i in (0,1)]
            for size in (784,512):
                model.matcher.config=replace(config,input_size=size)
                images=torch.stack([load_rgb_bicubic(row['image'+str(i)],size=size) for i in (0,1)])[None].cuda()
                features=model(images)
                shapes={str(k):list(v.shape) for k,v in features['pyramid'].items()}
                assert all(v.shape[-2:]==(size//k,size//k) for k,v in features['pyramid'].items())
                assert tuple(features['token_size'])==(size//16,size//16)
                warm=model.matcher.infer(features,[sizes],diagnostics=True)[0]
                result,timing=timed_inference(model,images,[sizes]);r=result[0]
                for key in ('points_a','points_b','confidence'):torch.testing.assert_close(r[key],warm[key],rtol=0,atol=0)
                regression=None
                if size==784:
                    path=Path('outputs/mrsi_expanded_e_gt_20260929/matches')/(row['id'].replace('/','_')+'.npz')
                    with np.load(path) as old:
                        regression={k:bool(np.array_equal(r[k].cpu().numpy(),old[k])) for k in ('points_a','points_b','confidence')}
                    if not all(regression.values()):raise AssertionError(f'784 prediction regression: {regression}')
                elif index==0:
                    from mhinet.downstream.visualize import render_pair
                    from mhinet.ops.geometry import pixel_homography_to_normalized
                    from torch.utils.data import default_collate
                    H=pixel_homography_to_normalized(torch.tensor(np.load(row['gt']),dtype=torch.float64),(sizes[0][1],sizes[0][0]),(sizes[1][1],sizes[1][0])).float()
                    batch=default_collate([dict(pair_id=row['id'],H_gt_norm=H,size_A=torch.tensor(sizes[0]),size_B=torch.tensor(sizes[1]),
                        mask_A_overlap=torch.ones(1,size,size),mask_B_overlap=torch.ones(1,size,size))])
                    render_pair(model,features,images,batch,warm,args.output/'visual512',args.output/'calibration',False)
                item=dict(index=index,pair_id=row['id'],input_size=size,shapes=shapes,matches=len(r['points_a']),failure_reason=r['failure_reason'],
                    token_size=list(features['token_size']),historical_784_bitwise_equal=regression,timing=timing)
                records.append(item);print(json.dumps(item),flush=True)
                del features,images,r,result,warm
    # Real 512 training batch with native GT/masks from the existing GoogleEarth data.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model.matcher.config=replace(config,input_size=512);model.train();model.zero_grad(set_to_none=True)
    dataset=SharedPairDataset(runtime.data_root/'train/pairs.jsonl',tier=3,image_size=512,max_pairs=1)
    sample=dataset[0];batch={k:v[None].cuda() for k,v in sample.items() if isinstance(v,torch.Tensor)}
    features=model(batch['images']);loss,info=model.matcher.training_losses(features,batch['H_gt_norm'],batch['mask_A_overlap'],batch['mask_B_overlap'])
    loss.backward();assert torch.isfinite(loss)
    gradients={}
    for group in model.optimizer_groups():
        grads=[p.grad for p in group['params'] if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        norm=float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt());assert norm>0
        gradients[group['name']]=norm
    frozen=[n for n,p in model.shared.named_parameters() if not p.requires_grad and p.grad is not None];assert not frozen
    report=dict(complete=True,checkpoint_sha256=sha256_file(args.checkpoint),script_sha256=sha256_file(Path(__file__)),
        scope='Three fixed pairs at both sizes; 784 bitwise regression versus stored original E predictions; one 512 visualization and full backward without optimizer update. Not a full accuracy or speed benchmark.',
        inference=records,backward512=dict(loss=float(loss.detach()),metrics=info,gradient_norms=gradients,frozen_with_grad=frozen),
        implementation={str(x):sha256_file(x) for x in [Path('mhinet/models/feature_provider.py'),*Path('mhinet/downstream').glob('*.py')]})
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Input-size checks completed',flush=True)

if __name__=='__main__':main()
