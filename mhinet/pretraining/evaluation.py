"""Full-gallery descriptor retrieval, separate from sampled training accuracy."""
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from mhinet.ops.geometry import safe_project_points
from mhinet.pretraining.loss import grid, sample, inverse_gt


@torch.no_grad()
def retrieval_metrics(pyramid, H, masks, *, seed=0, queries=32, images=None, folder=None):
    results = {}
    reverse = inverse_gt(H)
    for scale, pair in pyramid.items():
        for name, src, dst, transform in (('ab',0,1,H[0]), ('ba',1,0,reverse[0])):
            a, b = pair[0,src], pair[0,dst]
            h,w = a.shape[-2:]
            xy = grid(h,w,a.device)
            targets, valid, _ = safe_project_points(transform[None],xy[None])
            targets, valid = targets[0], valid[0]
            bound = targets.new_tensor([1-1/w,1-1/h])
            valid &= (targets.abs() <= bound).all(-1)
            valid &= sample(masks[src][0],xy)[:,0] > .999
            valid &= sample(masks[dst][0],targets)[:,0] > .999
            indices = valid.nonzero().flatten()
            gen = torch.Generator(device=a.device).manual_seed(seed)
            indices = indices[torch.randperm(len(indices),device=a.device,generator=gen)[:queries]]
            if len(indices) == 0:
                results[f'D{scale}_{name}'] = {'queries':0,'errors_input_px':[]}
                continue
            descriptors = F.normalize(sample(a,xy[indices]),dim=-1)
            best = torch.full((len(indices),),-float('inf'),device=a.device)
            locations = torch.zeros(len(indices),dtype=torch.long,device=a.device)
            flat = b.flatten(1)
            for offset in range(0,h*w,8192):
                gallery = F.normalize(flat[:,offset:offset+8192].float(),dim=0)
                scores = descriptors @ gallery
                value, loc = scores.max(-1)
                update = value > best
                locations[update] = loc[update] + offset
                best = torch.maximum(best,value)
            predicted = xy[locations]
            errors = ((predicted-targets[indices])*392).norm(dim=-1)
            results[f'D{scale}_{name}'] = dict(queries=len(indices),errors_input_px=errors.tolist(),
                mean_input_px=float(errors.mean()),
                **{f'recall_{t}px':float((errors<=t).float().mean()) for t in (1,3,5)})
            if folder is not None and name == 'ab':
                folder.mkdir(parents=True,exist_ok=True)
                rgb = (images[0].cpu().permute(0,2,3,1).clamp(0,1).numpy()*255).astype('uint8')
                canvas = Image.new('RGB',(1568,784))
                canvas.paste(Image.fromarray(rgb[0]),(0,0)); canvas.paste(Image.fromarray(rgb[1]),(784,0))
                draw = ImageDraw.Draw(canvas)
                for q, gt, pred in zip(xy[indices[:8]],targets[indices[:8]],predicted[:8]):
                    q,gt,pred = [(v.cpu()+1)*392-.5 for v in (q,gt,pred)]
                    qa,ga,pa = (float(q[0]),float(q[1])),(float(gt[0]+784),float(gt[1])),(float(pred[0]+784),float(pred[1]))
                    draw.line([qa,ga],fill='lime',width=2); draw.line([qa,pa],fill='red',width=1)
                    for point,color in ((qa,'yellow'),(ga,'lime'),(pa,'red')):
                        x,y=point; draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                canvas.save(folder/f'D{scale}_global_matches.png')
    return results
