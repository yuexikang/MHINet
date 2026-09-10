"""Localize batched GHIM numerical drift without changing pretrained weights."""
import json
import torch
from mhinet.config import RuntimePaths
from mhinet.models.model import build_model
from mhinet.engine.train import _safe_train_dataset, _seed_everything
from mhinet.ops.geometry import image_corners, safe_project_points, normalized_to_pixel

_seed_everything(0)
runtime = RuntimePaths.from_json('configs/runtime_paths.server.json')
model, _ = build_model(runtime)
model.eval()
p = model.feature_provider
dataset, _ = _safe_train_dataset(runtime, max_pairs=2)
images = torch.stack([dataset[i]['images'] for i in range(2)]).to(runtime.device)
corners = image_corners((784,784), normalized=True, device=images.device).unsqueeze(0).expand(2,-1,-1)
def pixel(h):
    xy, _, _ = safe_project_points(h.float(), corners)
    return normalized_to_pixel(xy, (784,784))
def head(c, split=False):
    if split:
        return torch.cat([p.stage1_head(x, position_dtype=torch.float32)['H_A_to_B_norm'] for x in c.split(1)])
    return p.stage1_head(c, position_dtype=torch.float32)['H_A_to_B_norm']
def context(d, split=False):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return torch.cat([p.mvt(x) for x in d.split(1)]) if split else p.mvt(d)
with torch.no_grad():
    db, _ = p._extract_batched_descriptors(images)
    ds = torch.cat([p._extract_batched_descriptors(x)[0] for x in images.split(1)])
    cb, cs = context(db), context(ds, True)
    ref = pixel(head(cs, True))
    results = {'dino_max_abs': float((db-ds).abs().max()),
               'context_max_abs': float((cb-cs).abs().max())}
    for name, c, split in [('batch',cb,False),('split_head',cb,True),
                            ('serial_dino',context(ds),False),
                            ('serial_dino_mvt',cs,False),
                            ('batch_dino_split_mvt_head',context(db,True),True)]:
        results[name+'_corner_max_px'] = float((pixel(head(c,split))-ref).abs().max())
print(json.dumps(results, indent=2))
