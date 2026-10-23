"""Hangzhou same-mother synthesis, split before cropping; no cross-source H labels."""
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
import rasterio
from PIL import Image
from .labels import project, support_labels, MATCH

CONFIG = Path(__file__).with_name('config.json')


def load_config():
    return json.loads(CONFIG.read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def mother_image(root, relative):
    with rasterio.open(Path(root)/relative) as ds:
        rgb = ds.read([1,2,3]).transpose(1,2,0)
        valid = ds.dataset_mask() == 255  # not the stale VALIDITY_BAND tag
    return rgb, valid


def select_mothers(config):
    root = Path(config['hangzhou_root'])
    spatial = json.loads((root/'spatial_splits.json').read_text())['blocks']
    candidates = defaultdict(set)
    with (root/'manifests/train.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if spatial[row['anchor_block_id']] != 'train':
                continue
            for side in 'AB':
                for path in row['source_'+side]['relative_paths']:
                    if 'tianditu' in path:
                        candidates[row['anchor_block_id']].add(path)
    old = json.loads(Path('outputs/d16_final_round/selection.json').read_text())
    exposed = {r['anchor_block_id'] for k,v in old.items() if k.startswith('hz_') and isinstance(v,list) for r in v}
    # Reserve fresh geographic blocks for calibration/evaluation; training may
    # use U0-exposed train blocks, explicitly recorded below.
    rng = np.random.default_rng(config['seed'])
    fresh = sorted(set(candidates)-exposed); rng.shuffle(fresh)
    seen = sorted(set(candidates)&exposed); rng.shuffle(seen)
    blocks = fresh + seen
    mothers = []
    for block in blocks:
        paths = sorted(candidates[block]); rng.shuffle(paths)
        bx,by = map(int,block.split(','))
        for relative in paths:
            with rasterio.open(Path(config['source_root'])/relative) as ds:
                l,b,r,t = ds.bounds
                inside = l >= 512+bx*8192 and r <= 512+(bx+1)*8192 and b >= 512+by*8192 and t <= 512+(by+1)*8192
                if ds.width < 1600 or ds.height < 1600 or not inside:
                    continue
                if not (ds.dataset_mask() == 255).all():
                    continue
                rgb = ds.read([1,2,3])
                if rgb.std() < 12:  # exclude nearly constant water/fill mothers
                    continue
            mothers.append(dict(mother_id=relative.split('/')[0],path=relative,block=block))
            break
        if len(mothers) == config['mother_count']:
            break
    if len(mothers) < config['mother_count']:
        raise ValueError(f'Only {len(mothers)} suitable independent blocks; adjust explicit experiment size')
    a,b,c = config['split_counts']
    if any(m['block'] in exposed for m in mothers[:b+c]):
        raise ValueError(f'fresh candidates={len(fresh)}, suitable fresh mothers={sum(m["block"] not in exposed for m in mothers)}; need {b+c}')
    for index,mother in enumerate(mothers):
        mother['split'] = 'cal' if index < b else 'eval' if index < b+c else 'train'
        mother['previous_u0_block_exposure'] = mother['block'] in exposed
    return dict(mothers=mothers, prior_u0_blocks_excluded_from_cal_eval=sorted(exposed),
                source='Hangzhou tianditu RGB uint8, original train geographic blocks only',
                valid_policy='full GDAL mask=255; complete mother inside one 8192m block',
                seed=config['seed'])


def window(center, angle=0., scale=1.):
    t = np.deg2rad(angle)
    R = scale*np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    C = np.eye(3); C[:2,:2] = R; C[:2,2] = center-R@np.array([255.5,255.5])
    return C


def overlap_mask(H):
    y,x = np.mgrid[:512,:512]
    target = project(np.stack((x,y),-1),H)
    return (target >= -.5).all(-1) & (target <= 511.5).all(-1)


def geometry(rng, shape, overlap):
    angle = rng.uniform(0,15)*rng.choice([-1,1])
    scale = rng.uniform(.9,1.1)
    ray = rng.uniform(0,2*np.pi)
    direction = np.array([np.cos(ray),np.sin(ray)])
    center = np.array(shape[:2][::-1])/2-.5+rng.uniform(-120,120,2)
    y,x = np.mgrid[3.5:512:8,3.5:512:8]
    points = np.stack((x,y),-1)
    lo,hi = 0.,900.
    for _ in range(20):
        distance = (lo+hi)/2
        A = window(center-direction*distance/2)
        B = window(center+direction*distance/2,angle,scale)
        H = np.linalg.inv(B)@A
        q = project(points,H)
        fraction = ((q >= -.5).all(-1)&(q <=511.5).all(-1)).mean()
        if fraction > overlap: lo = distance
        else: hi = distance
    return A,B,H,dict(rotation_degrees=float(angle),scale=float(scale),shift_distance=float(distance),shift_ray=float(ray))


def crop(rgb, transform):
    corners = project(np.array([[-.5,-.5],[511.5,-.5],[511.5,511.5],[-.5,511.5]]),transform)
    if (corners < 1).any() or (corners > np.array(rgb.shape[:2][::-1])-2).any():
        raise ValueError('crop touches mother image boundary')
    return cv2.warpPerspective(rgb,transform,(512,512),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)


def make_dataset(config):
    out = Path(config['output']); out.mkdir(exist_ok=True,parents=True)
    selection_file = out/'mother_selection.json'
    selection = json.loads(selection_file.read_text()) if selection_file.exists() else select_mothers(config)
    dump(selection_file,selection)  # authoritative split saved before any crops
    rng = np.random.default_rng(config['seed'])
    rows = []
    for split in ('train','cal','eval'):
        mothers = [m for m in selection['mothers'] if m['split']==split]
        for mi,mother in enumerate(mothers):
            rgb,_ = mother_image(config['source_root'],mother['path'])
            donor = mothers[(mi+1)%len(mothers)]
            donor_rgb,_ = mother_image(config['source_root'],donor['path'])
            donor_crop = crop(donor_rgb,window(np.array(donor_rgb.shape[:2][::-1])/2-.5))
            for case,overlap in enumerate([1.]+config['overlaps']):
                if overlap == 1:
                    A = B = window(np.array(rgb.shape[:2][::-1])/2-.5)
                    H = np.eye(3); params = dict(rotation_degrees=0.,scale=1.,shift_distance=0.)
                else:
                    A,B,H,params = geometry(rng,rgb.shape,overlap)
                image_a,image_b = crop(rgb,A),crop(rgb,B)
                # Mild fixed photometry on full overlap avoids exact duplicate RGB.
                if overlap == 1:
                    image_b = np.round(255*(image_b/255.)**1.05).astype(np.uint8)
                mask_ab,mask_ba = overlap_mask(H),overlap_mask(np.linalg.inv(H))
                base = f'{split}_{mother["mother_id"]}_{case}'
                directory = out/'data'/base; directory.mkdir(parents=True,exist_ok=True)
                Image.fromarray(image_a).save(directory/'A.png')
                Image.fromarray(image_b).save(directory/'B.png')
                np.savez_compressed(directory/'masks.npz',valid_a=np.ones((512,512),bool),valid_b=np.ones((512,512),bool),overlap_a=mask_ab,overlap_b=mask_ba)
                common = dict(split=split,mother_id=mother['mother_id'],mother_path=mother['path'],block=mother['block'],
                    H_A_to_B=H.tolist(),H_B_to_A=np.linalg.inv(H).tolist(),crop_A=A.tolist(),crop_B=B.tolist(),
                    requested_overlap=overlap,overlap_a=float(mask_ab.mean()),overlap_b=float(mask_ba.mean()),
                    crop_parameters=params,masks=str(directory/'masks.npz'),generation_version='pcf01-hangzhou-v1',
                    image_a=str(directory/'A.png'),image_b=str(directory/'B.png'),rectangle=None,donor=None)
                rows.append(dict(common,pair_id=base,kind='full' if overlap==1 else 'partial'))
                if overlap < 1:
                    labels = support_labels(H)
                    known_targets = labels['gt_target_xy'][labels['state']==MATCH]
                    for _ in range(100):
                        area = rng.uniform(*config['texture_area']); aspect = rng.uniform(.75,1.33)
                        w,h = round(512*np.sqrt(area*aspect)),round(512*np.sqrt(area/aspect))
                        center = known_targets[rng.integers(len(known_targets))]
                        x0,y0 = np.clip(np.round(center-np.array([w,h])/2).astype(int),0,[512-w,512-h])
                        rect = [int(x0),int(y0),int(x0+w),int(y0+h)]
                        if (support_labels(H,rect)['negative_kind']==2).sum()>=4: break
                    else: raise ValueError('failed to place texture inside overlap')
                    textured = image_b.copy()
                    dx,dy = rng.integers(0,513-w),rng.integers(0,513-h)
                    textured[y0:y0+h,x0:x0+w] = donor_crop[dy:dy+h,dx:dx+w]
                    Image.fromarray(textured).save(directory/'B_texture.png')
                    rows.append(dict(common,pair_id=base+'_texture',kind='texture',image_b=str(directory/'B_texture.png'),
                        rectangle=rect,texture_area=w*h/(512*512),donor=donor,donor_crop=[int(dx),int(dy),w,h]))
            print(f'{split}: mother {mi+1}/{len(mothers)}',flush=True)
    # Existing real cross-source development pairs are diagnostics only.
    old = json.loads(Path('outputs/d16_final_round/selection.json').read_text())
    hz_rows = old.get('hz_dev',[])
    for row in hz_rows:
        rows.append(dict(pair_id='hz_'+row['pair_id'],split='hz',kind=row['G'],mother_id=row['anchor_region_id'],
            image_a=str(Path(config['hangzhou_root'])/row['image_A']),image_b=str(Path(config['hangzhou_root'])/row['image_B']),
            reference_row=row,rectangle=None,donor=None))
    with (out/'pairs.jsonl').open('w') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump(out/'dataset_summary.json',dict(mothers=len(selection['mothers']),pairs=len(rows),
        split_pairs={s:sum(r['split']==s for r in rows) for s in ('train','cal','eval','hz')},
        mean_overlap_error=float(np.mean([abs(r['overlap_a']-r['requested_overlap']) for r in rows if r['split']!='hz']))))


if __name__ == '__main__':
    make_dataset(load_config())
