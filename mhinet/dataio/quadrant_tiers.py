"""Constrained quadrant homographies and visibility-aware three-tier synthesis."""
import json
import cv2
import numpy as np


def quad(rng, w, h):
    q=(rng.random((4,2))*.5+np.array([[0,0],[.5,0],[.5,.5],[0,.5]]))*[w-1,h-1]
    q=q.astype(np.float32)
    if not cv2.isContourConvex(q) or cv2.contourArea(q)<.3*(w-1)*(h-1): return None
    u=np.roll(q,1,axis=0)-q; v=np.roll(q,-1,axis=0)-q
    if np.linalg.norm(v,axis=1).min()<.15*min(w,h): return None
    angles=np.degrees(np.arccos(np.clip((u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)),-1,1)))
    return q if angles.min()>=25 and angles.max()<=155 else None


def legal(q,w,h):
    if (q<0).any() or (q[:,0]>w-1).any() or (q[:,1]>h-1).any(): return False
    # Rotation must retain the assigned quadrants as well as the mother bounds.
    return bool(q[0,0]<w/2 and q[0,1]<h/2 and q[1,0]>=w/2 and q[1,1]<h/2
                and q[2,0]>=w/2 and q[2,1]>=h/2 and q[3,0]<w/2 and q[3,1]>=h/2)


def mask_project(H, size, other):
    return cv2.warpPerspective(other,H,size,flags=cv2.INTER_NEAREST)


def radiation(img,rng):
    brightness,contrast,gamma=rng.uniform(.8,1.2,3)
    x=img.astype(np.float32)/255
    x=np.clip((x-.5)*contrast+.5,0,1)
    x=np.clip(x**gamma*brightness,0,1)
    return np.rint(x*255).astype(np.uint8),dict(brightness=float(brightness),contrast=float(contrast),gamma=float(gamma))


def occlude(img,rng):
    h,w=img.shape[:2]; fraction=float(rng.uniform(.1,.3))
    # A single non-wrapping rectangle; its measured area is constrained too.
    bw=int(round(w*np.sqrt(fraction)*rng.uniform(.8,1.2)))
    bw=max(1,min(w,bw)); bh=max(1,min(h,int(round(fraction*w*h/bw))))
    if not .1<=bw*bh/(w*h)<=.3: return occlude(img,rng)
    x=int(rng.integers(0,w-bw+1)); y=int(rng.integers(0,h-bh+1))
    kind=str(rng.choice(['black','gray','color','noise']))
    result=img.copy(); patch=result[y:y+bh,x:x+bw]
    if kind=='noise': patch[:]=rng.integers(0,256,patch.shape,dtype=np.uint8)
    else: patch[:]=0 if kind=='black' else (127 if kind=='gray' else rng.integers(0,256,3))
    visible=np.ones((h,w),np.uint8);visible[y:y+bh,x:x+bw]=0
    return result,visible,dict(kind=kind,box_xywh=[x,y,bw,bh],area_fraction=bw*bh/(w*h))


def generate(parent_a,parent_b,root,pair_id,pair_index,seed,config,metadata):
    if metadata['parent_image_A']!=metadata['parent_image_B']: raise ValueError('Same parent required')
    tier=int(metadata['tier']);ratio=float(metadata['resolution_ratio'])
    rng=np.random.default_rng(seed);h,w=parent_a.shape[:2]
    sizes=[(w,h),(round(w*ratio),round(h*ratio))]
    for attempt in range(20000):
        qa,qb=quad(rng,w,h),quad(rng,w,h)
        if qa is None or qb is None: continue
        theta=float(rng.uniform(-30,30));angle=np.radians(theta)
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        qb=((qb-[w/2,h/2])@R.T+[w/2,h/2]).astype(np.float32)
        if not legal(qb,w,h): continue
        area,_=cv2.intersectConvexConvex(qa,qb)
        fractions=[area/cv2.contourArea(q) for q in (qa,qb)]
        if min(fractions)<(.65 if tier==3 else .5): continue
        Ts=[]
        for q,(ow,oh) in zip((qa,qb),sizes):
            T=cv2.getPerspectiveTransform(q,np.float32([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]]))
            Ts.append(T)
        if any(not np.isfinite(t).all() or np.linalg.cond(t)>1e8 for t in Ts): continue
        if any(not (np.all(z>1e-8) or np.all(z<-1e-8)) for z in [(np.c_[q,np.ones(4)]@t[2]) for q,t in zip((qa,qb),Ts)]): continue
        H=Ts[1]@np.linalg.inv(Ts[0]);H/=H[2,2];inv=np.linalg.inv(H)
        safe=True
        for t,(ow,oh) in zip((H,inv),sizes):
            z=np.array([[0,0,1],[ow-1,0,1],[ow-1,oh-1,1],[0,oh-1,1]])@t[2]
            safe &= bool(np.isfinite(t).all() and (np.all(z>1e-8) or np.all(z<-1e-8)))
        if not safe: continue
        stability = None
        if metadata.get('stable_geometry', False):
            from .stable_geometry import check_geometry
            accepted, stability = check_geometry(H, sizes[0], sizes[1])
            if not accepted: continue
        va,vb=[np.ones((s[1],s[0]),np.uint8) for s in sizes]
        ma=mask_project(inv,sizes[0],vb); mb=mask_project(H,sizes[1],va)
        if min(ma.mean(),mb.mean())<.5: continue
        images=[cv2.warpPerspective(parent_a,t,s) for t,s in zip(Ts,sizes)]
        photo=[{},{}];occ=[{},{}]
        if tier>=2:
            for i in range(2):images[i],photo[i]=radiation(images[i],rng)
        if tier==3:
            # Balanced A-only, B-only, both; never identical masks by construction.
            selected=[[0],[1],[0,1]][int(rng.integers(3))]
            vis=[va,vb]
            for i in selected:images[i],vis[i],occ[i]=occlude(images[i],rng)
            va,vb=vis
        visible_a=va*mask_project(inv,sizes[0],vb)
        visible_b=vb*mask_project(H,sizes[1],va)
        if min(visible_a.mean(),visible_b.mean())<(.3 if tier==3 else .5):continue
        break
    else: raise RuntimeError(f'Sampling exhausted: {pair_id}')
    row={**metadata,'pair_id':pair_id,'pair_seed':seed,'pair_index':pair_index,
         'size_A':list(sizes[0]),'size_B':list(sizes[1]),'source_image_size':[w,h],
         'quad_A_source':qa.tolist(),'quad_B_source':qb.tolist(),
         'T_source_to_A':Ts[0].tolist(),'T_source_to_B':Ts[1].tolist(),
         'H_A_to_B':H.tolist(),'H_B_to_A':inv.tolist(),'rotation_B_degrees':theta,
         'mother_overlap_fractions':fractions,'geometric_overlap_fractions':[float(ma.mean()),float(mb.mean())],
         'visible_overlap_fractions':[float(visible_a.mean()),float(visible_b.mean())],
         'photometric':photo,'occlusion':occ,'attempts':attempt+1,
         'geometry_stability':stability,
         'mask_semantics':'overlap masks include own visibility and projected other visibility'}
    files={'image_A':images[0],'image_B':images[1],'mask_A_overlap':visible_a*255,
           'mask_B_overlap':visible_b*255,'mask_A_geometry':ma*255,'mask_B_geometry':mb*255,
           'mask_A_visible':va*255,'mask_B_visible':vb*255}
    for key,image in files.items():
        folder='images' if key.startswith('image') else 'masks'
        path=f'{folder}/{pair_id}_{key}.png'
        if not cv2.imwrite(str(root/path),image): raise OSError(path)
        row[key]=path
    row['metadata']=f'metadata/{pair_id}.json'
    (root/row['metadata']).write_text(json.dumps(row)+'\n')
    return row
