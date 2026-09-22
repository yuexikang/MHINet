"""Bounded, reproducible scientific diagnostics for the semidense cascade."""
from pathlib import Path
import html
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from mhinet.engine.checkpointing import capture_rng_state,restore_rng_state
from .supervision import to_uv,sample_uv,fine_labels,legal_h
from .loma_reference.matching_utils import project_normalized,flat_indices_to_centers
from .metrics import dense_metrics


def array(x):return x.detach().float().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)


def savefig(fig,path):
    fig.savefig(path,dpi=110,bbox_inches='tight');plt.close(fig)


def panel_index(folder,title):
    blocks=[]
    for p in sorted(list(folder.glob('*.png'))+list(folder.glob('*.gif'))):
        blocks.append(f'<details open><summary>{html.escape(p.stem)}</summary><a href="{p.name}"><img loading="lazy" src="{p.name}"></a></details>')
    (folder/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1600px;margin:auto}img{max-width:100%}summary{padding:8px}</style>'
        +f'<h1>{html.escape(title)}</h1><p><a href="../../index.html">Training overview</a> · <a href="trace.json">Trace and units</a></p>'+'\n'.join(blocks))


def feature_gallery(pair,folder,calibration,name,all_channels=False):
    values=array(pair); _,c,h,w=values.shape
    flat=values.transpose(0,2,3,1).reshape(-1,c)
    calibration.mkdir(parents=True,exist_ok=True)
    path=calibration/(name+'.npz')
    if path.exists():
        state=dict(np.load(path))
    else:
        sampled=flat[::max(1,len(flat)//4096)].astype(np.float64)
        mean=sampled.mean(0);cov=(sampled-mean).T@(sampled-mean)/max(1,len(sampled)-1)
        _,vec=np.linalg.eigh(cov);basis=vec[:,-3:]
        for j in range(3):
            if basis[np.argmax(np.abs(basis[:,j])),j]<0:basis[:,j]*=-1
        rgb=(sampled-mean)@basis
        state=dict(mean=mean.astype('float32'),basis=basis.astype('float32'),low=np.percentile(rgb,1,axis=0),
            high=np.percentile(rgb,99,axis=0),channel_max=np.percentile(np.abs(sampled),99),
            norm_max=np.percentile(np.linalg.norm(sampled,axis=-1),99),std_max=np.percentile(sampled.std(-1),99))
        np.savez_compressed(path,**state)
    rgb=(flat-state['mean'])@state['basis']
    rgb=np.clip((rgb-state['low'])/np.maximum(state['high']-state['low'],1e-8),0,1).reshape(2,h,w,3)
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for side in range(2):
        axes[side,0].imshow(rgb[side]);axes[side,0].set_title(f'{name} {"AB"[side]} fixed-basis PCA')
        for ax,value,title,vmax in ((axes[side,1],np.linalg.norm(values[side],axis=0),'raw norm',state['norm_max']),
                                    (axes[side,2],values[side].std(0),'channel std',state['std_max'])):
            im=ax.imshow(value,vmin=0,vmax=max(float(vmax),1e-6));fig.colorbar(im,ax=ax,fraction=.04);ax.set_title(title)
        for ax in axes[side]:ax.axis('off')
    savefig(fig,folder/f'features_{name}_summary.png')
    ids=np.arange(c) if all_channels else np.unique(np.linspace(0,c-1,min(32,c),dtype=int))
    # Two image-side atlases, fixed channel indices and fixed signed scale.
    for side in range(2):
        atlas=[]
        for ch in ids:
            im=Image.fromarray(values[side,ch]).resize((48,48),Image.Resampling.BILINEAR)
            atlas.append(np.asarray(im))
        cols=16 if len(ids)>48 else 8;rows=int(np.ceil(len(ids)/cols))
        sheet=np.full((rows*52,cols*52),np.nan,dtype='float32')
        for k,v in enumerate(atlas):sheet[(k//cols)*52:(k//cols)*52+48,(k%cols)*52:(k%cols)*52+48]=v
        fig,ax=plt.subplots(figsize=(cols,rows+1))
        vmax=max(float(state['channel_max']),1e-6)
        ax.imshow(sheet,cmap='coolwarm',vmin=-vmax,vmax=vmax)
        for k,ch in enumerate(ids):ax.text((k%cols)*52,(k//cols)*52+6,str(ch),fontsize=5,color='black',bbox=dict(facecolor='white',alpha=.7,pad=0))
        ax.set_title(f'{name} {"AB"[side]} channels; fixed scale ±{vmax:.3g}');ax.axis('off')
        savefig(fig,folder/f'features_{name}_{"AB"[side]}_channels.png')
    return dict(shape=list(values.shape),nonfinite=int((~np.isfinite(values)).sum()),
                clipped_fraction=float((np.abs(values)>state['channel_max']).mean()),calibration=str(path))


@torch.no_grad()
def query_maps(d8,diag,tau,chunk=128):
    """Exact dual-softmax rows with full support column denominators, not top-k softmax."""
    device=d8.device
    ia=diag['mask_a'].flatten().nonzero().flatten().to(device)
    ib=diag['mask_b'].flatten().nonzero().flatten().to(device)
    # Fixed spatial probes across every checkpoint, even when H0 excludes them.
    positions=torch.tensor([12,36,60,84],device=device)
    yy,xx=torch.meshgrid(positions,positions,indexing='ij');queries=(yy*98+xx).flatten()
    a=F.normalize(d8[0].flatten(1).T.float(),dim=-1);b=F.normalize(d8[1].flatten(1).T.float(),dim=-1)
    cosine=a[queries]@b.T
    probability=torch.zeros_like(cosine)
    reverse=torch.full((len(ib),),-1,dtype=torch.long,device=device)
    if len(ia) and len(ib):
        lse=torch.full((len(ib),),-torch.inf,device=device)
        maximum=torch.full_like(lse,-torch.inf)
        for start in range(0,len(ia),chunk):
            logits=a[ia[start:start+chunk]]@b[ib].T/tau
            lse=torch.logaddexp(lse,torch.logsumexp(logits,0))
            value,index=logits.max(0);better=value>maximum
            reverse[better]=ia[start:start+chunk][index[better]];maximum=torch.maximum(maximum,value)
        logits=cosine[:,ib]/tau
        logp=2*logits-torch.logsumexp(logits,-1,keepdim=True)-lse
        probability[:,ib]=logp.exp()
        probability[~diag['mask_a'].flatten()[queries.cpu()].to(device)]=0
    best=torch.full_like(queries,-1);mutual=torch.zeros_like(queries,dtype=torch.bool)
    if len(ib):
        local=cosine[:,ib].argmax(-1);best=ib[local];mutual=reverse[local]==queries
    return queries.cpu(),cosine.cpu(),probability.cpu(),best.cpu(),mutual.cpu()


def matches_plot(images,result,folder):
    rgb=np.concatenate(images,axis=1)
    fig,axes=plt.subplots(3,1,figsize=(14,18))
    for ax,title,ak,bk in zip(axes,('D8 coarse','D2 fine','QRRU final'),
            ('coarse_a','fine_all_a','points_a'),('coarse_b','fine_all_b','points_b')):
        ax.imshow(rgb)
        a=array(result.get(ak,torch.empty(0,2)));b=array(result.get(bk,torch.empty(0,2)))
        # Dataset native sizes can differ; these overlay coordinates are normalized separately upstream.
        if len(a):
            take=np.linspace(0,len(a)-1,min(100,len(a)),dtype=int)
            for u,v in zip(a[take],b[take]):ax.plot([u[0],v[0]+784],[u[1],v[1]],color='cyan',alpha=.4,lw=.5)
            ax.scatter(a[:,0],a[:,1],s=.3,c='yellow');ax.scatter(b[:,0]+784,b[:,1],s=.3,c='cyan')
        ax.set_title(f'{title}: {len(a)} matches, up to 100 evenly indexed lines');ax.axis('off')
    savefig(fig,folder/'stages_matches.png')


@torch.no_grad()
def render_pair(system,shared,images,batch,result,folder,calibration,all_channels=False):
    folder.mkdir(parents=True,exist_ok=True)
    rgb=array(images[0].permute(0,2,3,1)).clip(0,1)
    H=batch['H_gt_norm'][0].to(images.device);H0=shared['H0_norm'][0]
    trace=dict(pair_id=batch['pair_id'][0],coordinate_display='784 input pixels unless labeled D2',
        actual_matching='predicted cascade; GT only overlaid after prediction',
        frozen_h0_prior='coarse support and fine lattice only; QRRU uses regular grids',
        representative_fine_qrru='dynamic selected examples, not fixed query identities',features={})
    for scale in (8,4,2):
        trace['features'][f'D{scale}']=feature_gallery(shared['pyramid'][scale][0],folder,calibration,f'D{scale}',all_channels)
    projected=system.matcher.qrru.project(shared['pyramid'][2][0].float())
    trace['features']['Q48']=feature_gallery(projected,folder,calibration,'Q48',all_channels)
    # Matcher output is native; convert each side to resized input coordinate for overlays.
    plotted=dict(result)
    for key,side in (('coarse_a','A'),('fine_all_a','A'),('points_a','A'),('coarse_b','B'),('fine_all_b','B'),('points_b','B')):
        if key in result:
            size=batch['size_'+side][0].to(result[key])
            plotted[key]=(result[key]+.5)*784/size-.5
    matches_plot(rgb,plotted,folder)
    if 'diagnostics' not in result:
        trace['failure']=result['failure_reason'];(folder/'trace.json').write_text(json.dumps(trace,indent=2));panel_index(folder,trace['pair_id']);return
    diag=result['diagnostics'];trace['counts']=dict(diag['coarse_counts'],fine_before_cap=diag['fine_before_cap'],fine_after_cap=diag['fine_after_cap'],qrru_final=len(result['points_a']))
    fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,image,mask,name in zip(axes,rgb,(diag['mask_a'],diag['mask_b']),('A','B')):
        ax.imshow(image);ax.imshow(array(mask),extent=(-.5,783.5,783.5,-.5),alpha=.25,cmap='Greens',vmin=0,vmax=1)
        ax.set_title(f'{name}: predicted H0 support');ax.axis('off')
    savefig(fig,folder/'h0_support.png')
    queries,cosine,prob,best,mutual=query_maps(shared['pyramid'][8][0],diag,float(system.matcher.tau_c))
    query_norm=flat_indices_to_centers(queries.to(H.device),98,98)
    gt,_=project_normalized(query_norm,H);gtuv=array(to_uv(gt,(98,98)))
    selected={int(a):int(b) for a,b in zip(diag['coarse_source'],diag['coarse_target'])}
    qrecords=[]
    for k,q in enumerate(queries):
        q=int(q);top=torch.topk(prob[k],5).indices
        qrecords.append(dict(source_flat=q,in_support=bool(diag['mask_a'].flatten()[q]),
            selected_target=selected.get(q),status='selected' if q in selected else 'excluded_support' if not diag['mask_a'].flatten()[q] else 'not_mutual' if not mutual[k] else 'below_threshold' if best[k]>=0 and prob[k,best[k]]<system.matcher.config.confidence_threshold else 'capped',
            top5_flat=top.tolist(),top5_probability=prob[k,top].tolist()))
    trace['fixed_queries']=qrecords
    np.savez_compressed(folder/'selected_lineage.npz',coarse_source=array(diag['coarse_source']),coarse_target=array(diag['coarse_target']),
        coarse_confidence=array(diag['coarse_confidence']),fine_lineage=array(diag['lineage']),qrru_valid=array(diag['qrru_valid']))
    for values,name,cmap,low,high in ((cosine,'cosine','coolwarm',-1,1),(prob,'dual_probability','magma',0,1)):
        fig,axes=plt.subplots(4,4,figsize=(14,14))
        for k,ax in enumerate(axes.flat):
            im=ax.imshow(array(values[k].reshape(98,98)),cmap=cmap,**(dict(norm=LogNorm(vmin=1e-6,vmax=1)) if name=='dual_probability' else dict(vmin=low,vmax=high)))
            ax.scatter(*gtuv[k],marker='+',c='lime',s=50)
            q=int(queries[k]);target=selected.get(q)
            if target is not None:ax.scatter(target%98,target//98,marker='x',c='cyan',s=35)
            ax.set_title(f'A D8 {q}: {qrecords[k]["status"]}'+(' [log]' if name=='dual_probability' else ''),fontsize=7);ax.axis('off')
        fig.colorbar(im,ax=list(axes.flat),fraction=.02);savefig(fig,folder/f'coarse_{name}.png')
    np.savez_compressed(folder/'coarse_queries.npz',query_ids=array(queries),cosine=array(cosine),dual_probability=array(prob))
    fine_records=[]
    d2=shared['pyramid'][2][0].float()
    for index,ft in enumerate(diag['fine']):
        ga=ft['a'].to(H.device);gb=ft['b'].to(H.device)
        fa=F.normalize(sample_uv(d2[0],to_uv(ga,(392,392))),dim=-1)
        fb=F.normalize(sample_uv(d2[1],to_uv(gb,(392,392))),dim=-1)
        similarity=array(fa@fb.T);probability=ft['logp'].exp().numpy()
        m,si,ti,_,_=fine_labels(ga[None],gb[None],H,batch['mask_A_overlap'][0].to(H.device),batch['mask_B_overlap'][0].to(H.device),(392,392))
        pred=probability.argmax(-1);reverse=probability.argmax(-2)
        valid=array(ft['valid_a']).astype(bool)&array(ft['valid_b']).astype(bool)[pred]&(reverse[pred]==np.arange(16))
        gtp,_=project_normalized(ga,H);prior,_=project_normalized(ga,H0)
        points=[array(to_uv(x,(784,784))) for x in (ga,gb,gtp,prior)]
        fig,axes=plt.subplots(2,3,figsize=(14,9))
        for ax,mat,title,vmin,vmax in ((axes[0,0],similarity,'cosine',-1,1),(axes[0,1],probability,'dual probability',0,1)):
            im=ax.imshow(mat,cmap='magma',**(dict(norm=LogNorm(vmin=1e-6,vmax=1)) if title=='dual probability' else dict(vmin=vmin,vmax=vmax))); ax.scatter(array(ti),array(si),marker='+',c='lime');ax.scatter(pred[valid],np.arange(16)[valid],marker='x',c='cyan');ax.set_title(title+(' [log color]' if title=='dual probability' else ''));fig.colorbar(im,ax=ax,fraction=.04)
        axes[0,2].imshow(rgb[0]);axes[0,2].scatter(points[0][:,0],points[0][:,1],c='yellow',s=12)
        for k,p in enumerate(points[0]):axes[0,2].text(*p,str(k),fontsize=7)
        for ax,image,pts in ((axes[0,2],rgb[0],points[0]),(axes[1,0],rgb[1],points[1])):
            ax.imshow(image);center=pts.mean(0);ax.set_xlim(center[0]-24,center[0]+24);ax.set_ylim(center[1]+24,center[1]-24)
        axes[0,2].scatter(points[0][:,0],points[0][:,1],c='yellow',s=10);axes[0,2].set_title('A: 4x4 D2 children')
        for pts,color,label in ((points[3],'gray','H0 only'),(points[1],'orange','H0 + coarse residual'),(points[2],'lime','GT')):
            axes[1,0].scatter(pts[:,0],pts[:,1],c=color,s=12,label=label)
        axes[1,0].legend(fontsize=6);axes[1,0].set_title('B actual warped lattice')
        query=int(np.where(valid)[0][0]) if valid.any() else 0
        axes[1,1].imshow(probability[query].reshape(4,4),norm=LogNorm(vmin=1e-6,vmax=1),cmap='magma');axes[1,1].set_title(f'A token {query} → B probabilities [log]')
        axes[1,2].imshow(rgb[1]);sc=axes[1,2].scatter(points[1][:,0],points[1][:,1],c=probability[query],cmap='magma',norm=LogNorm(vmin=1e-6,vmax=1),s=60)
        center=points[1].mean(0);axes[1,2].set_xlim(center[0]-24,center[0]+24);axes[1,2].set_ylim(center[1]+24,center[1]-24);axes[1,2].set_title('Same probabilities in image coordinates')
        savefig(fig,folder/f'fine_{index:02d}.png')
        np.savez_compressed(folder/f'fine_{index:02d}.npz',a=array(ga),b=array(gb),similarity=similarity,probability=probability,gt_source=array(si),gt_target=array(ti))
        fine_records.append(dict(parent=ft['parent'],source_d8=int(diag['coarse_source'][ft['parent']]),gt_pairs=len(si),selected_tokens=np.where(valid)[0].tolist()))
    trace['fine']=fine_records
    qrecords=[]
    for qt in diag['qrru']:
        qa=qt['a'].to(H.device);truth,_=project_normalized(qa,H);truth=array(to_uv(truth,(392,392)))
        for k in range(len(qa)):
            centers=np.concatenate((array(to_uv(qt['b'][k:k+1],(392,392)))[None],array(qt['centers'][:,k:k+1])),axis=0)[:,0]
            fig,axes=plt.subplots(2,len(centers),figsize=(3*len(centers),6))
            for t in range(len(centers)):
                ax=axes[0,t];ax.imshow(rgb[1]);center=centers[0]*2+.5
                ax.set_xlim(center[0]-16,center[0]+16);ax.set_ylim(center[1]+16,center[1]-16)
                ax.scatter(*(truth[k]*2+.5),marker='+',c='lime',s=70);ax.scatter(*(centers[t]*2+.5),c='magenta',s=15)
                error=np.linalg.norm(centers[t]-truth[k]);ax.set_title(f't={t}, EPE={error:.3f} D2 px',fontsize=8)
                if t:
                    grid=array(qt['grids'][t-1,k])*2+.5
                    for line in grid:ax.plot(line[:,0],line[:,1],c='orange',lw=.6)
                    for line in grid.transpose(1,0,2):ax.plot(line[:,0],line[:,1],c='orange',lw=.6)
                    flow=array(qt['flows'][t-1,k]);gate=array(qt['gates'][t-1,k,0])
                    axes[1,t].imshow(gate,vmin=0,vmax=1,cmap='viridis')
                    yy,xx=np.meshgrid(np.arange(2),np.arange(2),indexing='ij')
                    axes[1,t].quiver(xx,yy,flow[0],-flow[1],color='red',angles='xy',scale_units='xy',scale=1)
                    axes[1,t].set_title('update gate [0,1] + control flow',fontsize=7)
                else:
                    from .supervision import offsets
                    grid=(array(to_uv(qt['b'][k],(392,392)))+array(offsets('cpu')))*2+.5
                    for line in grid:ax.plot(line[:,0],line[:,1],c='orange',lw=.6)
                    for line in grid.transpose(1,0,2):ax.plot(line[:,0],line[:,1],c='orange',lw=.6)
                    axes[1,t].text(.1,.5,'Regular initial window\nNo H0 input');axes[1,t].axis('off')
            if k==0:
                fig.canvas.draw();canvas=np.asarray(fig.canvas.buffer_rgba());frames=[]
                for ax in axes[0]:
                    bb=ax.get_window_extent();x0,y0,x1,y1=map(int,(bb.x0,bb.y0,bb.x1,bb.y1))
                    crop=canvas[canvas.shape[0]-y1:canvas.shape[0]-y0,x0:x1,:3]
                    frames.append(Image.fromarray(crop).resize((320,320)))
                frames[0].save(folder/'qrru_00.gif',save_all=True,append_images=frames[1:],duration=700,loop=0)
            savefig(fig,folder/f'qrru_{k:02d}.png')
            qrecords.append(dict(lineage=qt['lineage'][k].tolist(),epe_d2=np.linalg.norm(centers-truth[k],axis=-1).tolist(),centers_d2=centers.tolist()))
        np.savez_compressed(folder/'qrru_trace.npz',**{key:array(value) for key,value in qt.items()})
    trace['qrru']=qrecords
    (folder/'trace.json').write_text(json.dumps(trace,indent=2));panel_index(folder,trace['pair_id'])


def update_overview(output):
    root=Path(output);visual=root/'visualizations';visual.mkdir(exist_ok=True)
    train=[]
    path=root/'train.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:train.append(json.loads(line))
            except json.JSONDecodeError:continue
    if train:
        steps=[r['step'] for r in train]
        fig,axes=plt.subplots(2,3,figsize=(16,9))
        def plot(ax,values,label):
            ax.plot(steps,values,alpha=.25,lw=.6)
            window=min(20,len(values));smooth=np.convolve(values,np.ones(window)/window,'valid')
            ax.plot(steps[window-1:],smooth,label=label,lw=1.2)
        plot(axes[0,0],[np.mean([x['lc']+x['lf']+x['lq'] for x in r['records']]) for r in train],'sum (weights=1)')
        for key in ('lc','lf','lq'):
            plot(axes[0,0],[np.mean([x[key] for x in r['records']]) for r in train],key)
        for key in ('q_control','q_center','fine_coverage'):
            plot(axes[0,1],[np.mean([x[key] for x in r['records']]) for r in train],key)
        for key in train[0]['gradient_groups']:
            plot(axes[0,2],[r['gradient_groups'][key] for r in train],key)
        for key in ('lr_shared','lr_head'):
            if key in train[-1]:plot(axes[1,0],[r.get(key,0) for r in train],key)
        for key in ('tau_c','tau_f'):
            if key in train[-1]:plot(axes[1,1],[r.get(key,0) for r in train],key)
        plot(axes[1,2],[r['seconds'] for r in train],'seconds / optimizer step')
        for ax,title in zip(axes.flat,('GT-decoupled losses','Supervision / QRRU losses','Gradient norms','Learning rates','Temperatures','Training timing')):
            ax.set_title(title);ax.set_xlabel('optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=7)
        savefig(fig,visual/'training_curves.png')
    summaries=[]
    for p in sorted(visual.glob('step_*/summary.json')):
        summaries.append(json.loads(p.read_text()))
    if summaries:
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for stage in ('coarse','fine','final'):
            for ax,key in ((axes[0],'precision_1px'),(axes[1],'epe'),(axes[2],'matches')):
                ax.plot([r['step'] for r in summaries],[r['stages'].get(stage,{}).get(key,np.nan) for r in summaries],marker='.',label=stage)
        for ax,title in zip(axes,('Fixed-val cascade precision@1 (native px)','Overlap EPE (native target px)','Output count')):
            ax.set_title(title);ax.set_xlabel('optimizer step');ax.legend();ax.grid(alpha=.2)
        savefig(fig,visual/'cascade_curves.png')
    links=[]
    for folder in sorted(visual.glob('step_*'),reverse=True):
        pairs=' '.join(f'<a href="{folder.name}/{p.name}/index.html">{p.name}</a>' for p in sorted(folder.glob('pair_*')))
        links.append(f'<details open><summary>{folder.name}</summary>{pairs}</details>')
    (visual/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="120"><style>body{font:16px sans-serif;margin:30px}img{max-width:100%}a{padding:8px;display:inline-block}summary{font-weight:bold;padding:10px}</style>'
        '<h1>Semidense training diagnostics</h1><p>Fixed validation sample cohort. GT is used for evaluation overlays only. Auto-refresh: 120s.</p>'
        +''.join(f'<img src="{p.name}?v={int(time.time())}">' for p in visual.glob('*curves.png'))+'\n'.join(links))


@torch.no_grad()
def snapshot(system,dataset,device,output,step,*,pairs=12,all_channels=False):
    """Separate validation pass with RNG and module mode restoration."""
    from torch.utils.data import default_collate
    from mhinet.config import sha256_file
    state=capture_rng_state();was_training=system.training
    system.eval();started=time.perf_counter();root=Path(output);visual=root/'visualizations'
    folder=visual/f'step_{step:07d}';folder.mkdir(parents=True,exist_ok=True)
    manifest_path=visual/'manifest.json'
    ids=np.unique(np.linspace(0,len(dataset)-1,min(pairs,len(dataset)),dtype=int)).tolist()
    manifest=dict(indices=ids,pair_ids=[dataset.index[i].pair_id for i in ids],selection='evenly spaced fixed val-manifest indices, independent of predictions',
        query_d8_rows_cols=[12,36,60,84],coordinate_convention='align_corners=False',pca='fixed step0 joint A/B basis per pair and scale')
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise ValueError('Visualization cohort changed')
    manifest_path.write_text(json.dumps(manifest,indent=2));records=[]
    try:
        for n,index in enumerate(ids):
            batch=default_collate([dataset[index]])
            images=batch['images'].to(device);shared=system(images)
            sizes=[(tuple(batch['size_A'][0].tolist()),tuple(batch['size_B'][0].tolist()))]
            result=system.matcher.infer(shared,sizes,diagnostics=True)[0]
            pair_folder=folder/f'pair_{n:02d}'
            render_pair(system,shared,images,batch,result,pair_folder,visual/'calibration'/f'pair_{n:02d}',all_channels)
            row=dataset._read_record(dataset.index[index].offset)
            record=dict(pair_id=row['pair_id'],failure_reason=result['failure_reason'],
                rejected_qrru=result.get('rejected_qrru',0),qrru_outside_samples=result.get('qrru_outside_samples',0))
            for stage,ak,bk in (('coarse','coarse_a','coarse_b'),('fine','fine_all_a','fine_all_b'),('final','points_a','points_b')):
                a=array(result.get(ak,torch.empty(0,2)));b=array(result.get(bk,torch.empty(0,2)))
                record[stage]=dense_metrics(row,dataset.split_root,a,b)
            if 'fine_points_b' in result and len(result['points_a']):
                a=array(result['points_a']);fine=array(result['fine_points_b']);final=array(result['points_b'])
                gt=np.c_[a,np.ones(len(a))]@np.asarray(row['H_A_to_B']).T;gt=gt[:,:2]/gt[:,2,None]
                ef=np.linalg.norm(fine-gt,axis=-1);eq=np.linalg.norm(final-gt,axis=-1)
                record['same_retained_points_qrru']=dict(count=len(a),improved_fraction=float((eq<ef).mean()),
                    worsened_fraction=float((eq>ef).mean()),before_epe=float(ef.mean()),after_epe=float(eq.mean()))
            records.append(record);(folder/'pairs.json').write_text(json.dumps(records,indent=2))
            update_overview(root)
            print(f'VIS step={step} pair={n+1}/{len(ids)} saved={pair_folder}',flush=True)
            del shared,result
        stages={}
        for stage in ('coarse','fine','final'):
            epe=[r[stage]['overlap_finite_EPE'] for r in records if r[stage]['overlap_finite_EPE'] is not None]
            stages[stage]=dict(precision_1px=float(np.mean([r[stage]['thresholds']['1.0']['precision'] for r in records])),
                epe=float(np.mean(epe)) if epe else None,matches=float(np.mean([r[stage]['matches'] for r in records])))
        checkpoint=root/'latest.pt'
        summary=dict(step=step,pairs=len(records),stages=stages,seconds=time.perf_counter()-started,
            checkpoint_sha256=sha256_file(checkpoint) if checkpoint.exists() else None,
            source='current model at optimizer boundary; labels excluded from cascade',
            all_channels=all_channels,scope='fixed diagnostic subset, not full-val score')
        (folder/'summary.json').write_text(json.dumps(summary,indent=2));update_overview(root)
        return summary
    finally:
        restore_rng_state(state);system.train(was_training)
