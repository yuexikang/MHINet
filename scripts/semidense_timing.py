"""Evaluation-only CUDA-event spans; original model methods are restored after each pair."""
from contextlib import ExitStack
import time
import torch
import mhinet.downstream.semidense as matching

LABELS = dict(dino='DINO 特征提取',mvt='MVT',h0='H0 预测与拟合',vgg='VGG',cgmdp='CGMDP 描述子解码',
    shared_other='共享网络其余开销',matcher_setup='匹配输入检查',coarse='D8 粗匹配及支持区',
    fine='D2 细匹配及选点',qrru='QRRU 投影、迭代及输出整理',pipeline_other='模块间开销')


def distribution(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),
        p50=float(np.percentile(a,50)),p90=float(np.percentile(a,90)),p95=float(np.percentile(a,95)),
        p99=float(np.percentile(a,99)),max=float(a.max()))


def timed_inference(system, images, sizes):
    """Batch1 only. Stage events include stream idle/CPU submission gaps, not kernel-only time.

    Events require only one final synchronization, not a barrier per stage. No GT is used.
    Coarse/fine/QRRU phase boundaries reuse existing function calls in the registered inference path.
    """
    if len(images)!=1:raise ValueError('Timing requires one image pair')
    cuda=images.is_cuda
    def stamp():
        if cuda:
            e=torch.cuda.Event(enable_timing=True);e.record(torch.cuda.current_stream(images.device));return e
        return time.perf_counter()
    def elapsed(a,b):return a.elapsed_time(b) if cuda else (b-a)*1000
    if cuda:torch.cuda.synchronize(images.device)
    spans={};phase=[None,None]
    def switch(name):
        now=stamp()
        if phase[0] is not None:spans.setdefault(phase[0],[]).append((phase[1],now))
        phase[:]=[name,now]
    def wrap(owner, attr, key, stack):
        original=getattr(owner,attr);owned=attr in owner.__dict__;saved=owner.__dict__.get(attr)
        def restore():
            if owned:setattr(owner,attr,saved)
            else:delattr(owner,attr)
        def call(*a,**kw):
            start=stamp()
            try:return original(*a,**kw)
            finally:spans.setdefault(key,[]).append((start,stamp()))
        setattr(owner,attr,call);stack.callback(restore)
    with ExitStack() as stack:
        shared=system.shared
        for obj,method,key in [(shared,'_extract_batched_descriptors','dino'),(shared,'_contextualize','mvt'),
            (shared.stage1_head,'forward','h0'),(shared.vgg,'forward','vgg'),(shared,'_decode_pyramid','cgmdp')]:
            wrap(obj,method,key,stack)
        original_masks=matching.predicted_overlap_masks
        def masks(*a,**kw):
            resolution=a[1] if len(a)>1 else kw.get('resolution')
            if resolution==98:switch('coarse')
            elif resolution==392:switch('fine')
            else:raise ValueError(f'Unregistered timing mask resolution {resolution}')
            return original_masks(*a,**kw)
        matching.predicted_overlap_masks=masks;stack.callback(setattr,matching,'predicted_overlap_masks',original_masks)
        project=system.matcher.qrru.project;original_project=project.forward
        owned='forward' in project.__dict__;saved=project.__dict__.get('forward')
        def project_call(*a,**kw):switch('qrru');return original_project(*a,**kw)
        project.forward=project_call
        def restore_project():
            if owned:project.forward=saved
            else:delattr(project,'forward')
        stack.callback(restore_project)
        wall_start=time.perf_counter();total_start=stamp();shared_start=stamp()
        features=system(images);shared_end=stamp();switch('matcher_setup')
        result=system.matcher.infer(features,sizes);switch(None);total_end=stamp()
        if cuda:torch.cuda.synchronize(images.device)
        wall_ms=(time.perf_counter()-wall_start)*1000
    measured={k:sum(elapsed(a,b) for a,b in v) for k,v in spans.items()}
    shared_ms=elapsed(shared_start,shared_end);stream_ms=elapsed(total_start,total_end)
    measured['shared_other']=shared_ms-sum(measured.get(k,0) for k in ('dino','mvt','h0','vgg','cgmdp'))
    measured['pipeline_other']=stream_ms-shared_ms-sum(measured.get(k,0) for k in ('matcher_setup','coarse','fine','qrru'))
    breakdown={k:measured.get(k,0.) for k in LABELS}
    if any(v < -0.1 for v in breakdown.values()):raise RuntimeError(f'Negative stage duration: {breakdown}')
    timing=dict(module_ms=breakdown,stream_total_ms=stream_ms,wall_total_ms=wall_ms,
        method='cuda_events_single_final_sync' if cuda else 'cpu_perf_counter',
        accounting_error_ms=stream_ms-sum(breakdown.values()))
    return result,timing
