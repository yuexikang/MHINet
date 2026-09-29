"""E checkpoint evaluation on JL1Flight affine test pairs with native-pixel GT."""
import argparse,json,sys,subprocess,html
from pathlib import Path
from dataclasses import replace,asdict
from mrsi_gt_metrics import score_matches,aggregate,project,conditional_rmse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-size',type=int,choices=(512,784),default=784)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    sys.path.insert(0,str(args.implementation_root.resolve()))
    import numpy as np
    import torch
    from PIL import Image,ImageDraw
    from mhinet.config import RuntimePaths,sha256_file
    from mhinet.pretraining.model import build_shared_network
    from mhinet.downstream.semidense import SemidenseConfig,SemidenseMatcher
    from mhinet.downstream.training import SemidenseSystem
    from mhinet.dataio.data import load_rgb_bicubic
    from scripts.semidense_timing import timed_inference
    torch.set_num_threads(2);torch.manual_seed(0)
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate pair manifest')
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True);meta=state['metadata']
    if meta['task']!='semidense_qrru_v1' or state['progress']['optimizer_step']!=meta['total_steps']:raise ValueError('Expected completed semidense model')
    runtime=replace(RuntimePaths.from_json(meta['config']['runtime']),device='cuda:0')
    if sha256_file(runtime.dino_checkpoint)!=meta['dino_sha256']:raise ValueError('DINO identity mismatch')
    config=replace(SemidenseConfig(**meta['matcher']),inference_window_chunk=1024,input_size=args.input_size)
    shared,_=build_shared_network(runtime,lora=False)
    model=SemidenseSystem(shared,SemidenseMatcher(config)).cuda();model.load_state_dict(state['model'],strict=True);model.eval();del state
    args.output.mkdir(parents=True);(args.output/'matches').mkdir();(args.output/'visuals').mkdir()
    registration=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=sha256_file(args.checkpoint),
        manifest=str(args.manifest.resolve()),manifest_sha256=sha256_file(args.manifest),matcher=asdict(config),
        implementation_commit=subprocess.check_output(['git','-C',str(args.implementation_root),'rev-parse','HEAD'],text=True).strip(),
        evaluator_sha256=sha256_file(Path(__file__)),metrics_sha256=sha256_file(Path(__file__).with_name('mrsi_gt_metrics.py')),
        gt_basis='Native a-to-b homography from b_*_H_0to1.txt; all 1755 JL1Flight test pairs; no fine-tuning.',
        precision='Correct matches within common source support at <=1/3/5 native target pixels divided by all output matches; empty output=0; pair-macro headline, micro additionally provided.',
        matching_rmse='sqrt(mean(||H_GT(a)-b||^2)) over ALL finite, GT-projectable returned matches, not just RANSAC inliers or correct subset; pair macro. Missing pairs excluded with counts.',
        sr='Fit H only from predicted matches with OpenCV RANSAC(3px,10000 iterations,confidence .999,seed0); evaluate against GT on a 20x20 source grid restricted to GT-visible overlap; SR@t = fraction of ALL pairs with grid RMSE<=t. Fit failures count as failure.',
        aggregation='Overall weighted by image pairs, not equal modality weights; modality rows pair-macro.',
        rmse_protocol='SR@3px success; GT error <=5px correct matches; failure=10; pair macro',
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__)
    (args.output/'registration.json').write_text(json.dumps(registration,ensure_ascii=False,indent=2)+'\n')
    records=[]
    with torch.no_grad(),(args.output/'pairs.jsonl').open('w') as stream:
        for index,row in enumerate(rows):
            source=[Image.open(row['image'+str(i)]).convert('RGB') for i in (0,1)]
            sizes=[im.size for im in source]
            images=torch.stack([load_rgb_bicubic(row['image'+str(i)],size=config.input_size) for i in (0,1)])[None].cuda()
            if index==0:model.matcher.infer(model(images),[sizes]);torch.cuda.synchronize()
            results,timing=timed_inference(model,images,[sizes]);result=results[0]
            a,b=[result[k].cpu().numpy() for k in ('points_a','points_b')]
            H_gt=np.loadtxt(row['gt'])
            if row['gt_direction']!='0to1':raise ValueError('Unexpected GT direction')
            metrics=score_matches(a,b,*sizes,H_gt)
            metrics.update(conditional_rmse(a,b,*sizes,H_gt,success_override=metrics['sr']['3']))
            stages={}
            for stage,ka,kb in [('coarse','coarse_a','coarse_b'),('fine','fine_all_a','fine_all_b')]:
                stages[stage]=score_matches(result[ka].cpu().numpy() if ka in result else [],result[kb].cpu().numpy() if kb in result else [],*sizes,H_gt)
            record=dict(id=row['id'],modality=row['modality'],sizes=sizes,gt_sha256=sha256_file(Path(row['gt'])),H_gt=H_gt.tolist(),
                image_sha256=[sha256_file(Path(row['image'+str(i)])) for i in (0,1)],
                failure_reason=result['failure_reason'],metrics=metrics,stages=stages,timing=timing)
            name=row['id'].replace('/','_')
            np.savez_compressed(args.output/'matches'/f'{name}.npz',points_a=a,points_b=b,confidence=result['confidence'].cpu().numpy())
            canvas=Image.new('RGB',(sum(im.width for im in source),max(im.height for im in source)))
            canvas.paste(source[0],(0,0));canvas.paste(source[1],(source[0].width,0));draw=ImageDraw.Draw(canvas)
            for j in np.linspace(0,max(0,len(a)-1),min(100,len(a)),dtype=int):
                expected,_=project(a[j:j+1],H_gt)
                color=(50,210,80) if np.linalg.norm(expected[0]-b[j])<=5 else (235,75,65)
                draw.line((float(a[j,0]),float(a[j,1]),float(b[j,0])+source[0].width,float(b[j,1])),fill=color,width=1)
            canvas.thumbnail((1500,850));canvas.save(args.output/'visuals'/f'{name}.jpg')
            records.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
            print(json.dumps(dict(pair=index+1,id=row['id'],matches=len(a),precision=metrics['precision'],rmse=metrics['matching_rmse'])),flush=True)
    base_aggregate=aggregate
    def summarize(records):
        result=base_aggregate(records)
        result['rmse_correct5_or_failure10']=float(np.mean([r['metrics']['rmse_correct5_or_failure10'] for r in records]))
        return result
    groups={k:summarize([r for r in records if r['modality']==k]) for k in sorted({r['modality'] for r in records})}
    stage_reports={stage:dict(overall=aggregate([dict(r,metrics=r['stages'][stage]) for r in records]),by_modality={k:aggregate([dict(r,metrics=r['stages'][stage]) for r in records if r['modality']==k]) for k in groups}) for stage in ('coarse','fine')}
    report=dict(complete=True,registration=registration,overall=summarize(records),by_modality=groups,stages=stage_reports)
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    table=[]
    for label,g in [('总体',report['overall']),*groups.items()]:
        values=[str(g['pairs']),*[f'{100*g["precision"][t]:.3f}%' for t in ('1','3','5')],
            '—' if g['rmse_correct5_or_failure10'] is None else f'{g["rmse_correct5_or_failure10"]:.3f}',f'{100*g["sr"]["3"]:.2f}%',f'{g["mean_ncm"]["5"]:.1f}',f'{g["inference_ms_mean"]:.1f}']
        table.append('<tr><td>'+html.escape(label)+'</td>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    gallery=''.join(f'<figure><figcaption>{html.escape(r["id"])}</figcaption><img loading="lazy" width="750" src="visuals/{r["id"].replace("/","_")}.jpg"></figure>' for r in records)
    page=f'<!doctype html><meta charset="utf-8"><title>E组 JL1Flight {config.input_size}评估</title><style>body{{font-family:sans-serif;margin:28px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}img{{max-width:100%}}</style><h1>E组 JL1Flight {config.input_size}：总体与变换分组结果</h1><p>使用各目标影像的 H_0to1.txt 真值，方向 a→b。共1755对，按 t0–t4 变换编号分组。绿色线为≤5px，红色为&gt;5px。</p><p>RMSE：配准 SR@3px 成功对统计误差≤5px正确匹配的RMSE；失败对记10。SR@3为预测匹配拟合H后的公共网格RMSE≤3px比例，失败计入分母；总体按影像对宏平均。</p><table><tr><th>变换组</th><th>对数</th><th>P@1px</th><th>P@3px</th><th>P@5px</th><th>匹配RMSE(px)</th><th>SR@3px</th><th>NCM@5px</th><th>推理ms</th></tr>{"".join(table)}</table><p><a href="report.json">指标定义与完整统计</a> · <a href="pairs.jsonl">逐对结果</a></p>{gallery}'
    (args.output/'index.html').write_text(page)
    print(json.dumps(report['overall']),flush=True)


if __name__=='__main__':main()
