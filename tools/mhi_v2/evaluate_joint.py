"""Evaluate one saved joint checkpoint; retain before/after refinement metrics."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mhinet.v2.data import GeometryPairDataset
from mhinet.v2.pipeline import MHIV2
from mhinet.v2.pair_matching.joint_training import JointContext
from mhinet.v2.pair_matching.metrics import measure, totals

DATASETS = {
    'Hangzhou': (Path('/home/disk1/Data/datasets/kewei/HangZhou_cloud0_rgb_RGBMatch_v4'), 'manifests/test_real.jsonl'),
    'GoogleEarth': (Path('/home/disk1/Data/datasets/Test_Data/GoogleEarth/val'), 'pairs.jsonl'),
    'JL1Flight': (Path('/home/disk1/Data/datasets/Test_Data/JL1Flight/test'), 'pairs.jsonl'),
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    cv2.setNumThreads(1)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    scorer_path = Path('/home/disk1/LoMa/experiments/official_loma_rgb512/metrics.py')
    spec = importlib.util.spec_from_file_location('masked_metrics', scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    ctx = JointContext(state['config'], args.device)
    ctx.restore(state)
    model = MHIV2(ctx.shared, ctx.matcher, ctx.adapter, refiner=ctx.refiner).requires_grad_(False).eval()
    root, relative = DATASETS[args.dataset]
    manifest = root / relative
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    dataset = GeometryPairDataset(root, rows)
    report = dict(dataset=args.dataset, checkpoint=str(args.checkpoint), epoch=state['epoch'],
                  optimizer_step=state['step'], frozen_flow=state['config']['frozen_flow_checkpoint'],
                  manifest=str(manifest), pairs=len(rows), complete=False,
                  started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  batchnorm='all running statistics restored from the same saved rank-0 checkpoint',
                  inference='native outputs, max 2048, no mutual filtering, no GT used by model',
                  geometry_protocol='micro P@1/3/5/10 over all output points; EPE on known-visible GT queries',
                  masked_protocol='existing GE/JL scorer: per-pair macro precision with both overlap masks; RANSAC uses predictions only',
                  scorer=str(scorer_path), results={})
    write(out/'summary.json', report)
    del state, ctx
    geometry = {stage: [] for stage in ('pre_refine', 'post_refine')}
    masked = {stage: [] for stage in geometry}
    worse = visible = 0
    # Warm up without modifying BN buffers or learned weights.
    warm = dataset[0]['images'].to(args.device)
    for _ in range(2): model.match(warm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (out/'pairs.jsonl').open('w', buffering=1) as stream:
        for index, row in enumerate(rows):
            batch = dataset[index]
            images = batch['images'].to(args.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = model.match(images, diagnostics=True)
            torch.cuda.synchronize()
            network_ms = (time.perf_counter()-start)*1000
            a = prediction['points_a']
            before = prediction['diagnostics']['points_b_d2']
            after = prediction['points_b']
            if len(a)>2048 or not all(torch.isfinite(x).all() for x in (a,before,after)):
                raise ValueError(f'Invalid output: {row["pair_id"]}')
            gt = batch['geometry'].to(args.device)
            masks=[]
            for side in 'AB':
                path=row.get(f'valid_{side}',row.get(f'mask_{side}_overlap'))
                with Image.open(root/path) as image: masks.append(np.asarray(image.convert('L'))>0)
            record=dict(pair_id=row['pair_id'], group=row.get('G'), geometry={}, masked={})
            errors={}
            for stage,b in (('pre_refine',before),('post_refine',after)):
                counts,error,good=measure(a,b,gt)
                geometry[stage].append(counts)
                errors[stage]=error
                record['geometry'][stage]=counts
                an,bn=a.cpu().numpy(),b.cpu().numpy()
                start=time.perf_counter()
                fit=scorer.fit_ransac(an,bn)
                ransac_ms=(time.perf_counter()-start)*1000
                metrics=scorer.score_matches(an,bn,row['size_A'],row['size_B'],row['H_A_to_B'],mask_a=masks[0],mask_b=masks[1],fit=fit)
                masked[stage].append(dict(metrics=metrics,failure_reason='none',timing=dict(network_ms=network_ms,ransac_ms=ransac_ms,peak_allocated_bytes=torch.cuda.max_memory_allocated())))
                record['masked'][stage]=metrics
            worse+=int(((errors['post_refine']>errors['pre_refine'])&good).sum())
            visible+=int(good.sum())
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            if index==0 or (index+1)%50==0 or index+1==len(rows):
                progress=dict(dataset=args.dataset,done=index+1,total=len(rows),seconds=time.perf_counter()-started)
                write(out/'progress.json',progress)
                print(json.dumps(progress),flush=True)
    for stage in geometry:
        report['results'][stage]=dict(geometry=totals(geometry[stage]),masked=scorer.aggregate(masked[stage]))
    report.update(complete=True,seconds=time.perf_counter()-started,
                  worsened_visible_fraction=worse/max(1,visible),peak_gib=torch.cuda.max_memory_allocated()/2**30,
                  finished_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    write(out/'summary.json',report)
    print('FULL TEST COMPLETE',flush=True)
    print(json.dumps({stage:value['geometry'] for stage,value in report['results'].items()}),flush=True)


if __name__=='__main__':
    main()
