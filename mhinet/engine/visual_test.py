"""Original GoogleEarth pairs: qualitative inference only, never accuracy metrics."""
import argparse
import csv
import json
from pathlib import Path
import torch
from tqdm import tqdm
from mhinet.config import RuntimePaths, sha256_file
from mhinet.models.model import build_model
from mhinet.engine.checkpointing import load_checkpoint
from mhinet.dataio.data import load_rgb_bicubic
from mhinet.visualization.visualization import write_iteration_overlays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--runtime', type=Path, default=Path('configs/runtime_paths.server.json'))
    p.add_argument('--data-root', type=Path, default=Path('/home/disk1/Data/datasets/GoogleEarth/evaluation_data'))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-pairs', type=int, default=None)
    args = p.parse_args()
    if args.max_pairs is not None and args.max_pairs < 1:
        p.error('--max-pairs must be positive')
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with (args.data_root / 'test_pairs.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if args.max_pairs is not None:
        rows = rows[:args.max_pairs]
    for row in rows:
        for key in ('Source', 'Target'):
            if not (args.data_root / row[key]).is_file():
                raise FileNotFoundError(row[key])
    runtime = RuntimePaths.from_json(args.runtime)
    model, _ = build_model(runtime)
    load_checkpoint(args.checkpoint, model=model, map_location=runtime.device, restore_rng=False)
    model.eval()
    args.output_dir.mkdir(parents=True)
    for index, row in enumerate(tqdm(rows, desc='Visual test (NO GT)', unit='pair')):
        a, b = args.data_root / row['Target'], args.data_root / row['Source']
        images = torch.stack([load_rgb_bicubic(a), load_rgb_bicubic(b)])
        with torch.inference_mode():
            out = model(images[None].to(runtime.device))
        directory = args.output_dir / f'pair_{index:04d}'
        write_iteration_overlays(images, None, out['H_updates_norm'][0],
            out['update_scale_schedule'], directory, pair_id=str(index),
            h0=out['H0_norm'][0], ghim_valid=bool(out['stage1_valid'][0]))
        (directory/'source.json').write_text(json.dumps({'image_A':str(a.resolve()),
            'image_B':str(b.resolve()), 'has_ground_truth':False}, indent=2)+'\n')
    (args.output_dir/'run.json').write_text(json.dumps({'status':'completed',
        'mode':'visualization_only', 'pairs':len(rows), 'accuracy_metrics':None,
        'checkpoint':str(args.checkpoint.resolve()), 'checkpoint_sha256':sha256_file(args.checkpoint),
        'csv_sha256':sha256_file(args.data_root/'test_pairs.csv')}, indent=2)+'\n')


if __name__ == '__main__':
    main()
