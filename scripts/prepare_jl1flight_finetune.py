"""Prepare native-GT JL train-only manifests for the existing 512 loader."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/disk1/Data/datasets/jl1flight/train/affine_pairs_train')
INDEX = Path('/home/disk1/Data/datasets/jl1flight/index/scene_info/affine_pairs_train.npz')
OUTPUT = ROOT / 'outputs/jl1flight_finetune512_dataset'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    cv2.setNumThreads(1)
    archive = np.load(INDEX, allow_pickle=False)
    paths = archive['image_paths']
    pairs = archive['pair_infos']
    affines = archive['affine_matrices']
    bases = sorted(SOURCE.glob('a_*.png'))
    columns = sorted({int(p.stem.split('_')[2]) for p in bases})
    cutoff = columns[-3]
    # Spatial stripe holdout; explicitly check native tile footprints across it.
    train_bases = [p for p in bases if int(p.stem.split('_')[2]) < cutoff]
    assert max(int(p.stem.split('_')[2]) + Image.open(p).width for p in train_bases) <= cutoff
    counts = {}; ids = {}; handles = {}
    for split in ('train', 'val'):
        folder = OUTPUT / split
        (folder / 'masks').mkdir(parents=True)
        handles[split] = (folder / 'pairs.jsonl').open('w')
        counts[split] = 0; ids[split] = set()
    try:
        for ai, bi in pairs:
            a = SOURCE / Path(str(paths[ai])).name
            b = SOURCE / Path(str(paths[bi])).name
            base = a.stem[2:]
            split = 'val' if int(base.split('_')[1]) >= cutoff else 'train'
            assert b.stem.startswith('b_' + base + '_t')
            h = np.loadtxt(b.with_name(b.stem + '_H_0to1.txt'))
            aa = np.vstack([affines[ai], [0., 0., 1.]])
            bb = np.vstack([affines[bi], [0., 0., 1.]])
            assert np.allclose(h, bb @ np.linalg.inv(aa), atol=1e-4), b
            assert h.shape == (3, 3) and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-8
            with Image.open(a) as image: sa = image.size
            with Image.open(b) as image: sb = image.size
            assert sa == sb == (640, 640)
            one = np.full((640, 640), 255, dtype=np.uint8)
            masks = [cv2.warpPerspective(one, np.linalg.inv(h), sa, flags=cv2.INTER_NEAREST),
                     cv2.warpPerspective(one, h, sb, flags=cv2.INTER_NEAREST)]
            record = dict(pair_id=b.stem, image_A=str(a), image_B=str(b),
                          size_A=sa, size_B=sb, H_A_to_B=h.tolist(),
                          parent_image_A=str(a), parent_image_B=str(a),
                          source_dataset='jl1flight_train', source_path=a.name)
            for side, mask in zip(('A', 'B'), masks):
                name = f'masks/{b.stem}_{side}.png'
                assert mask.any()
                assert cv2.imwrite(str(OUTPUT / split / name), mask)
                record[f'mask_{side}_overlap'] = name
            handles[split].write(json.dumps(record) + '\n')
            counts[split] += 1; ids[split].add(base)
    finally:
        for handle in handles.values(): handle.close()
    assert counts['train'] + counts['val'] == 7490
    assert not ids['train'] & ids['val']
    summary = dict(source=str(SOURCE), source_index=str(INDEX),
                   source_index_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(),
                   counts=counts, parent_counts={s:len(v) for s,v in ids.items()},
                   validation_policy=f'last 3 columns of native training mosaic; second filename coordinate >= {cutoff}',
                   train_val_native_tile_overlap=False, test_used=False,
                   native_size=640, model_input_size=512,
                   resizing='Existing PIL bicubic 512 loader, align_corners=False normalized GT; H files remain native A-to-B',
                   masks='Native geometric overlap by H/inverse(H), nearest-neighbor; no added augmentation',
                   official_val_excluded='Official val points to affine_pairs_test; withheld from training/validation')
    (OUTPUT/'dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
