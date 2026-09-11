"""Fail closed if a temporal-four-pair dataset is incomplete or a smoke set."""
import argparse
import json
from pathlib import Path
from collections import Counter
from mhinet.dataio.data import parse_geo_region


def check_ready(root):
    root = Path(root)
    summary = json.loads((root / 'dataset_summary.json').read_text())
    if summary.get('status') != 'completed' or summary.get('smoke_only') is not False:
        raise ValueError('数据尚未生成完成，或是smoke数据；请等待完整生成结束后再启动训练')
    if summary.get('version') != 'temporal_four_pairs_v1':
        raise ValueError('不是预期的 temporal_four_pairs_v1 数据集')
    regions, parents, counts = {}, {}, {}
    kinds = {'same_past', 'same_current', 'cross_past_current', 'cross_current_past'}
    for split in ('train', 'val'):
        regions[split], parents[split] = set(), set()
        groups = {}
        ids = set()
        with (root / split / 'pairs.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['pair_id'] in ids:
                    raise ValueError(f'{split}存在重复pair_id')
                ids.add(row['pair_id'])
                groups.setdefault(row['parent_temporal_group'], Counter())[row['pair_kind']] += 1
                regions[split].add(parse_geo_region(row))
                parents[split].update((row['parent_image_A'], row['parent_image_B']))
                for key in ('image_A', 'image_B', 'mask_A_overlap', 'mask_B_overlap'):
                    if not (root / split / row[key]).is_file():
                        raise FileNotFoundError(root / split / row[key])
        expected = summary['planned'][split]
        if (len(ids) != expected['pairs'] or len(ids) != expected['generated_pairs']
                or len(groups) != expected['parent_groups']
                or any(set(c) != kinds or any(n != 1 for n in c.values()) for c in groups.values())):
            raise ValueError(f'{split}实际清单与完成摘要不一致，或母图组不是四种配对各一对')
        counts[split] = len(ids)
    if regions['train'] & regions['val'] or parents['train'] & parents['val']:
        raise ValueError('train/val母图或地理区域有交集')
    if not (root / 'test/pairs.jsonl').is_file():
        raise FileNotFoundError(root / 'test/pairs.jsonl')
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    runtime = json.loads(args.runtime.read_text())
    try:
        counts = check_ready(runtime['data_root'])
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'训练启动检查未通过：{exc}\n')
    print(f"新数据集检查通过：{runtime['data_root']}，train={counts['train']}，val={counts['val']}", flush=True)


if __name__ == '__main__':
    main()
