"""Compare the deployed deep release to the actual experiment on fixed inputs."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--rgb-root', type=Path, required=True)
    parser.add_argument('--hangzhou-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    import mhinet.v2
    mhinet.v2.__path__.append(str(args.experiment_root/'mhinet/v2'))
    from mhinet.v2.d2_capacity.model import attach, restore
    from mhinet.v2.pipeline import load_mhi_v2
    from mhinet.v2.data import GeometryPairDataset
    from mhinet.v2.pair_matching.metrics import measure
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(True)
    runtime = json.loads(args.runtime.read_text())
    old = load_mhi_v2(ROOT/'configs/mhi_v2/releases/flow_pair_v2_base.json', args.device, runtime=runtime)
    attach(old, 'deep')
    restore(old, torch.load(args.run/'best.pt', map_location='cpu', weights_only=False))
    model = load_mhi_v2(device=args.device, runtime=runtime)
    # Entire assembled weight set must match, not just the new residual blocks.
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    rows = []
    for split, root, manifest in (
        ('test', args.rgb_root/'test', args.rgb_root/'test/pairs.jsonl'),
        ('hangzhou_test_real', args.hangzhou_root, args.hangzhou_root/'manifests/test_real.jsonl'),
    ):
        records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        data = GeometryPairDataset(root, records[:16])
        archived = json.loads((args.run/f'best_{split}.json').read_text())['pairs']
        for index in range(len(data)):
            batch = data[index]
            expected = old.match(batch['images'], diagnostics=True)
            actual = model.match(batch['images'], diagnostics=True)
            for field in ('points_a', 'points_b', 'confidence', 'valid', 'parent_ids'):
                torch.testing.assert_close(expected[field], actual[field], rtol=0, atol=0)
            for field in ('flow16_norm', 'parents', 'centers', 'candidate_logp'):
                torch.testing.assert_close(expected['diagnostics'][field], actual['diagnostics'][field], rtol=0, atol=0)
            counts = measure(actual['points_a'], actual['points_b'], batch['geometry'].to(args.device))[0]
            assert batch['pair_id'][0] == archived[index]['pair_id']
            for key in ('n', 'visible', 'correct1', 'correct3', 'correct5', 'correct10'):
                assert counts[key] == archived[index][key], (split, index, key)
            rows.append(dict(split=split, pair_id=batch['pair_id'][0], counts=counts))
        print(f'{split}: 16 pairs exact, archived counts agree', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(release=model.release, exact_weight_match=True,
                                         exact_output_match=True, pairs=rows), indent=2))
    print('DEEP RELEASE EQUIVALENCE PASSED', flush=True)


if __name__ == '__main__':
    main()
