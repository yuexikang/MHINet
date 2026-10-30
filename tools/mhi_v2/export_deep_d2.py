"""Package the validated D2-only depth checkpoint with its frozen release components."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--deep', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = torch.load(args.base, map_location='cpu', weights_only=False, mmap=True)
    deep = torch.load(args.deep, map_location='cpu', weights_only=False)
    if deep['variant'] != 'deep':
        raise ValueError('Only the validated deep variant belongs in this release')
    # All other pyramid layers, VGG and temperatures stay exactly as in the base.
    pyramid = {k: v for k, v in base['dedode'].items() if not k.startswith('layers.2.')}
    pyramid.update({f'layers.2.{k}': v for k, v in deep['d2'].items()})
    state = dict(stage='d2', epoch=deep['epoch'], score=deep['score'],
                 d2_refinement='residual_deep', matcher_config=base['matcher_config'],
                 locator_checkpoint=base['locator_checkpoint'], vgg=base['vgg'],
                 dedode=pyramid, temperatures=base['temperatures'], adapter=deep['adapter'],
                 config=deep['config'],
                 provenance=dict(base=str(args.base), deep=str(args.deep),
                                 experiment_commit='766f44500eb8f245642a36f4f966f1710f3ee329',
                                 optimizer_included=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(json.dumps(dict(output=str(args.output), epoch=state['epoch'], score=state['score'])))


if __name__ == '__main__':
    main()
