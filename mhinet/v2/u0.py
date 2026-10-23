"""U0 D16 Flow0 inference, independent of experiment/report packages."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mhinet.config import RuntimePaths
from .feature_provider import build_fresh_shared
from .frozen_locator import load_locator_weights

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/mhi_v2/hangzhou_u0.json'
FLOW_FIELDS = ('flow16_target_norm', 'flow16_norm', 'flow16_matchability', 'flow16_valid')


def load_u0(config=DEFAULT_CONFIG, device='cuda:0'):
    """Return frozen DINO/MVT/original Flow head; use pyramid_scales=()."""
    settings = json.loads(Path(config).read_text())
    runtime = replace(RuntimePaths.from_json(settings['runtime']), device=device)
    model = build_fresh_shared(runtime)
    load_locator_weights(model, settings['frozen_locator_checkpoint'])
    model.requires_grad_(False)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_a', type=Path)
    parser.add_argument('image_b', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, default=Path('outputs/mhi_v2/flow0.pt'))
    args = parser.parse_args()
    images = []
    for path in (args.image_a, args.image_b):
        with Image.open(path) as image:
            if image.size != (512, 512):
                raise ValueError(f'{path}: U0 requires 512x512 images, got {image.size}')
            images.append(torch.from_numpy(np.array(image.convert('RGB'))).permute(2, 0, 1).float()/255)
    model = load_u0(args.config, args.device)
    with torch.inference_mode():
        result = model(torch.stack(images)[None].to(args.device), pyramid_scales=())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: result[key].cpu() for key in FLOW_FIELDS}, args.output)
    print(f'Saved 32x32 Flow0 to {args.output}')


if __name__ == '__main__':
    main()
