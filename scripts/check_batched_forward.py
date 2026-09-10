"""Compare two distinct pairs in batch, serially, and in reversed batch order."""
import argparse
import json
from pathlib import Path

import torch
from mhinet.config import RuntimePaths
from mhinet.models.model import build_model
from mhinet.engine.train import _safe_train_dataset, _seed_everything
from mhinet.ops.geometry import image_corners, safe_project_points, normalized_to_pixel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    runtime = RuntimePaths.from_json(args.runtime)
    _seed_everything(0)
    model, build = build_model(runtime)
    model.eval()
    dataset, _ = _safe_train_dataset(runtime, max_pairs=2)
    images = torch.stack([dataset[i]["images"] for i in range(2)]).to(runtime.device)
    with torch.no_grad():
        batched = model(images)
        reversed_batch = model(images.flip(0))
        serial = [model(images[i:i+1]) for i in range(2)]
    corners = image_corners((784, 784), normalized=True, device=images.device).unsqueeze(0).expand(2, -1, -1)
    def project(h):
        points, valid, _ = safe_project_points(h.float(), corners)
        if not bool(valid.all()):
            raise AssertionError("invalid projected corners")
        return normalized_to_pixel(points, (784, 784))
    a = project(batched["H0_norm"])
    b = project(torch.cat([s["H0_norm"] for s in serial]))
    c = project(reversed_batch["H0_norm"].flip(0))
    serial_diff = float((a-b).abs().max())
    order_diff = float((a-c).abs().max())
    passed = serial_diff < 0.1 and order_diff < 0.01 and bool(batched["stage1_valid"].all())
    result = {"status": "passed" if passed else "failed", "serial_H0_corner_max_abs_px": serial_diff,
              "reversed_order_H0_corner_max_abs_px": order_diff,
              "H_shape": list(batched["H_updates_norm"].shape),
              "call_counts": batched["shared_call_counts"],
              "pair_ids": [dataset[i]["pair_id"] for i in range(2)],
              "architecture_sha256": build["architecture_sha256"],
              "scope": "heads eval, zero initialized MHIR; pair isolation and BF16 batch arithmetic drift"}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(result)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
