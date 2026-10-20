"""Register 32 training IDs, then calibrate geometry on real frozen context."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mhinet.config import RuntimePaths, sha256_file
from mhinet.v2.train import buffer_digest, timestamp, write_json
from .common import (SOURCE_SHA, TRAIN_SHA, VAL_SHA, code_hashes, data, load_source, setup)
from .objective import flow_loss_components


def select_training_ids(dataset, seed=42):
    buckets = defaultdict(list)
    for index, entry in enumerate(dataset.index):
        source = dataset._read_record(entry.offset)["source_dataset"]
        buckets[source].append(index)
    selected = []
    rng = np.random.default_rng(seed)
    for source, count in (("flair_rgb", 23), ("googleearth_current", 6), ("whu", 3)):
        for index in rng.choice(buckets[source], size=count, replace=False).tolist():
            selected.append(dict(index=index, pair_id=dataset.index[index].pair_id, source=source))
    return selected


def gradient_summary(terms, parameters):
    gradients = {}
    for key in ("geo_old", "geo_new", "mat", "cls"):
        weight = {"mat": .01, "cls": .0001}.get(key, 1.)
        gradients[key] = torch.autograd.grad(weight*terms[key], parameters,
                                             retain_graph=True, allow_unused=True)
    def inner(a, b):
        return sum(float((x.double()*y.double()).sum()) for x, y in zip(a, b)
                   if x is not None and y is not None)
    norms = {key: inner(values, values)**.5 for key, values in gradients.items()}
    cosine = {}
    for a, b in (("geo_old", "geo_new"), ("geo_old", "mat"), ("geo_new", "mat")):
        denom = norms[a]*norms[b]
        cosine[a+"_vs_"+b] = inner(gradients[a], gradients[b])/denom if denom else None
    return dict(norms=norms, cosines=cosine,
                connected={k: sum(x is not None for x in v) for k, v in gradients.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, help="Preregistered training IDs; exact deterministic selection must agree")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    setup()
    train, _ = data(RuntimePaths.from_json(args.runtime))
    selection = select_training_ids(train)
    if args.selection and json.loads(args.selection.read_text())["selected"] != selection:
        raise ValueError("Training calibration IDs differ from the preregistered selection")
    args.output.mkdir(parents=True, exist_ok=False)
    registration = dict(schema="mhi.flow-loss-calibration-manifest.v1", created_at=timestamp(),
        seed=42, train_sha256=TRAIN_SHA, val_sha256=VAL_SHA, selected=selection,
        counts={"flair_rgb": 23, "googleearth_current": 6, "whu": 3},
        policy="IDs fixed before gradients; only training data calibrates alpha")
    write_json(args.output/"calibration_manifest.json", registration)
    runtime, _, _, system, payload, _, _ = load_source(
        args.base_config, args.runtime, args.checkpoint, args.device)
    del payload
    system.eval()
    frozen = buffer_digest(system)
    def forbidden_fitter(module, inputs):
        raise RuntimeError("Flow calibration invoked forbidden homography fitting")
    hook = system.shared.stage1_head.homography_fitter.register_forward_pre_hook(forbidden_fitter)
    dustbin = system.shared.stage1_head.no_match_logit
    trunk = [(name, p) for name, p in system.named_parameters()
             if p.requires_grad and ".match_embedding_decoder.net." in name]
    if len(trunk) != 4:
        raise ValueError("Expected four shared decoder trunk tensors")
    rows = []
    started = time.perf_counter()
    for selected, batch in zip(selection, DataLoader(Subset(train, [r["index"] for r in selection]),
            batch_size=1, num_workers=2, generator=torch.Generator().manual_seed(42))):
        tensors = {k: v.to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        shared = system(tensors["images"])
        terms = flow_loss_components(shared["stage1"], tensors["H_gt_norm"],
            tensors["mask_A_overlap"], tensors["mask_B_overlap"])
        record = dict(**selected, **gradient_summary(terms, [p for _, p in trunk]),
                      metrics=terms["metrics"])
        record["dustbin_gradients"] = {}
        for key in ("geo_old", "geo_new", "mat", "cls"):
            weight = {"mat": .01, "cls": .0001}.get(key, 1.)
            value, = torch.autograd.grad(weight*terms[key], [dustbin],
                                        retain_graph=True, allow_unused=True)
            record["dustbin_gradients"][key] = float(value) if value is not None else None
        # A classification gradient on the shared trunk would change the causal premise.
        if record["connected"]["cls"] != 0:
            raise RuntimeError("Unexpected classification gradient to decoder trunk")
        rows.append(record)
        write_json(args.output/"progress.json", dict(status="calibrating", pairs=len(rows), total=32))
    old = float(np.median([r["norms"]["geo_old"] for r in rows]))
    new = float(np.median([r["norms"]["geo_new"] for r in rows]))
    if not np.isfinite(old) or not np.isfinite(new) or min(old, new) <= 0:
        raise FloatingPointError("Cannot calibrate nonpositive/nonfinite geometry gradients")
    if buffer_digest(system) != frozen:
        raise RuntimeError("Frozen buffers changed in calibration")
    hook.remove()
    report = dict(schema="mhi.flow-loss-calibration.v1", status="passed", alpha=old/new,
        epsilon_px=.5, mat_weight=.01, cls_weight=.0001, train_sha256=TRAIN_SHA,
        val_sha256=VAL_SHA, source_checkpoint=dict(path=str(args.checkpoint.resolve()), actual_sha256=SOURCE_SHA),
        selection_sha256=sha256_file(args.output/"calibration_manifest.json"), selected=selection,
        trunk_tensors=[n for n, _ in trunk], median_old_geo_gradient_norm=old,
        median_new_geo_gradient_norm=new, pairs=rows, implementation=code_hashes(runtime),
        preregistered_selection_sha256=sha256_file(args.selection) if args.selection else None,
        actual_homography_fitter_calls=0, frozen_buffer_sha256=frozen,
        seconds=time.perf_counter()-started, completed_at=timestamp())
    write_json(args.output/"calibration.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("pairs", "implementation", "selected")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
