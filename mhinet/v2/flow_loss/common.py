"""Pinned data/source contracts shared by the isolated loss experiment."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import torch

from mhinet.config import RuntimePaths, sha256_file
from mhinet.v2.data import RGB512Dataset
from mhinet.v2.train import initialize_system, read_config, seed_all, implementation_hashes

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SHA = "26a24e5239a1b7a3719c638ec7b040a8490c226a7341a0606137d25f0c63c33a"
TRAIN_SHA = "11f37bcfcebc61a24310421a5eff2e1921c618d461b8dac8142ff050843f88dc"
VAL_SHA = "ad281e31dfd612a239ddfa63f25a42812511e22445f571359477e709c8207f5b"
SOURCE_STEP = 25535
EXTRA_STEPS = 5107


def digest_tree(value):
    """Lossless digest, including BF16 tensors and container ordering/types."""
    digest = hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            digest.update(str((item.dtype, tuple(item.shape))).encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def setup(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    seed_all(seed)


def data(runtime):
    train = RGB512Dataset(runtime.data_root/"train/pairs.jsonl")
    val = RGB512Dataset(runtime.data_root/"val/pairs.jsonl")
    if (len(train), len(val)) != (40849, 5106):
        raise ValueError("Expected complete RGB512 train40849/val5106")
    if sha256_file(train.manifest) != TRAIN_SHA or sha256_file(val.manifest) != VAL_SHA:
        raise ValueError("Pinned RGB512 manifest identity changed")
    if {r.parent_group for r in train.index} & {r.parent_group for r in val.index}:
        raise ValueError("Train/val source-image overlap")
    return train, val


def load_source(base_config, runtime_path, checkpoint, device):
    config, matcher = read_config(base_config)
    if config["stage"] != "flow_head":
        raise ValueError("Flow-head base configuration required")
    if sha256_file(checkpoint) != SOURCE_SHA:
        raise ValueError("Flow-best source SHA mismatch")
    runtime = replace(RuntimePaths.from_json(runtime_path), device=device)
    config = {**config, "runtime": str(Path(runtime_path).resolve())}
    system, provider, migration = initialize_system(runtime, config, matcher, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (payload["progress"]["optimizer_step"] != SOURCE_STEP
            or payload["progress"]["data"] != {"epoch": 4, "cursor": 40849}
            or payload["metadata"]["train_sha256"] != TRAIN_SHA
            or payload["metadata"]["val_sha256"] != VAL_SHA
            or payload["metadata"]["trainable_parameters"] != system.training_parameter_names()
            or len(system.training_parameter_names()) != 10):
        raise ValueError("Source progress/data/optimizer parameter topology mismatch")
    system.load_state_dict(payload["model"], strict=True)
    return runtime, config, matcher, system, payload, provider, migration


def code_hashes(runtime):
    hashes = implementation_hashes(runtime)
    hashes.update({str(p.relative_to(ROOT)): sha256_file(p)
                   for p in Path(__file__).parent.glob("*.py")})
    return hashes


def read_calibration(path, epsilon=.5):
    report = json.loads(Path(path).read_text())
    alpha = report.get("alpha")
    import math
    if (report.get("schema") != "mhi.flow-loss-calibration.v1"
            or report.get("status") != "passed"
            or report.get("source_checkpoint", {}).get("actual_sha256") != SOURCE_SHA
            or report.get("train_sha256") != TRAIN_SHA or report.get("val_sha256") != VAL_SHA
            or report.get("epsilon_px") != epsilon
            or isinstance(alpha, bool) or not isinstance(alpha, (float, int))
            or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Calibration identity/value mismatch")
    return report
