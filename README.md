# MHINet

This repository implements MHINet under training protocol v1.2. The current
mainline is deliberately truncated at `D2`: it has three functional modules
and six homography updates, not the former four-scale/eight-update path.

1. **Global Homography Initialization Module (GHIM):** shared DINOv3 and MVT
   infer the initial global homography `H0` and cross-image context
   `F_MVT = (F_MVT^A, F_MVT^B)`.
2. **Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP):** MVT context
   and VGG local features enter a DeDoDe-style cumulative decoder with active
   flow `F_MVT -> D16 -> D8 -> D4 -> D2`.
3. **Multi-scale Homography Iterative Refinement Module (MHIR):** an adapter,
   H-guided local correlation, and an MCNet-style correlation decoder perform
   two updates at each of `D8`, `D4`, and `D2`, giving
   `H0 -> H1 -> ... -> H6 = H_final`.

`Ds` means a fused matching descriptor at image scale `1/s`; it does not name
a DeDoDe layer or a feature produced by DeDoDe alone. The D1 implementation is
retained for future controlled comparisons, but the main model neither asks
CGMDP to decode `D1` nor executes the D1 adapter, correlation, or refinement
decoder. Thus the inactive D1 code has no full-resolution feature or compute
cost in a normal mainline forward. See
[`docs/model_architecture.md`](docs/model_architecture.md) for the precise
contracts and MCNet adaptation.

Python/config fields named `stage1_*` and external LoRetta source names remain
legacy compatibility aliases for GHIM. They do not denote another model
stage. Planar, FGO, overlap, and auxiliary-correlation losses remain disabled
on the mainline.

## Code layout and shell entry points

The Python source is organized by responsibility (not experiment run):

```text
mhinet/
  models/          # GHIM/CGMDP provider, MHIR layers/iteration, complete model
  ops/             # guarded homography geometry and local correlation
  engine/          # training, evaluation, losses, metrics, checkpoints
  dataio/          # pair loading and geographic grouping
  diagnostics/     # alignment, gradients, profiling, tiny-overfit and gates
  visualization/   # H0 plus six refinement overlay images
  config.py        # configuration and resource paths
  cli.py           # stable unified command dispatcher
  __init__.py
scripts/
  train_e01.sh      # formal E01, physical GPU 1 by default
  test_e00.sh       # pretrained H0 baseline on the full test split (1000 pairs)
  test.sh           # real-image accuracy evaluation with an explicit checkpoint
  unit_tests.sh     # CPU engineering unit tests
```

```bash
# Unit tests only; no real dataset or GPU required
bash scripts/unit_tests.sh

# Real-image accuracy, full test split by default (replace with your checkpoint)
bash scripts/test.sh /absolute/path/to/checkpoint.pt

# Optional small evaluation subset; not a replacement for full test results
bash scripts/test.sh /absolute/path/to/checkpoint.pt --max-pairs 16

# Inspect the command without training
DRY_RUN=1 bash scripts/train_e01.sh

# Independent E00 H0 baseline (no training checkpoint needed)
bash scripts/test_e00.sh

# Start formal E01 on physical GPU 1 (train=train, in-training validation=val)
bash scripts/train_e01.sh

# Optional GPU override or resume from an existing checkpoint
GPU_ID=2 bash scripts/train_e01.sh
bash scripts/train_e01.sh --resume /absolute/path/to/step_0000500.pt
```

These scripts resolve the repository from their own location and can be invoked
from another working directory. They use the existing `loma-repro` interpreter;
override `MHINET_PYTHON` when relocating the environment. Training retains the
existing configuration, tiny gate, progress bars and seven validation overlays.
It does not silently overwrite output or automatically run E00.

Real-image evaluation writes `report.md`, `metrics/summary.json`, per-pair
CSV/JSONL and `visualizations/pair_0000/` (H0 + six updates). It reports MACE,
5x5 grid error, all-pair success@1/3/5 px and normalized empirical-recall
AUC@1/3/5 px, invalid/rejected rates, memory and latency. Metrics include H0
through H6; failed outputs remain in the success/AUC denominator. Training uses
train (36000 pairs), in-training validation uses val (2500 pairs), and independent
evaluation (including E00) defaults to test (1000 pairs). Test results must not
be used to select hyperparameters or checkpoints. Full command
examples and metric conventions: [real-image evaluation](docs/real_image_evaluation.md).

`python -m mhinet.cli <command>` is unchanged. Direct Python imports now use
the subpackages, e.g. `from mhinet.models.model import MHINet` and
`from mhinet.engine.train import TrainConfig`. Historical logs/artifacts retain
their original paths and hashes; they are not rewritten as new evidence.

## Validation status

The MCNet-style MHIR rewrite and the D2 truncation change the architecture
identity. Therefore former D1, TINY-8, decoder zero-init, gradient-audit, and
eight-update tiny-overfit artifacts are **legacy evidence only** and cannot
open the current training gate. The current `TINY-S-D8`, `TINY-S-D4`,
`TINY-S-D2`, and `TINY-6` checks have passed, together with P0–P4 engineering
checks, independent checkpoint replays, and the merged gate. Work is stopped
before E00/E01 as requested. D8/D4/TINY-6 use the preregistered averaged
diagnostic readout; D2 passes with raw parameters. See the full
[pretraining results table](docs/pretraining_mcnet_d2_summary.md) and
[interpretation](docs/results_mcnet_d2.md). These checks use one exact training
image pair with 32 controlled H0 conditions; they do not demonstrate held-out
accuracy or joint-training convergence.

An artifact saying a command completed is not evidence that the model is
accurate. Model validation additionally requires the registered validation
metrics, failure rates, resource measurements, and comparisons. Current and
historical evidence is recorded in `implementation_log.md`.

## Server environment and resources

Run from `/home/disk1/MHINet` in the existing conda environment:

```bash
conda run --no-capture-output -n loma-repro python -m mhinet.cli --help
```

Resolved server paths live in `configs/runtime_paths.server.json`. The known
resources are:

- LoMa source: `/home/disk1/LoMa`
- LoRetta source: `/home/disk1/LoMa/third_party/LoRetta`
- data: `/home/disk1/Data/datasets/GoogleEarth_scale_pairs`
- LoMa-B weights: `/root/.cache/torch/hub/checkpoints/loma_B.pt`
- shared official LoRetta weights: the resolved snapshot and content-addressed
  hash recorded in `implementation_log.md`

Do not silently substitute checkpoints. The test split is sealed and is not
used for implementation choices, tiny diagnostics, checkpoint selection, or
ablation selection.

## Fast verification

```bash
conda run --no-capture-output -n loma-repro python -m compileall -q mhinet tests
conda run --no-capture-output -n loma-repro \
  python -m unittest discover -s tests -v

conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli geometry-corr-check \
  --output artifacts/p2_reference_checks.json --overwrite
```

The real-resource checks require a free CUDA device. Select it explicitly
with `CUDA_VISIBLE_DEVICES`; the process then sees that device as `cuda:0`.
Use new artifact names so the MCNet/D2 results cannot overwrite or be confused
with the superseded eight-update artifacts:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli preflight \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_preflight_mcnet_d2.json --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli alignment \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_p1_alignment_mcnet_d2.json --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli zero-init-smoke \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p3_zero_init_smoke_mcnet_d2.json --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli gradient-audit \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p4_gradient_audit_mcnet_d2.json --overwrite
```

Zero initialization can make gradients before the output projection zero on
the first optimizer step. The audit must therefore check the second and later
steps, both MVT gradient routes, and that freezing the GHIM head does not wrap
its forward in `no_grad` or `inference_mode`. `H0` and the cumulative corner
state must stay attached through all six updates.

## Tiny-overfit gate

The current P4 gate contains four independent experiments:

| Experiment | Active scales | Updates |
| --- | --- | ---: |
| `TINY-S-D8` | `D8` | 2 |
| `TINY-S-D4` | `D4` | 2 |
| `TINY-S-D2` | `D2` | 2 |
| `TINY-6` | `D8 -> D4 -> D2` | 6 |

The baseline diagnostic uses B=1, effective batch 4, AdamW lr `1e-3`, zero
weight decay, no scheduler, clip 1, and only the adapters/refinement decoders.
Controlled `H0` belongs only to this diagnostic and is never substituted into
formal training. Run one experiment per process with distinct output and
progress paths; for example:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli tiny-overfit \
  --runtime configs/runtime_paths.server.json --experiments D2 \
  --sample-protocol one_pair_residuals \
  --residual-profile translation --residual-bound-fraction 0.5 \
  --precision bf16 --sample-count 32 --seed 0 \
  --max-steps 2000 --eval-interval 32 --threshold-mace-px 0.1 \
  --weight-average-start-step 1536 --heartbeat-interval 8 \
  --progress-checkpoint outputs/tiny_overfit/mcnet_d2_scale_d2_progress.pt \
  --output artifacts/p4_tiny_s_d2_mcnet.json --overwrite
```

Repeat with `D8`, `D4`, and `TINY-6`, changing both paths. If a process is
interrupted, repeat every trajectory-defining argument and replace
`--progress-checkpoint PATH --overwrite` with `--resume-progress PATH`. A
progress checkpoint is accepted only when its architecture, data manifest,
runtime, weight hashes, RNG/data cursor, and protocol signature match exactly.
Never resume a pre-rewrite D1 or TINY-8 checkpoint into this model.

Merge only after all four current artifacts have completed and passed:

```bash
conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli tiny-gate-merge \
  --inputs artifacts/p4_tiny_s_d8_mcnet.json \
           artifacts/p4_tiny_s_d4_mcnet.json \
           artifacts/p4_tiny_s_d2_mcnet.json \
           artifacts/p4_tiny_6_mcnet.json \
  --output artifacts/p4_tiny_gate_mcnet_d2.json --overwrite
```

The merger must fail closed on a missing experiment, an architecture/protocol
hash mismatch, rejected/non-finite geometry, or a registered metric that does
not pass. Weight-averaged tiny readout is a diagnostic and remains
non-resumable; raw endpoint metrics must still be retained.

## D1 is deferred, not deleted

D1 adapter/correlation/decoder code remains available for a future explicitly
selected experiment. It is not part of the current model, current P4 gate, or
E00--E05. In particular, CGMDP stops after `D2`, so merely leaving D1 classes
in the source does not allocate a `784 x 784` descriptor.

The old dense-D1 5x5 runs, interrupted checkpoints, TINY-8 gate design, and
sparse/3x3 proposals are archived as legacy/pending revalidation. If D1 is
reactivated later, compare it against the frozen D2 mainline from a separately
registered branch, report accuracy as well as latency/memory, and never use
old D1 results as if they came from the MCNet-style decoder. The deferred plan
is detailed in [`docs/experiment_plan.md`](docs/experiment_plan.md).

## Minimal training and exact-boundary resume

The following two-step run checks the entrypoint and exact optimizer-boundary
resume only; it is not E01 and is not accuracy evidence:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli train \
  --runtime configs/runtime_paths.server.json \
  --config configs/train_minimal_smoke.json \
  --output-dir outputs/mhinet_resume_smoke_mcnet_d2 \
  --stop-after-optimizer-step 1 --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli resume \
  --runtime configs/runtime_paths.server.json \
  --config configs/train_minimal_smoke.json \
  --output-dir outputs/mhinet_resume_smoke_mcnet_d2 \
  --resume outputs/mhinet_resume_smoke_mcnet_d2/checkpoints/step_0000001.pt
```

Formal E01 is gated. Run it only after the current four-experiment gate is
explicitly `passed`:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli train \
  --runtime configs/runtime_paths.server.json \
  --config configs/e01_heads_v1.2.json \
  --tiny-gate-artifact artifacts/p4_tiny_gate_mcnet_d2.json \
  --output-dir outputs/E01_heads_mcnet_d2_seed0
```

## Evaluation and profiling

Evaluation must report `H0`, all six update states, explicit `H_final`, solve
failures, input/native-pixel geometry, per-pair rows, latency, and peak memory.
Use grouped `val` during development.

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli evaluate \
  --runtime configs/runtime_paths.server.json --split val \
  --checkpoint outputs/E01_heads_mcnet_d2_seed0/checkpoints/step_0010000.pt \
  --output-dir outputs/E01_heads_mcnet_d2_seed0/final_val

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli profile \
  --runtime configs/runtime_paths.server.json --profile joint \
  --optimizer-steps 2 \
  --output artifacts/p4_profile_joint_mcnet_d2.json --overwrite
```

`completed` in `run.json` means only that the requested optimizer budget
finished. It must never be relabeled as model validation success without the
registered held-out evidence.
