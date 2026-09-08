# MHINet

This repository is the server implementation of the MHINet v1 architecture
and training protocol v1.2 from `MHINet_server_handoff_v1.2/`. The model is
described as three functional modules:

1. **Global Homography Initialization Module (GHIM):** shared DINOv3 and MVT
   infer the initial global homography `H0` and the cross-image context
   `F_MVT = (F_MVT^A, F_MVT^B)`.
2. **Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP):** MVT context
   and VGG multi-scale local features enter a DeDoDe-style cumulative decoder
   to produce the matching descriptors `D8`, `D4`, `D2`, and `D1`.
3. **Multi-scale Homography Iterative Refinement Module (MHIR):** an adapter,
   H-guided local correlation, and a refinement decoder perform two homography
   updates at each scale, giving `H0 -> H1 -> ... -> H8 = H_final`.

`Ds` means the fused matching descriptor at image scale `1/s`; it does not
name a DeDoDe layer or a feature produced by DeDoDe alone. In particular,
`D8/D4/D2/D1` are jointly generated from `F_MVT`, VGG features, and the
DeDoDe-style cumulative decoder. See
[`docs/model_architecture.md`](docs/model_architecture.md) for the complete
data flow and terminology.

The Python/config fields named `stage1_*` and the external LoRetta source names
are retained as legacy compatibility aliases for GHIM. They do not denote an
additional model stage. Planar, FGO, overlap, and auxiliary-correlation losses
are not part of the mainline implementation.

The implementation is not yet a validated model result.  P0--P3 and the P4
gradient/profile checks pass; the staged tiny-overfit gate is still being
completed.  E00/E01 must not run until the gate artifact says `passed` for all
four TINY-S experiments and TINY-8.  The detailed evidence and known failures
are in `implementation_log.md`.

## Server environment and resources

Run from `/home/disk1/MHINet` in the existing conda environment:

```bash
conda run --no-capture-output -n loma-repro python -m mhinet.cli --help
```

The resolved server paths live in `configs/runtime_paths.server.json`.  The
current resources are:

- LoMa source: `/home/disk1/LoMa`
- LoRetta source: `/home/disk1/LoMa/third_party/LoRetta`
- data: `/home/disk1/Data/datasets/GoogleEarth_scale_pairs`
- LoMa-B weights: `/root/.cache/torch/hub/checkpoints/loma_B.pt`
- shared official LoRetta weights: the resolved snapshot path recorded in the
  runtime config and its content-addressed blob/hash in `implementation_log.md`

Do not silently substitute checkpoints.  The test split is sealed and is not
used for model selection or the tiny diagnostics.

## Fast verification

```bash
conda run --no-capture-output -n loma-repro python -m compileall -q mhinet tests
conda run --no-capture-output -n loma-repro \
  python -m unittest discover -s tests -v

conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli geometry-corr-check \
  --output artifacts/p2_reference_checks.json --overwrite
```

The real-resource checks require a free RTX 4090.  Select one explicitly with
`CUDA_VISIBLE_DEVICES`; the runtime then sees it as `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli preflight \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_preflight.json --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli alignment \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_p1_alignment.json --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli zero-init-smoke \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p3_zero_init_smoke.json --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli gradient-audit \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p4_gradient_audit.json --overwrite
```

## Tiny-overfit gate

The baseline command keeps B=1, effective batch 4, AdamW lr 1e-3, zero weight
decay, no scheduler, clip 1, and trains only adapters/refinement decoders.  It
uses controlled H0 only inside the diagnostic, never in formal training.

Long jobs are run one experiment at a time because D1 uses the full legal
residual fraction while D8/D4/D2/TINY-8 use `0.5`. This is the registered dense
D1 5x5 command:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli tiny-overfit \
  --runtime configs/runtime_paths.server.json --experiments D1 \
  --sample-protocol one_pair_residuals \
  --residual-profile translation --residual-bound-fraction 1.0 \
  --precision bf16 --sample-count 32 --seed 0 \
  --max-steps 2000 --eval-interval 32 --threshold-mace-px 0.1 \
  --weight-average-start-step 1536 --heartbeat-interval 8 \
  --progress-checkpoint outputs/tiny_overfit/d1_dense5_progress.pt \
  --output artifacts/p4_tiny_s_d1_translation_swa.json --overwrite
```

If the process is interrupted, repeat every trajectory-defining argument and
resume the raw optimizer boundary. Keep the same one-visible-GPU topology:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli tiny-overfit \
  --runtime configs/runtime_paths.server.json --experiments D1 \
  --sample-protocol one_pair_residuals \
  --residual-profile translation --residual-bound-fraction 1.0 \
  --precision bf16 --sample-count 32 --seed 0 \
  --max-steps 2000 --eval-interval 32 --threshold-mace-px 0.1 \
  --weight-average-start-step 1536 --heartbeat-interval 8 \
  --resume-progress outputs/tiny_overfit/d1_dense5_progress.pt \
  --output artifacts/p4_tiny_s_d1_translation_swa.json
```

The registered diagnostic uses `--weight-average-start-step 1536` to expose a
tiny-only equal-weight parameter-average readout while preserving every raw
endpoint metric.  It does not alter the optimizer, loss, or formal trainer and
must be reported as such; an averaged final tiny checkpoint is marked
non-resumable. The separate progress checkpoint is atomically written at raw
optimizer boundaries, contains RNG/data cursor/history/averager state, and is
strictly bound to architecture, data manifest, runtime and weight hashes.
Omitting the averaging flag is useful for raw-endpoint failure isolation, but
that output cannot be merged into the registered gate.

Run D8/D4/D2/TINY-8 with the same recipe but
`--residual-bound-fraction 0.5`, distinct progress/output paths and the matching
`--experiments` value. Long jobs may be merged only after all five artifacts
pass. The merge fails closed on missing experiments, protocol
differences, failed/rejected geometry, or a metric at/above 0.1 px:

```bash
conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli tiny-gate-merge \
  --inputs artifacts/p4_tiny_s_d8_translation_swa.json \
           artifacts/p4_tiny_s_d4_translation_swa.json \
           artifacts/p4_tiny_s_d2_translation_swa.json \
           artifacts/p4_tiny_s_d1_translation_swa.json \
           artifacts/p4_tiny_8_translation_swa.json \
  --output artifacts/p4_tiny_gate.json --overwrite
```

## D1 efficiency ablations

The implemented reference remains dense D1 correlation over all
`784 x 784 = 614,656` source positions with a `5 x 5` target search window.
Its results are the baseline for two planned, not-yet-implemented ablations:
sparse D1 queries guided by D2 support, and a `3 x 3` D1 search window. The
mainline tiny gate must pass before either ablation is evaluated. The frozen
comparison protocol, trajectory metrics, query recall/coverage diagnostics,
latency, and memory fields are registered in
[`docs/experiment_plan.md`](docs/experiment_plan.md).

## Minimal training and exact-boundary resume

The following two-step configuration is an entrypoint/resume smoke, not E01
and not evidence of accuracy:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli train \
  --runtime configs/runtime_paths.server.json \
  --config configs/train_minimal_smoke.json \
  --output-dir outputs/mhinet_resume_smoke \
  --stop-after-optimizer-step 1 --overwrite

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli resume \
  --runtime configs/runtime_paths.server.json \
  --config configs/train_minimal_smoke.json \
  --output-dir outputs/mhinet_resume_smoke \
  --resume outputs/mhinet_resume_smoke/checkpoints/step_0000001.pt
```

Formal E01 is gated.  Once and only once a complete tiny artifact has passed:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.cli train \
  --runtime configs/runtime_paths.server.json \
  --config configs/e01_heads_v1.2.json \
  --tiny-gate-artifact artifacts/p4_tiny_gate.json \
  --output-dir outputs/E01_heads_seed0
```

## Evaluation and profiling

Evaluation reports H0, every H update, final H, failures, input/native-pixel
geometry, per-pair rows, latency, and peak allocation.  Use `val` during model
development.

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli evaluate \
  --runtime configs/runtime_paths.server.json --split val \
  --checkpoint outputs/E01_heads_seed0/checkpoints/step_0010000.pt \
  --output-dir outputs/E01_heads_seed0/final_val

CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.cli profile \
  --runtime configs/runtime_paths.server.json --profile joint \
  --optimizer-steps 2 --output artifacts/p4_profile_joint_784.json --overwrite
```

`completed` in a training `run.json` means only that the requested optimizer
budget finished.  Model validation requires the recorded validation metrics;
code completion and checkpoint creation are never labeled as accuracy success.
