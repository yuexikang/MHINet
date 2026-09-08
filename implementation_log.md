# MHINet implementation log

This is an evidence log, not a claim that the model has been validated.  The
active requirements are the server handoff package training protocol v1.2 and
loss revision 1.1.  Files under `history/` were not used as implementation
instructions.

## 2026-09-08 — design package and repository

- Project/repository: `/home/disk1/MHINet`; initialized with `git init -b main`.
  The first implementation/evidence milestone through the D8/D4 tiny gates is
  commit `5ff680b` (`Implement MHINet v1 through P4 D8/D4 gates`).
- Design package: `/home/disk1/MHINet/MHINet_server_handoff_v1.2`.
- Read order followed: `START_HERE.md`, docs `01`, `03`, `05`, `06`, `02`,
  `07`, `08`, then all files under `configs/`.
- Package integrity command:

  ```bash
  conda run --no-capture-output -n loma-repro \
    python MHINet_server_handoff_v1.2/verify_package.py \
    MHINet_server_handoff_v1.2
  ```

  Result: 31 files checked, zero missing or hash errors.
- Architecture config:
  `/home/disk1/MHINet/MHINet_server_handoff_v1.2/configs/mhinet_v1.json`,
  SHA256 `4a859f413ee4be59acb1dc6c0cf189e80b8f4e7b3f54dc9f117acd1ac5cfe128`.
- Server runtime config written to
  `/home/disk1/MHINet/configs/runtime_paths.server.json`; model structure has no
  hard-coded server paths.

## P0 — real resources and legacy alignment: passed

Command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loma-repro python -m mhinet.preflight \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_preflight.json --overwrite
```

Evidence: `/home/disk1/MHINet/artifacts/p0_preflight.json`, artifact SHA256
`92edb6d2e79691df0d44d119ea13c249fe9347acb750c202fc9d9ca4685ce5cc`.

Resolved source and weights:

- LoMa: `/home/disk1/LoMa`, clean commit
  `899fdb99a4076e312196bbbf99a3c329739b5d7a`, branch
  `codex/stage1-dedode-pyramid-hroi-v1`.
- LoRetta source: `/home/disk1/LoMa/third_party/LoRetta`, clean commit
  `e274bb5628e6324dbe36a1590d88e5adbfe1d06e`.
- Selected shared DINO/MVT/Stage1 checkpoint (resolved blob):
  `/home/disk1/LoMa/outputs/loretta_stage1_h/cache/huggingface/hub/models--BRoss123--LoRetta/blobs/09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`;
  1,294,957,584 bytes; SHA256
  `09a502056b671d4e07819f454f96eb385ebe605a186ee1b059ce2447d8fb602e`.
- LoMa-B VGG/DeDoDe checkpoint:
  `/root/.cache/torch/hub/checkpoints/loma_B.pt`; 757,888,113 bytes; SHA256
  `3a38824391e22b33bb3e10377c7736243fd8a2fbf1561446e7e133e497c35758`.
- Data root: `/home/disk1/Data/datasets/GoogleEarth_scale_pairs`.
  Train/val/test contain 36,000/2,500/1,000 manifest rows and all referenced
  images/labels exist.  Manifest SHA256 values are respectively
  `3eed3d3d425caf8465afb5121dfd85960f00932fccc343f37187e14ce27d499c`,
  `a5e6f58bc46cef2dcc49db4e859e2a845755d6aee13be88c998f14a0386b5503`,
  and `3db0eca8c64a23cb1000d6905cbcf82dd221f653033e3b8463cd9484601dfb06`.

Environment (`conda` env `loma-repro`): Python 3.10.20, PyTorch
2.11.0+cu128, torchvision 0.26.0+cu128, CUDA 12.8, cuDNN 91900, NumPy
2.2.6, SciPy 1.15.3, OpenCV 4.13.0, Kornia 0.8.2, Einops 0.8.2.  Server has
four NVIDIA RTX 4090 GPUs (24,564 MiB each), driver 565.57.01.  `pytest` and
`openpyxl` are absent; the repository therefore uses the standard-library
`unittest` runner.  Direct `timm` import currently fails because a transitive
wandb path references removed NumPy 2 `np.float_`; MHINet does not depend on
that import path.

Legacy-wrapper audit:

- old full matcher `train()` forces eval;
- old pyramid extraction is wrapped in `inference_mode`;
- old MVT contextualization and correlation sampling use `no_grad`;
- the Stage1 head forward itself accepts input gradients;
- the old fitter can solve a whole mixed batch before masking an invalid item.

The new provider reuses lower-level compatible modules, not those inference
wrappers.  DINO alone stays under no-grad.  The frozen Stage1 head executes in
autograd, and the safe fitter isolates eligible samples before any
differentiable solve.

Live alignment command:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n loma-repro python -m mhinet.alignment \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p0_p1_alignment.json --overwrite
```

Evidence: `/home/disk1/MHINet/artifacts/p0_p1_alignment.json`, SHA256
`fa1acf5ea8039b478c6a139a026c965b986d17ed06e461f01b613e227e0c1505`.
On pair `000_pair0`, safe/new H0, D8 and D2 are exactly equal to the legacy
legal path; Stage1 support is 1,491 in both.  Shapes are H0 `1x3x3`, context
`1x2x49x49x1024`, and D8/D4/D2/D1
`1x2x256x{98,196,392,784}x{98,196,392,784}`.  DINO, MVT, VGG and the DeDoDe
full pyramid each ran exactly once.  The one-off unwarmed 530.2 ms and
3,416,032,768-byte peak are engineering observations, not a latency claim.

Grouping audit: `/home/disk1/MHINet/artifacts/grouping_audit.json`, SHA256
`5f6483972ead7897945982b67b57bd1a3c2346ef8a60fe57539ed1c4bacec8a4`.
Parent groups are disjoint.  A 0.01-degree geographic audit found one
train/val cell collision; the deterministic policy holds val fixed and removes
four affected train pairs (35,996 safe train pairs).  Test is sealed and is not
used for exclusions or selection.

## P1/P2 — feature ownership, geometry and correlation: passed at unit/reference level

Implemented shared provider, trainable adapters, align-corners-false coordinate
conversions, TL/TR/BL/BR h33=1 four-point DLT, pre-solve safety isolation,
9x9 denominator checks, H-guided local correlation, dy-outer/dx-inner
candidate ordering, sampled-feature renormalization and chunking.  H, T,
sampling grids and feature tensors are not detached in the production path.

Reference tests include identity/translation/perspective, FP64 autograd
gradcheck, a mocked assertion that `solve_ex` never sees the singular member of
a mixed batch, direct-loop correlation equality, chunked/un-chunked output and
gradient equality, and activation-checkpoint output/gradient equality.

Real D8 correlation diagnostic:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n loma-repro python \
  -m mhinet.real_correlation_audit \
  --runtime configs/runtime_paths.server.json --pair-index 0 --scale 8 \
  --seed 0 --output artifacts/p2_real_correlation_d8.json --overwrite
```

On 4,726 observable queries, a random 64D adapter places the nearest exact-H
integer candidate top-1 for 69.11% of queries and its argmax is within 1.5 D8
pixels for 95.18%; raw 256D values are 78.16% and 98.88%.  This is a fixed
feature correlation-signal diagnostic, not validation accuracy.

## P3 — exact parameter count and zero-init full forward: passed

Command:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n loma-repro python -m mhinet.engineering \
  --runtime configs/runtime_paths.server.json --pair-index 0 \
  --output artifacts/p3_zero_init_smoke.json --overwrite
```

Evidence SHA256:
`2c003cd950bffafc0d0f8b8fbd904924dbe5ba490d070b48ed98a036d32fc7b0`.
All four FC2 weight/bias tensors contain zero nonzero elements; exact new
parameter count is 2,544,160.  A real eight-update forward accepted all eight
updates and changed H0 corners by at most `6.103515625e-05` input px
(`<1e-3`).  Shared components each ran once.  Peak allocated memory was
3,426,232,320 bytes.  The unwarmed 5,833.6 ms includes a contended server and
is not a latency benchmark.

Failures found and fixed before this pass:

1. A first run on physical GPU 3 failed before iteration because unrelated
   processes left insufficient memory for a 602 MiB D1 allocation.  The target
   was moved to a GPU with verified free memory; no result was fabricated.
2. The next run exposed BF16 pyramid vs FP32 adapter weights outside autocast.
   Adapter/decoder CNN boundaries now use CUDA BF16 autocast while correlation,
   DLT and loss geometry stay FP32.
3. A later fail-fast adapter finite check attempted a scalar read on a meta
   tensor.  It now runs on materialized CPU/GPU tensors and the full-shape meta
   contract test remains valid.

## P4 — loss/trainer/gradient engineering: passed; tiny gate unresolved

Default loss is the uniform mean of the eight pre-guard proposal-corner
coordinate L1 values.  FGO, overlap, auxiliary correlation and Planar heads are
off.  An all-invalid batch produces a finite graph scalar and explicitly asks
the trainer to skip optimizer and scheduler.  Atomic checkpoints contain model,
optimizer, optional scheduler/scaler, optimizer/microbatch/data progress and
Python/NumPy/Torch CPU/CUDA RNG state; restricted `weights_only=True` loading is
tested.

Gradient-audit command:

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n loma-repro python -m mhinet.gradient_audit \
  --runtime configs/runtime_paths.server.json --pair-index 0 \
  --output artifacts/p4_gradient_audit.json --overwrite
```

Evidence SHA256:
`8501fb9ed2dd9f4e837d2c85ad91966625d8e0a6c56526197da1a8bebb85c799`.
On the first zero-init step, every FC2 has finite nonzero gradient while all
four adapters and decoder-upstream groups have exactly zero gradient.  After
one optimizer update, all those upstream groups have finite nonzero gradients.
A final-H-only loss reaches all four scales.  MVT has finite nonzero gradients
independently through H0 and through the pyramid; VGG and DeDoDe have finite
nonzero pyramid-route gradients.  DINO and Stage1-head parameter gradients are
all `None`.  Peak allocation was 4,630,508,032 bytes for cached-feature eight
round backward and 5,493,430,272 bytes for the real joint branch diagnostics.

Full 784 training profiles (two optimizer steps, one pair per step):

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -m mhinet.profile --runtime configs/runtime_paths.server.json \
  --profile heads --optimizer-steps 2 \
  --output artifacts/p4_profile_heads_784.json --overwrite

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n loma-repro \
  python -m mhinet.profile --runtime configs/runtime_paths.server.json \
  --profile joint --optimizer-steps 2 \
  --output artifacts/p4_profile_joint_784.json --overwrite
```

- `heads`: passed; artifact SHA256
  `c71d52d313a396c5072bc3f1ba82daba8365c86b4b846c3ac594c8616cc21932`;
  peak allocated/reserved 5,593,135,616/5,796,528,128 bytes.  Step 0
  forward/backward was 3.0465/2.8476 s and step 1 was 1.9259/2.6159 s.
- `joint`: passed on one 24 GiB RTX 4090; artifact SHA256
  `abfcc2a454a18c05d1dfef886f0b4b252889c377434e25c8cecbb0dd11cdc0f7`;
  peak allocated/reserved 14,186,542,080/14,512,291,840 bytes.  Step 0
  forward/backward was 2.4575/3.2071 s and step 1 was 2.2304/2.8643 s.
  At zero init, DeDoDe/VGG gradients were zero as expected; after the first
  optimizer update both were finite and nonzero.  MVT was already nonzero at
  step 0 via H0.  DINO and Stage1-head gradients remained absent.  Shared
  DINO/MVT/VGG/full-pyramid calls were one per step in both profiles.

After D2 timing exposed excessive Python overhead from constructing one
activation-checkpoint context per 1,024-query chunk, correlation was refactored
to checkpoint the complete chunk loop once.  Sampling still occurs in the same
1,024-query chunks, with identical ordering and no detach; the existing direct,
chunked/un-chunked and checkpointed output/gradient tests all pass.  Fresh
two-step 784 profiles for the current implementation both passed:

The current path was first exercised by a 32-residual, one-step D2 integration
with finite backward and zero rejected updates (expected `failed` solely due to
the one-step budget); peak allocated/reserved memory was
2,761,892,352/3,011,510,272 bytes.  Artifact SHA256
`805142ff226a1047aa1f9bf7d17de713e912adafaf1e111abc13bfc458281a87`.

- `heads`: peak allocated/reserved
  8,731,754,496/9,053,405,184 bytes; artifact SHA256
  `c6657331a21076f54da91dfbe7374214d105827afc4a41a941ba1d182c96a37b`.
- `joint`: peak allocated/reserved
  17,336,022,016/17,792,237,568 bytes on the 24 GiB RTX 4090; artifact SHA256
  `7dac145bf10f0e8d7bd582ddf2b96c6915f708bbb54cdf3bd253aca4b59ef9c6`.
  At step 1, DeDoDe, VGG and MVT gradients were all finite and nonzero; DINO
  and Stage1-head parameter gradients remained absent, and shared call counts
  remained one.  The optimization trades about 3.1 GiB extra peak allocation
  in `joint` for lower checkpoint-management overhead and remains below the
  measured device limit; both the old and current profiles are retained.

Minimal trainer smoke (D8, two rounds, two optimizer steps, B1 x accumulation
4) used `configs/train_minimal_smoke.json`, SHA256
`6946de077ed11b5f3a5dcf495003fe073a7c7e424815597b1cd716a66f0239c4`:

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n loma-repro \
  python -u -m mhinet.train --runtime configs/runtime_paths.server.json \
  --config configs/train_minimal_smoke.json \
  --output-dir outputs/mhinet_minimal_smoke --overwrite
```

It completed two steps with zero invalid attempts and zero shared-call
violations.  The losses were 4.722061 and 2.598176 px.  Its checkpoint is
`/home/disk1/MHINet/outputs/mhinet_minimal_smoke/checkpoints/step_0000002.pt`,
906,932,089 bytes, SHA256
`fb3f0bdd23c9b8807965ddee7471e13d0b4d234f7f8094946e80a83831c75a75`.
The two-pair validation smoke had H0/final mean MACE 6.61187/6.65973 px;
this is explicitly an entrypoint check, not an accuracy pass.  Validation
summary SHA256:
`113d0c2abdde39dadcabee5dc1f4e7fe6a008b53fd33f4a75db5ecbd9a39be9f`.

Pause/resume was then exercised at an optimizer boundary with
`--stop-after-optimizer-step 1` and the same config.  Step 1 saved data
position 4; resume restored model, optimizer, scheduler, CPU RNG and CUDA RNG,
then completed step 2 at data position 8.  The step-1 and resumed step-2
checkpoint hashes are respectively
`5a17a0d4582b478fbe22c3081044ae2a08aab69c5ac68bcae72618cb5b2155f3`
and `5b339a7a8f81c77b2998ce078ce79dcc6d3a0d68f6f817ea5a1ccb3ee534021a`.
Compared with the uninterrupted checkpoint, 963/964 model tensors were
bitwise equal; the D8 adapter differed in 134 elements with maximum absolute
difference `8.298084e-7`, consistent with the CUDA warnings for nondeterministic
grid/adaptive-pool backward.  Progress and scheduler state were exact.
Evidence: `/home/disk1/MHINet/artifacts/p4_resume_smoke.json`, SHA256
`d9ae16eb2d7a6e8125f76115efaee9ce327c0cb7384af46d1b7c7e69879e1a33`.

Current test command and result:

```bash
conda run --no-capture-output -n loma-repro \
  python -m unittest discover -s tests -v
```

Result at this point: 57 tests passed.  `compileall`, every CLI `--help`
entry, and the 31-file design-package verifier also passed.  The regression
suite now explicitly checks that an isolated non-finite decoder output is
reported instead of being hidden by its guarded zero placeholder, and that
the repeated-pair GPU-preload path can execute both raw and parameter-averaged
tiny evaluations with the correct call contract.

### Tiny-overfit attempts

The implemented CLI uses B=1, only new layers, AdamW lr `1e-3`, weight decay
zero, no scheduler, gradient-norm clip 1, uniform proposal L1 and at most 2,000
steps.  It records H0/every update/final MACE, rejection/failure rates, peak
memory, per-pair inference time and atomic checkpoint SHA256.  Controlled H0 is
diagnostic-only and is never used by formal training.

- A 2-pair/2-step integration smoke completed all data, shared-feature,
  backward and checkpoint paths.  It is deliberately marked failed because it
  is not the required 32-sample test.  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_integration_smoke.json`.
- The first 32-distinct-pair D8 run was stopped at step 704 after discovering
  the missing protocol gradient clip; its 5.16 px observation is not a final
  result.
- The corrected 32-distinct-pair D8 stress run (clip 1, controlled residuals
  about 5.6–8 input px) ran all 2,000 steps.  Final mean MACE was 1.431142 px
  with zero rejected updates: a real improvement but a failed `<0.1 px` gate.
  Artifact: `/home/disk1/MHINet/artifacts/p4_tiny_s_d8.json`; checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d8_seed0.pt`, SHA256
  `1f974ab3f9d52e8cb5d015e69420470305f731320b2e03185b737d988acdac84`.
- A one-exact-pair/32-H0-residual diagnostic using the same small D8 residual
  regime collapsed near 6.30 px through step 832 and was stopped for diagnosis.
  The real correlation audit showed that the small residual occupied only
  about 0.63 D8 pixels and the nearest-vs-center correlation margin averaged
  0.015.  A rerun now uses 0.7–1.0 of the declared half-bound (still no more
  than half of the decoder update bound), or about 1.4–2 D8 pixels.
- The corrected locally observable translation run used one exact train image
  pair and 32 deterministic H0 residuals, B1 x accumulation4, clip 1 and the
  required fixed optimizer recipe.  It completed 2,000 steps with zero failed
  pairs and zero rejected updates, but raw final mean/median/P90 MACE was
  0.190075/0.184014/0.243916 px, so it is a failed `<0.1` gate rather than a
  pass.  Artifact SHA256:
  `33f885db0d08cc85c518c03f2adfd5b83e03ed6ff9bdc62daaf4ef72e3cf0df4`;
  checkpoint SHA256:
  `43d71164b68ead7c94b4bb5ed4eeffc8c804420c28241af5844d8d969c82f576`.
- Failure isolation found no guard/window failure and no BF16 floor: the same
  checkpoint measured 0.190075 px with BF16 CNN autocast and 0.195496 px with
  FP32 CNN execution.  A single fixed residual crossed the threshold at step
  440 (0.084624 px), proving that the differentiable path can reach the target
  but not constituting the required 32-sample pass.  A deliberately
  off-protocol tenfold-lower-lr continuation reached 0.077395 px in 20 steps;
  it is diagnosis only because it exceeds the budget and changes the fixed lr.
  Averaging 33 adjacent endpoint weights reached 0.087943 px and isolated the
  issue as constant-lr L1 endpoint oscillation.  Full evidence and all
  non-gate labels are in
  `/home/disk1/MHINet/artifacts/p4_tiny_d8_failure_analysis.json`, SHA256
  `ba54bb1cab083cb2e40aadb2eff1fb07725d6393e723152d19b81b027648bc2b`.
- A fresh D8 run kept B1, accumulation4, lr `1e-3`, no scheduler, uniform L1,
  clip 1, BF16 CNN/FP32 geometry, and only-new-layer training.  It predeclared
  a tiny-only equal-weight parameter readout from step 1,536 and crossed the
  threshold at step 1,728 using 193 in-budget snapshots: mean/median/P90 final
  MACE 0.085699/0.074983/0.157325 px, zero failed pairs and zero rejected
  updates.  The simultaneous raw endpoint was 0.459274 px, so this is recorded
  specifically as an **averaged-readout pass; raw endpoint did not pass**.
  The averaging is disabled in formal training and the diagnostic checkpoint
  is marked non-resumable.  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d8_translation_swa.json`, 208,644
  bytes, SHA256
  `719852b5af01aa446cb8850a0014bc5ae82632e0582fcffc0a3c0193aee6034d`.
  Checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d8_one-pair-residuals_translation_bf16_weight-average-from-1536_seed0.pt`,
  14,482,191 bytes, SHA256
  `dc7360186b688c245909c5b0da01abf491f17234313754a0ea422b0c8cc1af98`.
- TINY-S-D4 used the same registered recipe.  It passed at optimizer step
  1,568 with 33 in-budget averaged snapshots: mean/median/P90 final MACE
  0.069876/0.065841/0.091630 px, zero failed pairs and zero rejected updates.
  The simultaneous raw endpoint was 0.126651 px, so this is again an
  **averaged-readout pass; raw endpoint did not pass**.  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d4_translation_swa.json`, 188,466
  bytes, SHA256
  `b5387f294019e9a3700cda147d04305eec93a770e55a69f78e42c3fea303df13`.
  Checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d4_one-pair-residuals_translation_bf16_weight-average-from-1536_seed0.pt`,
  15,112,741 bytes, SHA256
  `9ff3d1f54fa0ef657907658d63baad70218aaa85489db9b9f5003a120eeb7650`.
- Before starting D2, a fresh real-resource one-step D4 integration exercised
  the newer repeated-pair single-GPU-copy path and both raw and averaged
  evaluations.  All forward/backward/checkpoint branches completed with zero
  rejected updates.  It is deliberately `failed` because it contains only two
  diagnostic residuals and one optimizer step, not because of a runtime error.
  Artifact SHA256:
  `997f739826d1f747d3f210d93e2ffb35ecf63c0d0e5e0b4db50a8ecfd409bbc7`.
- TINY-S-D2 passed at optimizer step 416 using the raw parameter readout,
  before the predeclared averaging window: mean/median/P90 final MACE
  0.095326/0.100922/0.124073 px, zero failed pairs and zero rejected updates.
  Artifact:
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d2_translation_swa.json`, 107,374
  bytes, SHA256
  `78cb079f66d69f13719f9a79627fe90aaa86b0c29cf8cde98999066015e82f73`.
  Checkpoint:
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d2_one-pair-residuals_translation_bf16_weight-average-from-1536_seed0.pt`,
  15,644,987 bytes, SHA256
  `7ea700fb910714bd631f72e7908ba7f3ce1f0c866ac3742b272975aaa45f2c9b`.
- A two-residual, one-step D1 integration then exercised the current grouped
  checkpoint implementation at full 784 resolution.  Forward, backward, raw
  evaluation, averaged evaluation and checkpoint save all completed with
  finite gradients, zero failed pairs and zero rejected updates.  Peak
  allocated/reserved training memory was
  6,697,497,088/7,021,264,896 bytes.  Its `failed` status is expected because
  it is not the 32-residual learnability run.  Artifact SHA256
  `2a3f52d3e395e49ba2ebc53dad253c612b8417a42775a52238d81af32f9cb5e3`.
  TINY-S-D1 is now the active next gate; TINY-8 has not been run.
- The independent-run merger was exercised with only D8.  It failed closed
  and listed D4/D2/D1/TINY-8 as missing; formal E01 then rejected that artifact
  before model construction or any optimizer step.  Partial artifact SHA256:
  `0680e3cb6015fc17807aede502ed5b285617a45f514e70f31e8b9fa108a257e4`.

CUDA warns that `grid_sampler_2d_backward_cuda` and
`adaptive_avg_pool2d_backward_cuda` have no deterministic implementation.
Seeds, sample IDs and RNG states are recorded, but current CUDA results should
not be described as bitwise deterministic.

## P5/P6 status

Not started.  E00/E01 and later experiments remain blocked by the incomplete
TINY-S/TINY-8 learnability gate.  No test item has been evaluated and no model
validation result has been claimed.
