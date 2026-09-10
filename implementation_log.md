# MHINet implementation log

This is an evidence log, not a claim that the model has been validated.  The
active requirements are the server handoff package training protocol v1.2 and
loss revision 1.1.  Files under `history/` were not used as implementation
instructions.

Canonical model terminology from this point forward is GHIM (global
homography initialization, producing `H0` and `F_MVT`), CGMDP (MVT-guided
multi-scale descriptor pyramid, producing the fused matching descriptors
`D8/D4/D2` on the current mainline) and MHIR (six-update homography refinement,
producing `H_final = H6`). D1 code remains registered but inactive and is not
decoded, optimized, or executed by default. Historic `stage1_*` names below
quote legacy code, checkpoints or artifacts and remain compatibility aliases;
they do not denote an additional functional module.

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

Result at this point: 65 tests passed.  `compileall`, every CLI `--help`
entry, and the 31-file design-package verifier also passed.  The regression
suite now explicitly checks that an isolated non-finite decoder output is
reported instead of being hidden by its guarded zero placeholder, and that
the repeated-pair GPU-preload path can execute both raw and parameter-averaged
tiny evaluations with the correct call contract.  It also asserts that all
1,024-query chunks use one activation-checkpoint context and that optimizer
protocol drift makes the tiny gate fail closed.

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
  TINY-8 has not been run.
- A real-feature signal comparison was added before interpreting the slow D1
  curve.  With the registered half-bound D1 residual, the random-adapter
  nearest-vs-centre correlation margin was only `0.002778` on average, versus
  `0.016658` at D2.  Using the full legal D1 decoder bound increased that D1
  margin to `0.008696`; this is diagnosis only and the active run still uses
  the registered half-bound residual.  Half-bound D1, full-bound D1 and
  half-bound D2 artifact SHA256 values are respectively
  `f8f6e3ebaee6537e213ab544c49d0d1dc67310771e1a4db868f99acded7c1751`,
  `5a6d85493dcd28399008da0efc50cd05b4c0ff118e319f96f65799d92ecc1c67`
  and `72c8c74a108e6f1b548d403b76a595ae99b3f5b41739cd9693afe0a239800a9e`.
- The correlation audit now also emulates the decoder-boundary conversion
  `FP32 -> BF16 -> FP32` without changing production forward precision.  For
  the random 32-channel adapter, half-bound D1 had a mean valid-score
  quantization error of `0.000966`, `30.23%` nearest/centre ties after
  conversion, only `73.12%` retention of originally positive margins,
  `69.77%` margin-sign preservation and `59.16%` argmax preservation.  Full-
  bound D1 improved those values to `16.53%`, `87.13%`, `83.47%` and `60.10%`;
  half-bound D2 was materially stronger at `7.91%`, `94.47%`, `92.09%` and
  `83.47%`.  The raw 256-channel descriptor shows the same scale ordering.
  This makes BF16 loss of small D1 contrast a plausible contributor, not a
  proven sole cause; the separately running FP32 training comparison is the
  causal check.  Updated D1-half/D1-full/D2-half artifact SHA256 values are
  `2166fe976a49b801ff4e581d2d498c4d4206eaa65905ac012334832c51aafff3`,
  `9a0dd15088da2b4c82dff3b78ffa73cd5515f631877fc24e3d3f77a1c19b23a2`
  and `8e9812521051d3c79cb8a13721aebcc2ca07471c7ae3b1b676a20ee2635f579e`.
  Two pure-CPU edge/reference tests cover ties, rank loss, invalid shapes and
  empty valid rows; the complete suite passed 69/69 in 7.794 seconds.
- D1 profiling showed that 614,656 queries create 601 fixed 1,024-query
  sampling chunks.  Correlation now samples the 32-channel target and its
  finite mask in one 33-channel `grid_sample`, and constructs the complete
  candidate grid once inside the same non-reentrant checkpoint before slicing
  it into the unchanged row-major chunks.  No precision, ordering, detach or
  gradient rule changed.  The existing output/gradient/checkpoint reference
  suite passed.  A real two-residual D1 one-step smoke reduced elapsed time
  from `18.1092 s` to `14.9944 s` (17.2%) while peak allocated/reserved memory
  rose from 6,697,497,088/7,021,264,896 to
  7,141,153,280/7,409,238,016 bytes; gradients stayed finite and update
  rejection stayed zero.  The final optimization artifact SHA256 is
  `cd47126d9b3faa97542ddef6187981da10eb64a4a97ddeac494b66314e0e914b`.
  An intermediate one-sampler-only smoke is retained with SHA256
  `92c45486e8a9642691b1abff94ac84c2e9a6bfb6a689af1dc9277573ec11d43b`.
- The registered dense D1 5x5 run used the pre-optimization process and reached
  step 864 before its hosting exec session ended without a Python traceback,
  OOM record, final artifact or checkpoint.  It therefore remains an
  interrupted **non-pass**, not a completed 2,000-step result.  Mean final MACE
  fell from `0.927098 px` at step 32 to `0.474308 px` at step 160, then
  plateaued: the last ten 32-step observations averaged `0.4606961 px` in the
  range `[0.457148, 0.466198]`, with zero rejected updates throughout.  The
  raw log SHA256 is
  `92661c88c1d18d5efe45c9749e20afb52b9cf0380ba4ae1ac3bcf4e6a22fc81f`;
  the preserved observation artifact SHA256 is
  `dbfa8521480c0939e4619f1b6ceca7a066dcae28ebb496d30e4f5a24ccf0fa5d`.
  This is an initial dense full-grid 5x5 failure diagnosis, **not** the formal
  baseline for sparse-query D1 and 3x3-window D1 comparisons: it has no final
  checkpoint/artifact and cannot support a same-checkpoint paired comparison.
  The formal `D1-DENSE-5` reference remains unestablished and must be produced
  only after P4, E00/E01, E02--E05 and main-configuration selection.  Neither
  variant may replace the mainline until matched-budget validation shows its
  accuracy, failure-rate, memory and latency trade-off.
- A dense D1 5x5 **diagnostic**, started at Git `d77b487` after the correlation
  optimization, expanded the controlled residual to the full legal D1 decoder
  bound and completed its declared 256-step budget.  Exact command:
  `CUDA_VISIBLE_DEVICES=3 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.tiny_overfit --runtime configs/runtime_paths.server.json --experiments D1 --sample-protocol one_pair_residuals --residual-profile translation --precision bf16 --sample-count 32 --seed 0 --max-steps 256 --eval-interval 32 --threshold-mace-px 0.1 --residual-bound-fraction 1.0 --output artifacts/p4_tiny_s_d1_full_bound_diagnostic_256.json --overwrite`.
  Mean final MACE moved from `1.858603 px` at step 0 to `1.853464`,
  `0.989137`, `0.955050`, `0.929901`, `0.946010`, `0.921074`,
  `0.920057`, and `0.930858 px` at steps 32--256.  The final
  median/P90/max were `0.822484/1.731238/1.863900 px`; H1 and H2 means were
  `0.931836` and `0.930858 px`.  All 32 conditions remained finite with zero
  rejected updates, but the best observed mean was still far above `0.1 px`.
  Peak allocated/reserved memory was
  `7,138,167,296/7,411,335,168` bytes and measured training time was
  `3,241.151 s`.  Artifact SHA256:
  `920a5386fad222c5a648cfadca49f7d107d388109ae468eeac78d3f4d8164353`;
  raw stderr SHA256:
  `9ccb831427b6c37dd5a0b838755770eeee762213350b0b2764d1fad953765cab`;
  16,250,961-byte checkpoint
  `/home/disk1/MHINet/outputs/tiny_overfit/tiny-s-d1_one-pair-residuals_translation_bf16_raw_seed0.pt`
  SHA256 `b33f3b148b58e96f7d5af4bfde1681f56c6a2c1be0dda803a51bd6e97b8dadd4`.
  This is a completed short diagnostic failure, not the registered 2,000-step
  averaged D1 gate.  Increasing residual amplitude alone therefore does not
  resolve D1 learnability and does not justify a blind long continuation.
- The independent-run merger was exercised with only D8.  It failed closed
  and listed D4/D2/D1/TINY-8 as missing; formal E01 then rejected that artifact
  before model construction or any optimizer step.  Partial artifact SHA256:
  `0680e3cb6015fc17807aede502ed5b285617a45f514e70f31e8b9fa108a257e4`.
- The gate validator now checks each experiment's two-updates-per-scale
  schedule, B1/accumulation-4 AdamW recipe, lr/weight decay/clip/scheduler,
  BF16 mode, translation residual protocol, finite gradients, controlled-H0
  provenance and seed.  It no longer trusts a reported checkpoint hash: it
  verifies the on-disk path/bytes/SHA256, restricted-loads the payload, binds
  progress and metadata to the experiment, and checks raw/averaged readout
  ownership.  It also recomputes mean/median/P90/max, failures and rejections
  from all 32 per-condition rows, checks history consistency and requires the
  five independent artifacts to share data/resource evidence.  D8/D4 predate
  two readout flags; only their two independently reloaded immutable checkpoint
  hashes receive an explicit legacy migration exception.  A fresh strict
  partial merge accepts D8/D4/D2 and fails only for the legitimately missing
  D1 and TINY-8 experiments.  Partial artifact SHA256:
  `99ff441f1f00b005b0c301dbf899161c9ccdaf9dcd3769d86fb058bcc8901f3b`.
- Long TINY-S/TINY-8 runs now have an independent atomic progress checkpoint
  path and strict `--resume-progress` entrypoint.  The progress role always
  stores the raw iterator and matching AdamW state at a completed optimizer
  boundary; it also stores the exact data cursor, Python/NumPy/Torch/CUDA RNG,
  compact metric history, original step-0 endpoint, accumulated timing/peak
  memory and the FP32 equal-weight parameter averager.  Averager tensors live
  only in the binary `auxiliary_state` and are deliberately omitted from the
  JSON-facing save report.  A restricted `weights_only=True` preflight runs
  before live state or an existing partial JSON is changed.  Resume fails on a
  mismatch in experiment/scales, seed, budget, evaluation interval, threshold,
  sample protocol/order, residual profile/bound, precision, averaging window,
  optimizer recipe, target size, architecture, manifest, runtime config or
  DINO/GHIM/pyramid checkpoint path/size/SHA256.  Heartbeats default to every
  eight optimizer steps and perform no extra forward pass; an evaluation step
  and heartbeat are merged into one save.  The same visible-CUDA-device
  topology is required because all visible CUDA RNG streams are restored.
- The final evidence checkpoint filename now includes residual-bound fraction
  and optimizer budget, preventing a short diagnostic from overwriting a
  registered 2,000-step result.  A resumed checkpoint that had already passed
  at its current evaluated step finalizes without another optimizer update.
  The redundant per-parameter CUDA finite-gradient synchronization was removed;
  `clip_grad_norm_(error_if_nonfinite=True)` remains the single fail-closed
  finite-norm check before every optimizer step.
- A CPU interruption-equivalence regression stops immediately after the
  step-1 heartbeat, reloads the restricted progress payload, and reaches step 3
  with bit-exact model and AdamW tensor states relative to an uninterrupted
  run, including the parameter-average window.  Name mismatch and non-finite
  averager state are rejected.  Command:
  `CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -v`.
  Result: all 67 tests passed in 2.741 seconds.  This validates checkpoint
  mechanics only; it is not a D1/TINY-8 learnability or model-validation pass.
- A real CUDA smoke at Git `1ccedfae8574cf01eb398ad00b12af8e4f7ef610`
  then ran D8 for one optimizer step on two controlled conditions, wrote a
  14,488,399-byte raw progress checkpoint, and resumed it with zero additional
  optimizer steps.  The restored report confirms model, AdamW, Torch CPU RNG
  and CUDA RNG restoration; history remained `[0, 1]`, MACE remained exactly
  `11.89400863647461 -> 2.285935878753662 px`, and failures/rejections remained
  zero.  Both processes correctly returned status `failed` because two
  conditions and one step are not the registered gate.  The progress checkpoint
  is `/home/disk1/MHINet/outputs/tiny_overfit/resume_smoke_1ccedfa_progress.pt`,
  SHA256 `c7419803b9a89baa48cb258960de2c57594965677240bc87067f5afcef9c6be2`;
  the post-resume final checkpoint SHA256 is
  `672709c64d459f2e177a75f500fbc6f64f8ca3c130637cd9de03fe0b6d7ad92e`.
  Exact commands, environment versions, output paths, resource hashes and
  expected-failure scope are in
  `/home/disk1/MHINet/artifacts/p4_tiny_resume_cuda_smoke.json`, SHA256
  `f588599ffaef020a8f6a4f0ab21d5932bc35612baf0619d3af49b7ecdd371058`.
- Added the read-only `tiny-checkpoint-audit` entrypoint for diagnosing a
  serialized raw/averaged endpoint without resuming training.  It snapshots a
  possibly live atomic progress file before restricted preflight, rejects
  formal/unknown checkpoint roles and protocol conflicts, requires explicit
  supplements for missing legacy seed/bound fields, reconstructs the exact
  train-only controlled conditions, and strictly loads only the MHIR iterator
  with `restore_rng=False` and no optimizer.  It records H0, every H, both
  per-scale decoder deltas, support/condition/solve/saturation fields and
  canonical serialized/load/post-forward state hashes.  Its status is
  deliberately `diagnostic_complete`, never `passed`.
- Tiny endpoint serialization now restores the iterator's original train/eval
  mode even when projection diagnosis raises, records per-update corner
  residuals and emits strict JSON (`null` for non-finite diagnostics rather
  than non-standard `Infinity`/`NaN`).  Command:
  `/root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -v`.
  Result: all 77 tests passed in 1.576 seconds, including direct rejected-update,
  invalid-H0 restoration, protocol-conflict, legacy-supplement and exact-state-
  digest regressions.  This is checkpoint/diagnostic engineering only; no new
  D1 learnability or validation result is claimed.
- Checkpoint delta summaries now keep x/y coordinates separate and report the
  fraction of controlled conditions improved by each update, preventing an
  easy horizontal correction from hiding a vertical failure in one aggregate.
  The real-correlation audit accepts an explicit `--condition-index` so the
  exact horizontal- or vertical-dominant H0 conditions from tiny-overfit can be
  reproduced.  Full-suite command unchanged; all 79 tests passed in 1.824
  seconds.  No production forward, loss, precision or training rule changed.
- Added the diagnostic-only `--condition-index-offset` to tiny-overfit so a
  vertical-dominant condition can be trained independently instead of silently
  reusing condition 0.  A nonzero offset is bound into progress/final metadata
  and the final checkpoint filename; the registered default zero deliberately
  omits the new signature key, preserving exact resume compatibility with the
  already-running jobs.  Checkpoint audit reconstructs the recorded offset.
  The cache description now reports its actual sample count instead of always
  saying 32.  Full suite: 80/80 passed in 1.796 seconds.  This option is failure
  isolation only and cannot satisfy the 32-condition P4 gate.
- The completed dense-D1 BF16 step-256 checkpoint was reloaded read-only at Git
  `018fb0a` using:
  `CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli tiny-checkpoint-audit --runtime configs/runtime_paths.server.json --checkpoint outputs/tiny_overfit/tiny-s-d1_one-pair-residuals_translation_bf16_raw_seed0.pt --experiment D1 --sample-count 32 --sample-protocol one_pair_residuals --residual-profile translation --precision bf16 --seed 0 --residual-bound-fraction 1.0 --output artifacts/p4_tiny_s_d1_full_bound_bf16_checkpoint_audit.json --overwrite`.
  The 16,250,961-byte snapshot SHA256 is the expected
  `b33f3b148b58e96f7d5af4bfde1681f56c6a2c1be0dda803a51bd6e97b8dadd4`;
  canonical serialized, loaded and post-forward iterator-state hashes all equal
  `b0ae7b605211a74ebeaf64eb73d4f0f316a0d92c69173f2f878b0fa889c5a419`.
  No optimizer/RNG/backward/step was used, the source inode did not change while
  snapshotting, and DINO/MVT/VGG/CGMDP were each evaluated once for the one
  exact pair.  Artifact: 178,948 bytes, SHA256
  `3c9fde2cc1141dcb68afbb64ac7211c24b4ddafa6137d877622b7e6f97b8569a`;
  stderr was empty.  This legacy checkpoint lacks embedded resource hashes and
  seed/bound, so the artifact explicitly marks those two CLI supplements and
  does not claim strict historical resource binding.
- That audit reproduces H0/H1/H2 mean MACE
  `1.858605 -> 0.931836 -> 0.930858 px`, with no failure/rejection or saturation
  and well-conditioned DLT (`condition_number` mean about `3.438`).  It exposes
  a directional failure: update 1 reduces mean absolute x residual from
  `1.432011` to `0.083152 px`, but y changes from `0.915114` to
  `0.920478 px`; update 2 leaves x/y at `0.120899/0.914302 px` and improves only
  10/32 conditions.  Thus guards, saturation and a hidden state mutation are
  ruled out for this checkpoint; the decoder learned mostly horizontal
  correction and its shared second application added no aggregate benefit.
- Real D1 correlation was then audited on the exact x-dominant condition 0 and
  y-dominant conditions 2/14 with full legal residual bound.  Commands used
  `python -m mhinet.cli real-correlation-audit --runtime
  configs/runtime_paths.server.json --pair-index 0 --scale 1 --seed 0
  --condition-index {0,2,14} --residual-bound-fraction 1.0` on GPU2.  Artifact
  SHA256 values are respectively
  `7d4cdc3ee38e4ba2732d5fe63bf63a63b1a1703eec7e35f1c4fcc7fba530950c`,
  `9a2ad84fd78a764ce95bfa8a6cf4e8c5342fb0f3a78b27ca4c3f868e0270a69d`
  and `08d29c365b8160896303533ce90a35c4b6f5111b869a39875bfe23876d26c600`;
  all stderr files were empty.  Random-adapter nearest-candidate top-1 rates
  were `0.2248/0.2946/0.2392`, while raw-descriptor rates were
  `0.4071/0.5013/0.4521`; condition 2's y-positive correlation margin
  (`0.01221`) was stronger than condition 0's x-dominant margin (`0.00870`).
  Vertical evidence is therefore present before the decoder; this narrows, but
  does not yet prove, the failure to learned representation/optimization or
  multi-condition interference.
- Two condition-0 one-sample D1 diagnostics launched at Git `e39d187` completed
  128 raw steps in BF16 and FP32.  Exact commands matched the full-bound D1
  recipe with `--sample-count 1 --max-steps 128 --eval-interval 16`, precision
  `bf16` or `fp32`, and their separately recorded progress/output paths.  BF16
  moved `1.418984 -> 0.065078 px` (best sampled endpoint `0.012468`); FP32 moved
  `1.418984 -> 0.047024 px` (best `0.045429`).  Both had zero failure/rejection.
  Their top-level status correctly remains `failed`, because one condition can
  never meet the registered 32-condition criterion.  Artifact SHA256 values:
  `0126f91ca908226dd34df7dcea86fb08223dd28102ee1079566a98cc9c0fe8fa`
  and `9a0c5ab3934f9ae96abc6854ab2cb55b2a0826fddf7a03c3c4b9b0998383e00a`.
  Final checkpoint SHA256 values:
  `e2770fbc012d7d186940385f8952bc7b14e3ae3033dd3cfe4b651947e54cca2d`
  and `a01ab71d2745710beaa5e7c2875ca3d855f60a6b38558e74427a434d4210b845`.
  BF16/FP32 training elapsed was `1553.745/1498.777 s`; peak allocated/reserved
  memory was `7,138,167,296/7,411,335,168` and
  `7,921,545,728/8,204,058,624` bytes.  This proves only that the x-dominant
  condition is individually learnable in either precision, not that D1 or P4
  passes.  Separate y-positive/y-negative one-condition jobs are in progress.
- The full 32-condition D1 FP32 diagnostic launched from Git `e39d187`
  completed all 256 optimizer steps.  Exact command:
  `CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m
  mhinet.tiny_overfit --runtime configs/runtime_paths.server.json
  --experiments D1 --sample-protocol one_pair_residuals --residual-profile
  translation --precision fp32 --sample-count 32 --seed 0 --max-steps 256
  --eval-interval 32 --threshold-mace-px 0.1 --residual-bound-fraction 1.0
  --heartbeat-interval 8 --progress-checkpoint
  outputs/tiny_overfit/d1_full_bound_fp32_256_progress.pt --output
  artifacts/p4_tiny_s_d1_full_bound_fp32_diagnostic_256.json --overwrite`.
  Its H0/H1/H2 mean MACE was `1.858605 -> 0.948720 -> 0.929889 px`;
  the best sampled endpoint was `0.918012 px` at step 160.  It had zero
  failures/rejections but missed the registered `<0.1 px` threshold.  Training
  took `3310.370 s`, with peak allocated/reserved CUDA memory
  `7,921,545,728/8,204,058,624` bytes.  Artifact: 101,431 bytes, SHA256
  `8345c778c5a8df086b62131dd1f2d2e6b6db49a7a16214347359457a787fb405`;
  final checkpoint: 16,252,945 bytes, SHA256
  `6b5820e86c370cac9c8320ce1232230186a41a8bfd8d8b6c422ac66d10be729d`.
  FP32 therefore does not explain or fix the 32-condition D1 plateau.
- The condition-2 y-positive and condition-14 y-negative one-sample BF16
  diagnostics launched from Git `b50ef2e` also completed 128 optimizer steps.
  They used the same D1 full-bound command as the earlier one-condition runs,
  with `--condition-index-offset 2` on GPU2 or
  `--condition-index-offset 14` on GPU3 and their separately recorded
  progress/output paths.  Condition 2 moved
  `1.602059 -> 0.810867 -> 0.023410 px`; condition 14 moved
  `1.404833 -> 0.699856 -> 0.011812 px`.  Both had zero failure/rejection.
  The top-level statuses correctly remain `failed` because each has only one
  sample and cannot satisfy the 32-condition gate.  Artifact bytes/SHA256 are
  `52,267`/`b43c45918818130b08db47253751a2d18fdd12c39b81033fd844bc53564e3d23`
  and
  `52,112`/`7bafe83bce9f426d04c0ba59ad2abf8239e08543447d507173dc277db8d7a152`.
  Final-checkpoint SHA256 values are
  `5bf6ea1726f60340784165379225a971d857e40eb796aaf949dfc32f0b501424`
  and
  `aad9fea00bee55000049d696f0694995e6cad9081512d3e99d3f3e54e8e81fdb`.
  Training elapsed was `1486.101/1480.555 s`; each run used
  `7,138,167,296/7,411,335,168` peak allocated/reserved bytes.  Together with
  condition 0, these results show that both horizontal and vertical residuals
  are individually learnable; the unresolved failure is joint coverage or
  multi-condition optimization, not a demonstrated missing vertical signal.
- A registered 32-condition BF16/SWA D1 run with a 2,000-step budget started
  afterward but did **not** finish.  The process is no longer present and its
  stderr ends after a normal heartbeat without a Python exception.  The last
  atomic progress checkpoint is a restricted-loadable `tiny_progress` at step
  184 (19,389,293 bytes, SHA256
  `b046f763ea844087ec3803ecce24a347e11de1056cdcc27648b65b179f17f38e`);
  the last completed evaluation was step 160 at `0.925703 px`.  SWA had not
  begun because its configured start is step 1536.  The partial running JSON is
  deliberately not committed as final evidence.  The checkpoint records the
  exact 2,000-step signature and resource identities, so it can be considered
  for strict resume after the unexplained external termination is accounted
  for; it is not a D1 pass.

CUDA warns that `grid_sampler_2d_backward_cuda` and
`adaptive_avg_pool2d_backward_cuda` have no deterministic implementation.
Seeds, sample IDs and RNG states are recorded, but current CUDA results should
not be described as bitwise deterministic.

## P5/P6 status

Not started.  E00/E01 and later experiments remain blocked by the incomplete
TINY-S/TINY-8 learnability gate.  No test item has been evaluated and no model
validation result has been claimed.

The D1 efficiency work is now explicitly ordered after E00/E01, the equal-start
E02--E05 feature-group comparison and validation-only main-configuration
selection.  The earlier step-864 interrupted dense observation was corrected
from “baseline” to an initial failure diagnosis because it has neither a final
artifact nor a checkpoint for paired evaluation.  The machine-readable sparse-
query/3x3-window registration is a deliberately non-executable draft at
`/home/disk1/MHINet/configs/d1_efficiency_ablation_v1.2.json`, 96 lines, SHA256
`d6cce6aa4ce41caeceab6ee61297c77f9af2ccc5fd6a205862aa017b33d7b2f9`.
Known facts (dense 5x5, D2-guided eligibility, 3x3's theoretical 64% candidate
reduction, grouped validation and seed-0 then seed-0/1/2 policy) are fixed;
unknown selector thresholds, exact validation IDs, runtime repetitions and
accuracy/speed/memory decision margins remain `null` blockers rather than
invented defaults.  No D1 efficiency implementation or result is claimed.

P5 entrypoint preparation has begun without executing E00/E01.  The evaluator
now emits both explicitly named all-finite-geometry diagnostics and
conditional-valid H0/every-update/final summaries, so a finite fallback from a
failed initializer cannot masquerade as a valid estimate.  A rejected proposal
keeps its guarded retained H in the trajectory denominator while rejection is
reported separately.  Per-pair JSONL and flattened CSV now retain every state,
accepted/reason code and name, support count, condition number, solve status,
corner-update magnitude and saturation.  Window recall remains explicitly
unavailable rather than fabricated because the current forward contract does
not retain GT-to-window membership.  Five CPU evaluation regression tests
passed; this is output-contract engineering, not P5 validation evidence.

## 2026-09-09 — MCNet-style MHIR rewrite and D2 mainline cutoff

This section supersedes the **current applicability** of the earlier P3/P4,
D1, TINY-8, parameter-count, and eight-update records; it does not erase their
historical provenance. No model-validation claim is made here.

### User decision and active architecture

- Public module names remain GHIM, CGMDP, and MHIR. The active CGMDP path is
  `F_MVT -> D16 -> D8 -> D4 -> D2`; it stops the cumulative decoder before
  scale `1`. The active MHIR schedule is `(8,8,4,4,2,2)`, producing
  `H0 -> H1 -> ... -> H6 = H_final`.
- D1 was not physically deleted. Its adapter, 25-channel local correlation,
  nine-block decoder, and explicit diagnostic entry remain registered. In all
  mainline training profiles its parameters are frozen and absent from the
  optimizer; a normal forward neither constructs `D1` nor calls D1 adapter,
  correlation, or decoder.
- Active architecture config:
  `/home/disk1/MHINet/configs/mhinet_mcnet_v1.2.json`, 2026-09-09 SHA256
  `92095ffe16a5e650bab641df3fd17e16a0bf1f34ab47c82c59f37faa43ae092a`.
  The original handoff package was deliberately preserved instead of edited.
  Verification command:

  ```bash
  /root/miniconda3/envs/loma-repro/bin/python \
    MHINet_server_handoff_v1.2/verify_package.py \
    MHINet_server_handoff_v1.2
  ```

  Result: 31 files checked, zero errors; original architecture-config SHA256
  remains `4a859f413ee4be59acb1dc6c0cf189e80b8f4e7b3f54dc9f117acd1ac5cfe128`.

### MCNet source pin and dimension adaptation

- Structural reference: `https://github.com/zjuzhk/MCNet.git`, fixed commit
  `cc03479689b3cf40f0c384954f338b434765c155`. On 2026-09-09,
  `git ls-remote` resolved both remote `HEAD` and `refs/heads/master` to this
  commit. Audited reference files were `update.py`, `network.py`, and the
  four-point geometry utilities.
- Each decoder now consumes correlation only. It applies
  `Conv1x1(K,64,bias=True)`, repeated
  `Conv3x3(s1,p1,bias=True) -> GroupNorm(8) -> ReLU -> MaxPool2(s2)`, then
  `Conv1x1(64,2,bias=True)` directly on a `2x2` spatial corner grid. BCHW is
  permuted to BHWC and flattened row-major as TL/TR/BL/BR, with x/y last.
- MCNet's power-of-two inputs do not directly cover the current grids. Pooling
  therefore uses `ceil_mode=True`: D8 uses 6 blocks
  (`98->49->25->13->7->4->2`), D4 uses 7, D2 uses 8, and retained D1 uses 9.
  There is no adaptive pool or MLP in the production path.
- Search windows remain the previously registered 9x9, 9x9, 7x7, and dormant
  5x5 windows (`K=81/81/49/25`). This avoids changing decoder topology and D1
  window policy in the same experiment. Candidate-valid masks remain outside
  the decoder and are used only for support/geometry guards.
- Intentional MHINet differences from official MCNet are explicit: H0 rather
  than identity initialization, normalized FP32 cumulative corner/H state,
  zero-initialized output projections, per-scale tanh bounds
  `32/16/6/(2 dormant) px`, direct H-guided sampling with invalid-zero masks,
  and pre-screened differentiable DLT. A rejected update retains the preceding
  H/T; no update detaches H or T.
- Registered new parameters, including inactive D1, are `1,176,712`. The
  mainline D8/D4/D2 trainable-new-parameter count is `833,222`. Counts include
  the four retained adapters and scale-specific decoders; they do not imply
  D1 activation/FLOP execution.

### Runtime, checkpoint, and gate changes

- `SharedFeatureProvider` defaults to `(8,4,2)` and breaks cumulative decoding
  after scale `"2"`. It records `dedode_steps=4` and `dedode_scale1=0` for the
  mainline; a unit test separately proves that explicit `(8,4,2,1)` still
  reaches scale `"1"`.
- Formal training configs reject D1-containing active schedules. E01 now uses
  `(8,4,2)` twice. Evaluation and profiling default to the same six updates.
- The required tiny gate is now TINY-S-D8, TINY-S-D4, TINY-S-D2, and TINY-6;
  its identifier is `P4_TINY_S_TINY_6`. D1/TINY-8 remain optional diagnostics
  and cannot satisfy or be merged into the current gate.
- MHINet-owned checkpoint format was bumped from version 1 to version 2.
  Version-1 decoder/optimizer/SWA state is rejected. Formal resume/evaluation
  validates architecture SHA256 and `mhir_revision` before mutating model or
  optimizer state. New tiny filenames contain an architecture-hash prefix and
  progress signatures bind architecture, data, runtime, weights, and protocol.
- In particular, the old step-184 D1 progress checkpoint with SHA256
  `b046f763ea844087ec3803ecce24a347e11de1056cdcc27648b65b179f17f38e`
  is not resumable under this architecture. The concurrently written
  `/home/disk1/MHINet/artifacts/p4_tiny_s_d1_translation_swa.json` remains an
  unstaged legacy heartbeat and is not current evidence.

### Verification completed so far

Targeted command:

```bash
/root/miniconda3/envs/loma-repro/bin/python -m unittest \
  tests.test_model tests.test_feature_provider tests.test_modules \
  tests.test_iterator tests.test_tiny_overfit tests.test_tiny_gate \
  tests.test_train tests.test_checkpointing tests.test_config \
  tests.test_losses_metrics -v
```

Result: 51/51 tests passed. This covers exact MCNet block topology and
parameter counts, 784-derived meta shapes, TL/TR/BL/BR mapping, output
zero-initialization, second-step upstream gradient, six-round no-op/update
state, correlation recomputation, no-detach final gradients, default D1
non-execution, explicit D1 availability, CGMDP scale-1 early stop, checkpoint
metadata fail-closed behavior, TINY-6 protocol, and six-update loss/metrics.

Full CPU regression command:

```bash
/root/miniconda3/envs/loma-repro/bin/python -m compileall -q mhinet tests
/root/miniconda3/envs/loma-repro/bin/python -m unittest discover -s tests -v
```

Result: compileall succeeded and 88/88 tests passed in 1.440 seconds.

Real-resource P3 command (physical GPU 1 exposed as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli zero-init-smoke \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p3_zero_init_smoke_mcnet_d2.json --overwrite
```

Result: `passed`. Six of six updates were accepted; schedule was
`8,8,4,4,2,2`; all registered output-projection weights/biases and all deltas
were exactly zero; DINO/MVT/VGG ran once, CGMDP executed scales 16/8/4/2 only
(`dedode_steps=4`, `dedode_scale1=0`). H0-to-H6 maximum/mean corner difference
was `6.103515625e-05/2.986308027175255e-05 px`. Single unwarmed forward was
`742.833 ms` and peak allocated memory was `2,061,168,128` bytes; neither is a
production benchmark. Artifact: 4,486 bytes, SHA256
`29e90988578e1a1cb9ad32a75a0c0146a0282605c50da06c9806807bb53416be`.

Real-resource P4 gradient command (physical GPU 2 exposed as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli gradient-audit \
  --runtime configs/runtime_paths.server.json \
  --output artifacts/p4_gradient_audit_mcnet_d2.json --overwrite
```

Result: `passed`. At step 0, all D8/D4/D2 output projections had nonzero finite
gradient and all upstream decoder/adapter gradients were exactly zero as
expected. At step 1, every active upstream decoder and adapter gradient was
finite and nonzero. A final-H6-only loss reached all three earlier decoders.
The MVT-to-GHIM and MVT-to-CGMDP routes both had nonzero finite MVT gradients;
the CGMDP route also reached VGG and the active DeDoDe decoder. DINO and GHIM
head parameter gradients remained absent, and registered D1 parameters stayed
frozen with no gradients. Peak allocated memory was `3,103,597,056` bytes for
the cached six-round heads check and `5,821,080,576` bytes for the separate
joint-branch diagnostics. The artifact also embeds the active architecture,
provider, and checkpoint provenance. Artifact: 10,308 bytes, SHA256
`cca64b4f3573f9c40a70b980f4221b874fa6fe34c3dad06445ef5bf9c3be06fd`.

Real-resource P4 two-step heads-only profile command (physical GPU 1 exposed
as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli profile \
  --runtime configs/runtime_paths.server.json --profile heads \
  --optimizer-steps 2 \
  --output artifacts/p4_profile_heads_mcnet_d2.json --overwrite
```

Result: `passed` on an RTX 4090 at the real `784x784` input size. Peak
allocated/reserved memory was `3,999,755,264/4,882,169,856` bytes. Unwarmed
step-0 forward/backward time was `0.558349/0.558144 s`; step 1 was
`0.257731/0.434172 s`. Both steps accepted all six updates and recorded one
DINO, MVT, VGG, and cumulative DeDoDe call, four decoder steps through D2,
and zero scale-1 steps. At zero initialization, step 0 had 372 nonzero new-head
gradient elements; after the first update, step 1 had 798,968, consistent with
the expected delayed upstream-gradient activation. DINO and the frozen GHIM
head had no gradients. Artifact: 8,582 bytes, SHA256
`c4dcb825c5d01057d3c9a116eee837ff411e353ebdd25727f73856c1832f34a1`.

Real-resource P4 two-step joint profile command (physical GPU 2 exposed as
`cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u \
  -m mhinet.cli profile \
  --runtime configs/runtime_paths.server.json --profile joint \
  --optimizer-steps 2 \
  --output artifacts/p4_profile_joint_mcnet_d2.json --overwrite
```

Result: `passed` on an RTX 4090. Peak allocated/reserved memory was
`10,934,943,232/11,836,325,888` bytes. Unwarmed step-0 forward/backward time
was `0.569247/0.920599 s`; step 1 was `0.265485/0.889338 s`. Both steps used
the six-update `8,8,4,4,2,2` schedule, kept H/proposal tensors differentiable,
ran shared DINO/MVT/VGG once, stopped the cumulative decoder after four steps,
and did not execute D1. Total active trainable parameters were `107,061,671`.
The expected zero-head boundary was visible at step 0: DeDoDe and VGG gradient
tensors existed but had zero norm, while MVT already received gradient through
H0. At step 1, DeDoDe/VGG/MVT gradient norms were respectively
`0.361962/0.489198/34.208104`, all finite and nonzero; DINO and the frozen
GHIM head still had no gradients. Artifact: 8,707 bytes, SHA256
`92a67d98bee23f1de71e8c3d468c6f00442737b65e8357d76b8ff92b1a391fe4`.

Still pending at that checkpoint for the new architecture: checkpoint-v2 exact-boundary resume
smoke, D8/D4/D2/TINY-6 tiny-overfit, E00, and E01. Earlier artifacts do not
fill these gaps. E00/E01 remain blocked until the new P3/P4 and TINY-6 gate
pass.

## 2026-09-10：执行正式训练前完整检查（进行中）

停止边界：完成 P0–P4 审查及 tiny gate 后停止，不执行 E00/E01，也不执行解冻或结构消融。

发现并修复两个入口问题：alignment 的样本来源由 test 改为 train 并显式记录 split/test_used；CGMDP 的 scale1 decoder 虽然已被 forward 截断，但联合 profile 仍将其参数标记为可训练，现强制冻结并保持 eval，同时保留 state_dict。新增回归测试确认 scale1 参数不进入 optimizer，重复 train/eval 不会重新激活。89/89测试通过，原始设计包31项校验通过。

| 检查 | 本次结果 | 证据 |
| --- | --- | --- |
| P0真实资源与环境 | passed | `artifacts/p0_preflight_mcnet_d2.json` |
| P1 train样本旧输出对齐 | H0/D8/D2逐元素一致；D1调用0 | `artifacts/p1_alignment_mcnet_d2_train.json` |
| P2几何和相关性reference | 20/20 passed | `artifacts/p2_reference_mcnet_d2.json` |
| joint双步复测 | passed；活跃参数107,039,206；峰值10,934,943,232 bytes | `artifacts/p4_profile_joint_mcnet_d2_frozen_d1.json` |

上述四项SHA256依次为：

- `b867ee4753f890ee138fdef037e4908d722877e3cb7ff03b88b9d2978ec47659`
- `633faf7a7ffa98053999578f2a27567a9accf87a6c5e48bb6d76c52ca8a50977`
- `32e4861895df398a31951eece97eb1e382de4a7e238b041c26e03ebaa9c787be`
- `773f79164c3fa95469adbb585f82d5523d7f6649e83a947a57e0ba868e4ded67`

真实Python为 `/root/miniconda3/envs/loma-repro/bin/python`，3.10.20；PyTorch2.11.0+cu128、CUDA12.8、cuDNN91900、torchvision0.26.0、NumPy2.2.6、GPU驱动565.57.01。父shell继承的CONDA_PREFIX为base，preflight新增 sys.executable/sys.prefix 以消除歧义。可选pytest/openpyxl不可导入，timm导入触发NumPy旧API错误；当前已执行入口使用unittest和现有LoMa加载路径正常运行，不据此声称全部可选依赖可用。

复测命令（工作目录 `/home/disk1/MHINet`）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli preflight --runtime configs/runtime_paths.server.json --output artifacts/p0_preflight_mcnet_d2.json --overwrite
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli alignment --runtime configs/runtime_paths.server.json --output artifacts/p1_alignment_mcnet_d2_train.json
/root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli geometry-corr-check --output artifacts/p2_reference_mcnet_d2.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli profile --runtime configs/runtime_paths.server.json --profile joint --optimizer-steps 2 --output artifacts/p4_profile_joint_mcnet_d2_frozen_d1.json
```

四项tiny共用参数：`--sample-protocol one_pair_residuals --residual-profile translation --residual-bound-fraction 0.5 --precision bf16 --sample-count 32 --seed 0 --max-steps 2000 --eval-interval 32 --threshold-mace-px 0.1 --weight-average-start-step 1536 --heartbeat-interval 64`。均为 `python -u -m mhinet.cli tiny-overfit --runtime configs/runtime_paths.server.json`，不改变L1或加入额外损失。

| experiments | 物理GPU | progress路径（outputs/tiny_overfit下） | artifact（artifacts下） |
| --- | ---: | --- | --- |
| D8 | 2 | mcnet_d8_full_progress.pt | p4_tiny_s_d8_mcnet.json |
| D4 | 3 | mcnet_d4_full_progress.pt | p4_tiny_s_d4_mcnet.json |
| D2 | 3 | mcnet_d2_full_progress.pt | p4_tiny_s_d2_mcnet.json |
| TINY-6 | 2 | mcnet_6_full_progress.pt | p4_tiny_6_mcnet.json |

D8/D4初次启动未限制线程，观察到各162线程；随后SIGINT结束这两个已确认身份的进程，从各自原始边界以 `--resume-progress` 恢复，设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。D2/TINY-6自启动即使用相同线程限制。此调整为资源配置，未变更预算/优化器/样本顺序，不宣称跨线程设置逐位相同。源码冻结修复不改变heads tiny的执行图或权重。

`scripts/summarize_pretraining.py` 从现有artifact生成 `docs/pretraining_mcnet_d2_summary.md` 与JSON索引，包含每轮误差、原始/平均读出、显存、缓存前向时间和checkpoint身份；缺失或运行中项不会被推定通过。运行时序及最终状态以原始artifact和progress为准。

### D8完整预算与checkpoint重载（已完成）

1664步达到登记门槛并提前停止。1536–1664共129次参数平均后的H0/H1/H2为14.868890/0.291660/0.098226 px，失败0/32、拒绝0/64；原始终点H2为0.290066 px，单独保留。训练循环累计617.205 s，峰值571,810,816 bytes；共享GPU下缓存前向56.555 ms/样本，不能视作独占端到端基准。结果表见 `docs/pretraining_mcnet_d2_summary.md`。

原始artifact SHA256：`7c2f1fae0cf576c7b753fce0c2875c88746101b8b78d3723b85127adc386f294`。平均checkpoint为 `outputs/tiny_overfit/tiny-s-d8_one-pair-residuals_translation_bf16_weight-average-from-1536_arch-92095ffe16a5_bound-0p5_budget-2000_seed0.pt`，SHA256 `0da998f748cb901d0674e2767931603bbff2987e38aea08cc9591a460c39a15b`，不能用于优化器续训。

独立重载命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -m mhinet.cli tiny-checkpoint-audit --runtime configs/runtime_paths.server.json --checkpoint outputs/tiny_overfit/tiny-s-d8_one-pair-residuals_translation_bf16_weight-average-from-1536_arch-92095ffe16a5_bound-0p5_budget-2000_seed0.pt --experiment D8 --sample-count 32 --sample-protocol one_pair_residuals --residual-profile translation --precision bf16 --seed 0 --residual-bound-fraction 0.5 --output artifacts/p4_tiny_s_d8_mcnet_checkpoint_audit.json
```

重载资源签名严格匹配，序列化/载入/前向后权重哈希一致，重新计算的轨迹与原始artifact一致。重载artifact SHA256：`3a96402d0273aa835d8ebcd1299428d10ae18d0ea08846640589dbda5873cccf`。

`scripts/analyze_tiny_conditions.py` 按H0平均残差的主轴与符号分组，生成表格及JSON。D8平均读出 x+/x-/y+/y- 的最终误差分别为0.090299/0.087680/0.129368/0.120216 px。总体均值门槛通过并不表示每方向或每样本均低于0.1 px。饱和比例和拒绝更新均为0。32步probe与完整结果均保存对应分方向证据；其余三个完整实验仍在运行，合并gate尚未通过。

## 2026-09-10：补齐续训对照与 D8 短程结果

新增 `docs/results_mcnet_d2.md`，用表格与文字同时记录六轮资源、checkpoint-v2 对照和 D8 32步诊断。复核此前保存的两条运行记录及最终权重：恢复边界、RNG、数据游标正确，但 adapter 的119个权重元素存在最大1.9595e-6的差异，不能声明逐位一致。详情、原始目录和 checkpoint SHA256 均见结果表。

此前续训命令（工作目录 `/home/disk1/MHINet`，物理 GPU1）：

```bash
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/train_minimal_smoke.json --output-dir outputs/mhinet_resume_smoke_mcnet_d2_v2 --stop-after-optimizer-step 1 --overwrite
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli resume --runtime configs/runtime_paths.server.json --config configs/train_minimal_smoke.json --output-dir outputs/mhinet_resume_smoke_mcnet_d2_v2 --resume outputs/mhinet_resume_smoke_mcnet_d2_v2/checkpoints/step_0000001.pt
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli train --runtime configs/runtime_paths.server.json --config configs/train_minimal_smoke.json --output-dir outputs/mhinet_uninterrupted_smoke_mcnet_d2_v2 --overwrite
```

本次 D8 命令（物理 GPU2，独立新输出路径）：

```bash
CUDA_VISIBLE_DEVICES=2 /root/miniconda3/envs/loma-repro/bin/python -u -m mhinet.cli tiny-overfit --runtime configs/runtime_paths.server.json --experiments D8 --sample-protocol one_pair_residuals --residual-profile translation --residual-bound-fraction 0.5 --precision bf16 --sample-count 32 --seed 0 --max-steps 32 --eval-interval 32 --threshold-mace-px 0.1 --heartbeat-interval 16 --progress-checkpoint outputs/tiny_overfit/mcnet_d8_probe32_progress.pt --output artifacts/p4_tiny_s_d8_mcnet_probe32.json
```

运行完成32步，梯度有限，H0/H1/H2均值为14.868890/9.773121/7.634817 px，零失败和零拒绝。未达到0.1 px，按失败结果保存，不填补正式 tiny gate。新记录仅执行 D8，未启动 D1。原有 `artifacts/p4_tiny_s_d1_translation_swa.json` 用户工作区改动继续保留且不提交。
