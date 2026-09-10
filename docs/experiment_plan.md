# MHINet D2-mainline implementation and experiment plan

Data-split clarification (user instruction, 2026-09-10): training uses train
(36000 pairs); in-training validation uses val (2500 pairs); independent
evaluation, including E00, uses the full test split (1000 pairs). Test results
must not select hyperparameters or checkpoints. E00 is not a hard prerequisite
for starting E01. This clarification supersedes any earlier val-default wording.

Status: **P0–P4 mainline checks and merged tiny gate passed; stopped before E00/E01**. Training
protocol v1.2 remains authoritative. The active model is

```text
F_MVT -> D16 -> D8 -> D4 -> D2
H0 -> H1 -> ... -> H6 = H_final.
```

Completed evidence is summarized in [pretraining_mcnet_d2_summary.md](pretraining_mcnet_d2_summary.md).
The current controlled-H0 tiny gate passes using registered parameter averaging
for D8/D4/TINY-6 and raw parameters for D2; this is learnability evidence, not
held-out accuracy. Formal training has not started. The tables below retain
P5/P6 as future work, outside the current requested stopping boundary.

D1 code is retained but inactive. The current plan does not generate a D1
descriptor, train D1 modules, execute D1 refinement, or require D1 to pass a
gate. Former D1/TINY-8/eight-update artifacts are legacy evidence and cannot
be merged with this architecture.

## Non-negotiable protocol

- DINOv3 is always frozen. MVT, VGG, the active DeDoDe-style cumulative
  decoder, adapters, and MHIR decoders may be jointly trained in their
  registered phases.
- GHIM head parameters are frozen by default, but its inputs retain gradients.
  Do not wrap the frozen head in `inference_mode` or `no_grad` and do not
  detach `H0` in joint training.
- The mainline uses two rounds each at D8/D4/D2. Do not detach `H` or `T`
  between any of the six rounds; recompute H-guided correlation every round.
- The default loss is equal-weight proposal-corner L1 in input-target pixels.
  FGO, Planar, overlap, grid, and auxiliary-correlation losses are off.
- Screen finite/support/condition validity before differentiable solve. Never
  evaluate a singular inverse or unsafe division and hide its NaN afterward.
- Data splits are grouped by parent image/geographic region. Test remains
  sealed and never selects a model, checkpoint, loss, window, or D1 policy.
- “Code complete,” “run complete,” and “model validated” are different states.

## Architecture identity and legacy cutoff

The MHIR decoder is now correlation-only and structurally follows official
MCNet commit `cc03479689b3cf40f0c384954f338b434765c155`: 1x1 input projection,
repeated Conv3x3/GN/ReLU/MaxPool blocks, and a direct two-channel 2x2 corner
output. For 784-derived grids it uses ceil pooling and 6/7/8 active blocks at
D8/D4/D2. Zero output initialization, per-scale `tanh` bounds, and guarded
normalized-FP32 DLT are declared MHINet adaptations.

This rewrite and the D2 truncation define a new architecture hash. The
following cannot serve as current pass evidence:

- old 4x4-pool/MLP or mask/H-flow-input decoder checks;
- any dense-D1, sparse-D1, or D1-3x3 run;
- any `TINY-8` gate/result or checkpoint with four active scales;
- any former P3 zero-init/gradient/profile artifact whose architecture hash
  predates the MCNet decoder;
- interrupted D1 progress checkpoints, including otherwise valid optimizer
  and RNG state.

Keep those files for provenance; label them `legacy` or `superseded` and do not
overwrite them with current artifacts.

## P0-P6 execution order

| Phase | Current work | Required evidence before advancing | Stop condition |
| --- | --- | --- | --- |
| P0: resources and compatibility | Resolve actual LoMa/LoRetta/data/weight paths, environment versions, commits, hashes, grouped splits; audit old wrappers | Reproducible preflight; GHIM and shared DINO/MVT forward; selected checkpoint identity | Missing real resources are reported exactly; no fabricated result |
| P1: active shared descriptors | Compute DINO/MVT once; CGMDP stops after D2 and exports real D8/D4/D2; D1 is not decoded | Old compatible GHIM/D8/D2 alignment where applicable; exact active shapes; freeze/train groups; proof D1 branch was not called | Any active shape/provenance or trainability mismatch |
| P2: geometry and correlation | Four-corner state/DLT, coordinates, sampling, masks, chunked reference | Identity/translation/perspective; corner order; chunk equality; finite difference/gradcheck; unsafe branches isolated before solve | Any geometry/correlation failure; do not start training |
| P3: MCNet-style updater and full forward | Correlation-only decoders; D8/D4/D2 x2; six-update diagnostics | Parameter/shape checks; exact zero-init no-op; FP32/BF16 forward/backward finite; `H0 -> H6` attached; D1 execution count zero | Any mismatch, non-finite, detach, or unexpected D1 compute |
| P4: training mechanics and tiny | Loss, optimizer groups, second-step gradients, resume, four current tiny diagnostics | D8/D4/D2 single-scale tiny and TINY-6 all pass; two MVT gradient branches; VGG/active decoder gradients; full-invalid rule; profile | Gate remains closed on missing/mismatched/failed artifact |
| P5: mainline training | Freeze data/seed/config; E00 then E01; profile before E02-E05 | Validation `H0`, all six `H`, final, failure rate, memory, latency and per-pair rows; genuine non-no-move improvement | Do not expand experiments if baseline fails |
| P6: mechanisms and finalization | Equal-budget unfreezing; scale/round/resampling ablations; selected runs at three seeds | Main/ablation tables, trajectories, hard groups, commands and checkpoint hashes; one final frozen test evaluation | Training-only or single-mean gains are insufficient |

P0-P2 checks whose mathematical/operator contract is unchanged may be reused
as reference evidence, but only after confirming their recorded source/config
identity. P3 and P4 must be rerun for the current architecture.

## P3 acceptance details

The registered new modules include dormant D1 parameters for state-dict/code
stability, but the active D8/D4/D2 count and executed graph must be reported
separately. P3 checks at least:

1. decoder inputs are exactly correlation channels `81/81/49`;
2. spatial paths are `98->49->25->13->7->4->2`,
   `196->98->49->25->13->7->4->2`, and
   `392->196->98->49->25->13->7->4->2`;
3. output is `[B,2,2,2]`, then row-major `TL/TR/BL/BR` with `(x,y)` last;
4. zero-initialized output projections make every initial delta zero and
   `H_final` agree with accepted `H0` within the registered tolerance;
5. each second round recomputes correlation from the preceding accepted `H`;
6. no D1 descriptor/correlation/decoder forward occurs in the default model;
7. FP32 and CUDA BF16-autocast paths remain finite, with geometry in FP32.

## P4 tiny-overfit gate

The current gate has exactly four required rows:

| ID | Active scales | Update schedule | Role |
| --- | --- | --- | --- |
| `TINY-S-D8` | D8 | `8,8` | coarse decoder learnability |
| `TINY-S-D4` | D4 | `4,4` | middle decoder learnability |
| `TINY-S-D2` | D2 | `2,2` | fine active decoder learnability |
| `TINY-6` | D8/D4/D2 | `8,8,4,4,2,2` | complete active-chain learnability |

All four use the same architecture/config/data/resource hashes, seed policy,
controlled-condition generator, optimizer recipe, metric threshold, and
registered residual-bound policy. Each artifact records H0, every active H,
proposal corners, delta magnitude/saturation, solve acceptance/reason/support,
final MACE, peak memory, and latency.

The merger fails closed on a missing row, name/schedule mismatch, legacy
TINY-8 identity, protocol mismatch, non-finite result, rejected required
geometry, or metric failure. A diagnostic checkpoint may resume only at an
exact raw optimizer boundary with matching architecture and trajectory
signature. Parameter averaging may expose a diagnostic readout, but never
replaces raw endpoint reporting and its averaged checkpoint is non-resumable.

FC2/output-projection zero initialization has a special gradient audit: before
the first update, upstream decoder and adapter gradients may legitimately be
zero. After the projection has changed, repeat the audit at step two and later.
Separately verify both MVT routes:

```text
MVT -> GHIM -> H0 -> MHIR/loss
MVT -> CGMDP -> descriptors -> MHIR/loss.
```

## Main training table

All continuation rows start from the same seed-specific frozen E01-W
checkpoint and receive the same additional optimizer budget, data order, and
restart policy. Do not continue E02 into E03 and call that equal-budget.

| ID | Active groups / intervention | Start | Budget | Primary comparison |
| --- | --- | --- | ---: | --- |
| E00 | evaluate GHIM `H0` only | selected original checkpoint | 0 | initialization baseline |
| E01 | active adapters + D8/D4/D2 MHIR heads; six-update L1 | original checkpoint + new initialization | 10k | E00; must be non-no-move |
| E01-C | continue only active heads | same E01-W | 10k | E02 attribution control |
| E02 | active DeDoDe cumulative decoder + heads | same E01-W | 10k | E01-C when claiming decoder unfreezing gain |
| E03 | E02 groups + VGG | same E01-W | 10k | E02; MVT frozen |
| E04 | E02 groups + MVT | same E01-W | 10k | E02; audit both MVT routes |
| E05 | active DeDoDe + VGG + MVT + heads | same E01-W | 10k | E02/E03/E04 |

“Active DeDoDe” ends at D2. None of E00-E05 may silently instantiate or train
the full-resolution CGMDP decode. E02 relative to E01 includes extra training;
an isolated DeDoDe-unfreezing claim therefore requires E01-C.

Use seed 0 for the budget screen. Reuse that run, then add seeds 1 and 2 for
the chosen main configuration and key controls. Select configurations and
checkpoints on grouped validation only. Once registrations are frozen, run
the selected preregistered rows once on sealed test.

## Scale, iteration, and resampling ablations

Run structural comparisons only after the D2 mainline and equal-budget
unfreezing screen are sound. With frozen feature-training policy and matched
training budgets:

| ID | Active path | Updates | Comparison |
| --- | --- | ---: | --- |
| E06 | D8 only | 2 | E00/E01 |
| E07 | D8 -> D4 | 4 | E06/E01 |
| E01 | D8 -> D4 -> D2 | 6 | current full mainline |
| E09-D2 | one round each at D8/D4/D2 | 3 | E01; report FLOPs/time |
| E10-D2 | two rounds but reuse first-round correlation within each scale | 6 | E01; keep H/T graph attached |

The old E08 row (“D8/D4/D2 as a prefix of a D1 mainline”) is now identical in
architecture to E01 and is retired rather than rerun under a misleading new
label. The former four-scale E01 and four-round E09/E10 results remain legacy.
A free early-stop readout from a longer model is not a replacement for an
independently trained truncated model.

Loss comparisons, if later justified, modify one factor at a time from the D2
E01 baseline. SmoothL1 or sequence gamma can be registered as simple controls.
FGO and extra losses stay disabled until the base L1 model passes; they must
not conceal a geometry, gradient, or learnability failure.

## Required reporting for every train/eval row

For each pair and aggregate, record:

- `H0`, `H1` through `H6`, and explicit `H_final`;
- proposal-corner error and accepted-H error at every update, in input and
  native-target pixels;
- grid error, registered success/AUC metrics, update acceptance/rejection and
  reason histograms;
- GHIM validity, non-finite rate, overall failure rate, and denominators;
- peak CUDA allocated/reserved memory and synchronized end-to-end latency
  (median/p95, warm-up and repetition counts);
- architecture/config/data/checkpoint hashes, code revision, environment,
  command, seed, precision, active groups, and actual optimizer steps.

Do not omit failed pairs or choose a configuration from final error alone.
Inspect the full trajectory: a final improvement can coexist with unstable or
systematically rejected intermediate updates.

## Deferred D1 comparison

D1 is no longer a near-term implementation phase. Its adapter, 25-channel
correlation decoder, and nine-block MCNet-style path remain in source only so
future work does not require physically deleting and reconstructing code. The
present mainline saves both:

1. CGMDP's `D2 -> D1` full-resolution cumulative decode; and
2. two D1 MHIR rounds over `784 x 784 = 614,656` source queries.

If D1 is reactivated after P0-P6, create a separate frozen registration and a
new architecture hash. The first question is whether adding trained dense D1
improves accuracy enough to justify its measured cost relative to the D2/H6
mainline. Only then compare these D1 variants:

| Deferred ID | Query policy | Window | Purpose |
| --- | --- | --- | --- |
| `D1-DENSE-5` | all 614,656 positions | 5x5 (25) | trained D1 accuracy/cost reference |
| `D1-SPARSE-5` | deterministic D2-support/overlap/confidence/leverage subset | 5x5 (25) | reduce query count |
| `D1-DENSE-3` | all positions | 3x3 (9) | reduce candidates by theoretical 64% |

That future comparison must account for the extra CGMDP D1 decode, not only
D1 correlation. It must report H0-H6 identically before D1, H7/H8 afterward,
paired accuracy deltas, failure rates, query/window recall, support/coverage,
peak memory, end-to-end and module-only latency. Sparse selection overhead is
included. Projected candidate reduction is not measured speedup.

Old D1 long runs and partial checkpoints cannot supply this baseline because
they used a superseded decoder/architecture and may be incomplete. Preserve
them for history, never resume them into the D2 mainline, and do not call their
partial progress a failed or successful current experiment.
