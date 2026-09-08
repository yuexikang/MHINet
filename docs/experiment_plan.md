# D1 efficiency ablation plan

Status: **pre-registered plan; not implemented**. The current model remains the
dense D1 reference. This plan does not change training or inference logic.

## Objective and ordering

D1 has `784 x 784 = 614,656` source query positions. The reference evaluates
25 target candidates (a 5 x 5 window) for every position in each D1 refinement
round. The ablations test whether that cost can be reduced without materially
degrading homography accuracy or robustness.

The experiment order is fixed:

1. Complete the existing mainline P0--P4/TINY gate with dense D1. Do not use an
   ablation to bypass or redefine a failed mainline gate.
2. Preserve the dense D1 result and freeze its exact comparison checkpoint,
   data manifest, seed, evaluation pairs, and runtime protocol.
3. Run the dense reference under the profiling protocol below.
4. Run sparse-query D1 and 3 x 3-window D1 separately against that reference.
5. Consider a combined sparse + 3 x 3 variant only after both individual
   effects have been reported.

The test split remains sealed. Development and selection use the same
registered validation subset for every variant.

## Registered variants

| ID | D1 query policy | D1 search window | Role |
| --- | --- | --- | --- |
| `D1-DENSE-5` | all 614,656 source positions | 5 x 5 (25 candidates) | implemented reference and accuracy baseline |
| `D1-SPARSE-5` | deterministic subset selected from D2 evidence | 5 x 5 (25 candidates) | test sparse geometrically guided refinement |
| `D1-DENSE-3` | all 614,656 source positions | 3 x 3 (9 candidates) | test narrower D1 micro-adjustment |

`D1-DENSE-3` reduces D1 correlation candidates by
`1 - 9/25 = 64%` in theory. Its search center is the position projected by the
post-D2 homography (`H6`), so D1 is explicitly a local correction.

### Sparse-query policy

The sparse selector must use inference-time evidence only; ground truth may be
used to measure recall, never to select a query. Before the first comparison
run, freeze these selector details in the run configuration:

- derive an eligible D1 mask from D2 valid support and the current-H projected
  overlap;
- rank eligible locations with D2 correlation confidence;
- retain a fixed count or fraction while enforcing deterministic spatial
  coverage, including locations with leverage on the four-corner update;
- define tie-breaking, mask upsampling, boundary handling, and the behavior
  when fewer than the requested number of valid queries exist;
- keep the selected set fixed for both D1 rounds, or explicitly register a
  per-round reselection policy before running either variant.

The first ablation is same-checkpoint and inference-only: compute correlation
only at selected queries, scatter the results back into the existing dense
channel layout, mark unselected candidates invalid/zero, and reuse the
unchanged D1 adapter and refinement decoder. This isolates correlation savings
without attributing gains to retraining. A native sparse decoder would be a
different architecture and requires a separately registered, matched-budget
experiment.

### 3 x 3-window policy

The first 3 x 3 ablation is also same-checkpoint and inference-only. Compute
only the central nine offsets, place them in their corresponding locations in
the existing 25-candidate channel layout, and mark the other offsets
invalid/zero. This keeps the D1 refinement-decoder checkpoint identical while
measuring the correlation saving. A decoder whose input shape is changed to
nine candidates is a later architecture/training experiment, not part of this
direct comparison.

## Fair-comparison lock

Every row in the direct comparison must have identical values for the
following fields unless the field is the registered intervention itself:

- checkpoint path and SHA-256 (including identical GHIM, CGMDP, adapters, and
  refinement-decoder weights);
- code revision plus an explicit ablation patch/config identifier;
- seed and deterministic settings;
- data manifest hash, split, pair IDs, order, preprocessing, and sample count;
- training/fine-tuning budget (`0` for the primary same-checkpoint ablations);
- input size, batch size, precision, autocast policy, and query chunking;
- GPU model and ID, software versions, power/performance settings when
  available, warm-up count, timed repetitions, and synchronization method;
- active scales, two iterations per scale, H direction, and all geometry
  guards.

Run variants in an interleaved or counterbalanced order on the same otherwise
idle device. Never compare a warm-cache ablation with a cold-start baseline.
Keep every failed pair in denominators and preserve raw per-pair records.

## Required output fields

Each artifact must record the following, with the dense and ablation values
reported side by side and paired by pair ID.

### Accuracy and trajectory

- `H0`, every refinement state `H1` through `H8`, and explicit `H_final`;
- MACE in input and native-target pixels for `H0`, each `Hi`, and `H_final`;
- grid error and the registered success rates/AUC metrics at the same states;
- paired per-pair accuracy deltas relative to `D1-DENSE-5`, not only aggregate
  means;
- update acceptance/rejection and D1 delta magnitude/saturation for both D1
  rounds.

`H0` through `H6` should be numerically identical in a correct D1-only
same-checkpoint comparison. Record and fail the comparison on an unexplained
pre-D1 difference rather than treating it as ablation noise.

### Robustness

- GHIM validity failure count/rate (serialized legacy field:
  `stage1_failure_rate`);
- D1 update rejection count/rate and reason-code histogram;
- non-finite output count/rate;
- overall invalid/failure count/rate and success denominator.

### Resources and timing

- peak CUDA allocated and reserved bytes;
- end-to-end forward latency distribution (at least median and p95);
- D1-only latency and D1-correlation latency using synchronized measurements;
- warm-up/repetition counts and total evaluated pairs;
- selected-query construction/scatter latency, which must not be excluded from
  the sparse method's end-to-end latency;
- theoretical and measured speedup relative to `D1-DENSE-5`.

### Query recall and coverage

Record these fields for each D1 round, both per pair and in aggregate:

- selected query count and `query_fraction = selected / 614656`;
- valid-support coverage: selected valid D1 queries divided by all valid dense
  D1 queries;
- evaluation-only overlap coverage: selected queries in ground-truth overlap
  divided by dense queries in ground-truth overlap;
- query recall: fraction of evaluable ground-truth correspondences retained by
  the sparse selector;
- window recall: fraction of evaluable ground-truth targets lying inside the
  tested candidate window, reported for 3 x 3 and 5 x 5;
- spatial coverage by fixed image bins/quadrants and coverage of each
  corner-influence region;
- empty/under-budget selector count and the fallback used.

The exact evaluability mask, correspondence tolerance, confidence score,
spatial bins, and corner-influence definition must be frozen in configuration
before results are inspected. Recall/coverage diagnostics may use labels only
after inference and must not alter the selected queries.

## Decision and reporting rules

Before launching the ablations, register the allowed accuracy-loss margin and
minimum useful latency/memory improvement. A variant is not an efficiency win
if it crosses the accuracy or failure-rate margin, even if it is faster.
Report confidence intervals for paired accuracy and latency deltas, all failed
pairs, and both absolute and relative resource values.

The dense D1 result remains the primary reference. Do not overwrite its
artifact, merge ablation results into the mainline gate, or present a projected
64% candidate reduction as measured end-to-end speedup.
