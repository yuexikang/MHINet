# MHINet model architecture and terminology

This document defines the current functional architecture. The mainline stops
at `D2`; D1 source code is retained only as an inactive, explicit opt-in for a
future comparison. Serialized names inherited from LoRetta or earlier MHINet
revisions do not define extra model stages.

## Canonical three-module interface

| Functional module | Inputs | Mainline outputs | Question answered |
| --- | --- | --- | --- |
| Global Homography Initialization Module (GHIM) | image pair `(I_A, I_B)` | `H0`, `F_MVT^A`, `F_MVT^B` | How should the two images be aligned globally? |
| Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP) | each image `I` and its cross-image context `F_MVT` | `D8`, `D4`, `D2` for both images | Which descriptor should provide local correspondence evidence at each active scale? |
| Multi-scale Homography Iterative Refinement Module (MHIR) | `H0` and the three active descriptor pairs | `H1, ..., H6`, with `H_final = H6` | How should local matching evidence refine the global initialization? |

The complete active flow is:

```text
Image pair (I_A, I_B)
          |
          v
1. Global Homography Initialization Module (GHIM)
   DINOv3 -> MVT -> global homography head/fitter
          |
          +---- H0 ---------------------------------------+
          |                                               |
          +---- F_MVT^A, F_MVT^B                          |
                         |                                |
                         v                                |
2. Cross-image Guided Multi-scale Descriptor Pyramid     |
   F_MVT -> D16 -> D8 -> D4 -> D2                         |
                         |                                |
                         v                                v
3. Multi-scale Homography Iterative Refinement (MHIR)
   adapter -> H-guided local correlation
           -> MCNet-style correlation decoder
           -> delta-corner -> guarded homography update
                         |
                         v
                  H1 -> ... -> H6 = H_final
```

There is no mainline arrow from `D2` to `D1`. This saves both the D1 MHIR
work and CGMDP's full-resolution cumulative decode. D1 classes and registered
weights can remain in the repository without being called by the default
forward.

## 1. Global Homography Initialization Module (GHIM)

GHIM is the single global-initialization function

```text
(I_A, I_B) -> DINOv3 -> MVT -> H0.
```

DINOv3 supplies high-level image features. MVT performs two-image interaction,
producing the per-image cross-image contexts `F_MVT^A` and `F_MVT^B`. The
global homography head and weighted fitter use that context to estimate the
initial A-to-B normalized homography `H0`.

The functional outputs are therefore

```text
GHIM(I_A, I_B) = (H0, F_MVT),
F_MVT = (F_MVT^A, F_MVT^B).
```

`F_MVT` is reused as CGMDP's coarsest cross-image context. DINOv3 and MVT are
evaluated once per image pair, not once per scale or iteration. DINOv3 remains
frozen. MVT may be trainable in the joint phase. The GHIM head parameters are
frozen by default, but its input computation must remain in the autograd graph:
freezing parameters is not permission to run the head under `no_grad` or
`inference_mode`, and `H0` is not detached in joint training.

Implementation names such as `stage1_head`, `compute_stage1`, `stage1_valid`,
and `selected_stage1_checkpoint` remain compatibility aliases for GHIM-related
objects or outputs. “Stage1” is not the public architecture name.

## 2. Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP)

CGMDP is not merely “VGG + DeDoDe,” and its outputs are not standalone
“DeDoDe features.” It combines:

- `F_MVT`, the coarsest cross-image context produced by MVT interaction; and
- `F_VGG^8`, `F_VGG^4`, and `F_VGG^2`, the active local VGG feature levels.

The DeDoDe-style cumulative decoder propagates coarse context into each finer
active level:

```text
F_MVT   -> D16 (internal coarse cumulative state)
              |
F_VGG^8 ------+-> D8
                    |
F_VGG^4 ------------+-> D4
                          |
F_VGG^2 ------------------+-> D2  (mainline endpoint)
```

The accurate short description is:

> MVT cross-image context and VGG multi-scale local features are combined by a
> DeDoDe-style cumulative decoder to generate MVT-guided multi-scale matching
> descriptors.

For a `784 x 784` input, the mainline descriptor contract is:

| Symbol | Meaning | Contract shape |
| --- | --- | --- |
| `D8` | fused descriptor at 1/8 scale | `B x 2 x 256 x 98 x 98` |
| `D4` | fused descriptor at 1/4 scale | `B x 2 x 256 x 196 x 196` |
| `D2` | fused descriptor at 1/2 scale | `B x 2 x 256 x 392 x 392` |

The view axis of size two stores A and B. `Ds` identifies a spatial scale, not
a source network layer. `D_s = CGMDP_s(I, F_MVT)` makes the provenance
explicit.

The dormant path can still define a `D1` tensor of shape
`B x 2 x 256 x 784 x 784` for a separately selected experiment. The current
main model must not request that output; otherwise it would pay the
full-resolution decoder and activation-memory cost that the D2 truncation is
intended to remove.

## 3. Multi-scale Homography Iterative Refinement Module (MHIR)

### 3.1 Six-update mainline

MHIR consumes the active descriptors from coarse to fine:

```text
D8 -> D4 -> D2.
```

Each scale reuses its scale-specific adapter and decoder for two rounds. Every
round recomputes its local correlation using the newly accepted homography:

```text
H0 --D8--> H1 -> H2
   --D4--> H3 -> H4
   --D2--> H5 -> H6 = H_final.
```

Neither `H` nor the cumulative four-corner displacement `T` is detached
between rounds. A rejected update retains the previous `H` and `T`; the next
round still recomputes evidence around that retained state.

### 3.2 Adapter and H-guided local correlation

A scale-specific 1x1 adapter maps each 256-channel descriptor to 64 channels
at D8/D4 and 32 channels at D2, using the same weights for views A and B. It
then applies pointwise L2 normalization.

For every source position `p`, the current normalized A-to-B homography maps
`p` to a target center `H_q(p)`. Target descriptors are sampled at the local
offsets around that center and cosine-correlated with the source descriptor.
There is no softmax. Invalid candidates are exactly zero in the correlation
tensor and remain separately represented by the boolean mask used by the
support guard.

| Scale | Source grid | Radius | Decoder input |
| --- | ---: | ---: | ---: |
| D8 | `98 x 98` | 4 | `B x 81 x 98 x 98` |
| D4 | `196 x 196` | 4 | `B x 81 x 196 x 196` |
| D2 | `392 x 392` | 3 | `B x 49 x 392 x 392` |

Candidate order is `dy` outer, `dx` inner, both ascending. Only the correlation
tensor enters the learnable decoder. Masks, H-flow channels, and query
coordinates are not concatenated to it. Current-H guidance is expressed by
where correlation is sampled.

### 3.3 MCNet-style correlation decoder

The structural reference is the official MCNet source at commit
`cc03479689b3cf40f0c384954f338b434765c155`, primarily `update.py` and
`network.py`. The learnable path follows its formula and topology:

```text
local correlation C_q
    -> 1x1 Conv(K_s -> 64, bias=True)
    -> N_s x [3x3 Conv(64 -> 64, stride=1, padding=1, bias=True)
              -> GroupNorm(8) -> ReLU -> 2x2 MaxPool(stride=2)]
    -> 1x1 Conv(64 -> 2, bias=True)
    -> B x 2 x 2 x 2
    -> permute BCHW to BHWC and row-major flatten
    -> delta T_q in B x 4 x 2, ordered TL/TR/BL/BR.
```

MCNet was written for a different image/correlation resolution. MHINet's
`784`-pixel pyramid uses ceil-mode pooling so boundary evidence is retained
while each production path terminates exactly on the 2x2 corner grid:

| Scale | Blocks | Spatial path |
| --- | ---: | --- |
| D8 | 6 | `98 -> 49 -> 25 -> 13 -> 7 -> 4 -> 2` |
| D4 | 7 | `196 -> 98 -> 49 -> 25 -> 13 -> 7 -> 4 -> 2` |
| D2 | 8 | `392 -> 196 -> 98 -> 49 -> 25 -> 13 -> 7 -> 4 -> 2` |

The direct 2x2 output is important: output channel zero/one are the two target
displacement coordinates, while the four spatial cells are the four corners.
It is not a pooled feature followed by an arbitrary eight-value MLP.

For code retention only, D1 remains registered as correlation-only `K=25`,
nine blocks, and
`784 -> 392 -> 196 -> 98 -> 49 -> 25 -> 13 -> 7 -> 4 -> 2`. It is excluded
from the active parameter group and forward path unless explicitly selected.

### 3.4 Declared MHINet adaptations

This implementation is **MCNet-style**, not a bitwise reproduction. The
following differences are intentional and must remain visible in configs and
reports:

- non-power-of-two `784` geometry uses the ceil-mode paths above;
- windows are scale-specific (`9x9`, `9x9`, `7x7`; dormant D1 is `5x5`);
- the final 1x1 output projection is initialized to all zeros;
- `tanh` bounds the per-round input-image-pixel residual to 32, 16, and 6
  pixels at D8, D4, and D2 respectively (dormant D1: 2 pixels);
- the homography/four-corner state and solve run in normalized FP32 with
  explicit validity guards.

Zero output initialization makes the initial refinement an exact no-op. It
also means upstream decoder/adapter gradients may be zero on the first
optimizer step; they must be checked again after the output projection has
updated, including both MVT gradient routes.

### 3.5 Cumulative corner update and safe solve

Let normalized image corners be `c = (TL, TR, BL, BR)`. Initialize

```text
T0 = project(H0, c) - c.
```

At update `q`, the decoder predicts input-target-pixel residual `delta_px`.
After converting it to normalized target coordinates:

```text
T_proposal = Tq + normalize(delta_px)
Q_proposal = c + T_proposal
H_proposal = four_point_DLT(c, Q_proposal).
```

The update is accepted only after finite/support/conditioning checks. Invalid
samples are isolated **before** the differentiable 8x8 `solve_ex`; the code
must never compute a singular inverse or unsafe division and then hide a NaN
with `where`. A post-solve projection-denominator check is also required.
Accepted samples take `(H_proposal, T_proposal)`; rejected samples retain
`(Hq, Tq)`. Training protocol v1.2 supervises the pre-guard proposal corners
with the default equal-weight proposal-corner L1.

## Parameters and inactive D1 accounting

The registered source still instantiates all four adapters/decoders so a future
D1 experiment has stable code and state-dict names. The expected registered
new-parameter total is `1,176,712`; the active D8/D4/D2 trainable-new-parameter
count is `833,222`. Reports must distinguish these values and must not count
inactive D1 parameters as executed FLOPs or activation memory.

## Terminology and evidence rules

Use **GHIM**, **CGMDP**, and **MHIR** in new text. Use “multi-scale matching
descriptors” or “MVT-guided multi-scale descriptors” for D8/D4/D2. Avoid
“Stage1” as an architecture name, “VGG + DeDoDe features” for CGMDP outputs,
or treating D8/D4/D2 as original DeDoDe layers.

Any former D1, TINY-8, eight-update, or pre-MCNet-decoder measurement is
legacy evidence. It can document history but cannot validate the current
six-update architecture or be merged into its gate.
