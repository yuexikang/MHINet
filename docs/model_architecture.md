# MHINet model architecture and terminology

This document defines the canonical three-module description of MHINet. It
describes functional data flow; source-repository class names and serialized
keys do not define additional model stages.

## Canonical interface

| Functional module | Inputs | Outputs | Question answered |
| --- | --- | --- | --- |
| Global Homography Initialization Module (GHIM) | image pair `(I_A, I_B)` | `H0`, `F_MVT^A`, `F_MVT^B` | How should the two images be aligned globally? |
| Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP) | each image `I` and its cross-image context `F_MVT` | `D8`, `D4`, `D2`, `D1` for both images | Which descriptor should provide local correspondence evidence at each scale? |
| Multi-scale Homography Iterative Refinement Module (MHIR) | `H0` and the four descriptor pairs | `H1, ..., H8`, with `H_final = H8` | How should local matching evidence refine the global initialization? |

The complete flow is:

```text
Image pair (I_A, I_B)
          |
          v
1. Global Homography Initialization Module (GHIM)
   DINOv3 -> MVT -> global homography head/fitter
          |
          +---- H0 ------------------------------------------+
          |                                                  |
          +---- F_MVT^A, F_MVT^B                             |
                         |                                   |
                         v                                   |
2. Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP)|
   F_MVT + VGG local features                                |
       -> DeDoDe-style cumulative decoder                    |
                         |                                   |
                         +---- D8 / D4 / D2 / D1             |
                                           |                 |
                                           v                 v
3. Multi-scale Homography Iterative Refinement Module (MHIR)
   adapter -> H-guided local correlation -> refinement decoder
          -> corner residual -> guarded homography update
                         |
                         v
                  H1 -> ... -> H8 = H_final
```

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

`F_MVT` is not discarded after initialization: it is also the coarsest
cross-image context consumed by CGMDP. The shared DINOv3/MVT path is evaluated
once per image pair.

The implementation retains names such as `stage1_head`, `compute_stage1`,
`stage1_valid`, and `selected_stage1_checkpoint` for checkpoint, configuration,
artifact, and API compatibility. These are **legacy aliases for GHIM-related
objects or outputs**, not the name of a fourth module and not the public name
of GHIM.

## 2. Cross-image Guided Multi-scale Descriptor Pyramid (CGMDP)

CGMDP is not merely “VGG + DeDoDe,” and its outputs are not “DeDoDe
features.” It combines two sources:

- `F_MVT`, the coarsest cross-image context produced through two-image
  interaction;
- `F_VGG^8`, `F_VGG^4`, `F_VGG^2`, and `F_VGG^1`, the local VGG features at
  progressively finer image scales.

The DeDoDe-style cumulative decoder propagates the coarse description into
each finer level and adds the information decoded at that level:

```text
F_MVT   -> D16 (internal coarse decoder state)
              |
F_VGG^8 ------+-> D8
                    |
F_VGG^4 ------------+-> D4
                          |
F_VGG^2 ------------------+-> D2
                                |
F_VGG^1 ------------------------+-> D1
```

`D16` above names the internal coarse cumulative state; MHIR consumes the four
exported descriptors `D8`, `D4`, `D2`, and `D1`. The accurate description is:

> MVT cross-image context and VGG multi-scale local features are combined by a
> DeDoDe-style cumulative decoder to generate multi-scale matching
> descriptors.

For an input of height `H` and width `W`, `Ds` is the fused matching
descriptor at image scale `1/s`, with contract shape
`B x 2 x 256 x (H/s) x (W/s)`. The symbol says only which spatial scale the
descriptor represents:

| Symbol | Meaning | Size for a 784 x 784 input |
| --- | --- | --- |
| `D8` | fused matching descriptor at 1/8 scale | 98 x 98 |
| `D4` | fused matching descriptor at 1/4 scale | 196 x 196 |
| `D2` | fused matching descriptor at 1/2 scale | 392 x 392 |
| `D1` | fused full-resolution matching descriptor | 784 x 784 |

“MVT-guided multi-scale descriptors” is an acceptable short name for these
outputs. If provenance must be made explicit, use
`D_s = CGMDP_s(I, F_MVT)`. Do not interpret `D8/D4/D2/D1` as four original
DeDoDe network layers.

## 3. Multi-scale Homography Iterative Refinement Module (MHIR)

MHIR consumes the descriptor pyramid from coarse to fine:

```text
D8 -> D4 -> D2 -> D1.
```

At each scale it performs two refinement rounds. A round has the functional
path

```text
matching descriptor pair
    -> shared-view adapter
    -> current-H-guided local correlation
    -> correlation/refinement decoder
    -> four corner residuals (delta-corner)
    -> guarded four-point homography update.
```

No homography or descriptor is detached between rounds. With two rounds at
each of four scales, the state sequence is

```text
H0 --D8--> H1 -> H2
   --D4--> H3 -> H4
   --D2--> H5 -> H6
   --D1--> H7 -> H8 = H_final.
```

The current dense reference uses the following target-coordinate search
windows around the location projected by the current homography:

| Descriptor | Radius | Candidates per source query |
| --- | ---: | ---: |
| `D8` | 4 | 9 x 9 = 81 |
| `D4` | 4 | 9 x 9 = 81 |
| `D2` | 3 | 7 x 7 = 49 |
| `D1` | 2 | 5 x 5 = 25 |

The D1 reference evaluates all `784 x 784 = 614,656` source positions. Sparse
D1 queries and a 3 x 3 D1 window are efficiency ablations, not properties of
the current reference implementation; they are registered in
[`experiment_plan.md`](experiment_plan.md).

## Terminology rules

Use the following names in new reports, figures, and discussions:

- **GHIM** for global initialization from DINOv3/MVT, producing `H0` and
  `F_MVT`;
- **CGMDP** for the MVT-guided descriptor pyramid, producing
  `D8/D4/D2/D1`;
- **MHIR** for coarse-to-fine iterative homography refinement, producing
  `H_final`;
- **multi-scale matching descriptors** or **MVT-guided multi-scale
  descriptors** for `D8/D4/D2/D1`.

Avoid using “Stage1” as the public architecture name, “VGG + DeDoDe features”
as a name for CGMDP outputs, or “DeDoDe D8/D4/D2/D1 layers.” Legacy identifiers
may be quoted exactly when documenting compatibility, source provenance,
configuration keys, or serialized artifacts.
