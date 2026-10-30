# V5 S2 model design and source audit

The first V5 comparison uses a frozen causal five-frame V3 direct RGB descriptor,
four adjacent signed optical-flow fields, their validity masks, and four actual
frame intervals. The output is one `[vx, vy]` vector in m/s. Model inputs never
include labels, pose, CAN, IMU, or split metadata; the forward mapping rejects
every key outside `descriptor, flows, valid, fb_mask, pair_dt`.

## Audited reuse and adjustment

| Original suggestion | Evidence / issue | V5 choice and affected comparison |
|---|---|---|
| Prefer ImageXY as R00 | Its supplied best checkpoint says `online_mode=false`, T=13, gray 64x64, despite the current YAML saying true. It trained on official train, which includes the V3 dev scenes. | Use the V3 direct T=5 RGB encoder for every RGB experiment. Preserve ImageXY as historical evidence, not a current causal baseline. |
| Reuse V3 simple fusion | V3 and V4 actually gate a geometric velocity against a direct RGB velocity. | Reuse only RGB encoding; introduce a single velocity head on concatenated features. |
| Reuse V3 three-pair adapter | V3 uses five images but only its final three flows for geometry. | Use all four adjacent pairs from the same five images for R01-R05. |
| R01 light vs R03 independent | Two identical CNN branches would duplicate an experiment. | R01 fixed 4x8 grid pooling + one linear projection; R03 learned spatial and temporal convolutions. |

The V3 direct checkpoint is
`../PureVision_V3/checkpoints/pretrained/direct_seed_42/best.pth`, SHA-256
`1d24368adb86643e68e28b0f69ea4ec658e008ff52b5fb7fc09d7bad06ac0e82`.
Its saved config names the retained V3 split manifest. The source trainer uses
`train_core` for optimization and `dev` for selection. V5 must preserve these
partition memberships, including when taking subsets. The old checkpoint does
not embed a split hash; provenance is supported by its saved configuration,
source script, retained manifest, and source verification documents. Historical
checkpoint metrics are not new V5 evaluation results.

`purevision_v5/_rgb_reference/flexinet_backbone.py` and `modules.py` are exact,
unmodified copies of the corresponding `../PureVision_V3/model/` files. Only
these two small source modules are vendored in a private namespace. Their actual
SHA-256 values are recorded below; there is no legacy dataset, geometry, gate,
uncertainty, or old velocity head dependency.

- `flexinet_backbone.py`: `3ce31001db9cb20f445db0fafa707804bfc2023d150d4b0314dd446041c1a155`
- `modules.py`: `78b1d491d8e3435b0492510799ad1f9774910108e16bbc2c232658b371cfcbb8`

## Frozen RGB contract

- RGB input is `[B,5,3,72,128]`, historical to current. PIL bilinear resize to
  128x72, convert to RGB, uint8/255, then `(x-.5)/.5`. `preprocess_rgb` matches the
  V3 torchvision transform. Existing resized uint8 caches need only normalization.
- The copied encoder includes CMA absolute image differences and has no raw RGB
  bypass. Thus R00 is an image-derived engineering baseline, not a no-difference
  representation control. Numerical flow is never passed through CMA.
- Load precisely `backbone.*` and `region_projection.*` with strict key/shape
  matching. Pool to `(time=1,height=2,width=3)`, reverse the height axis, flatten
  near-to-far and left-to-right regions, project each 256D region to 64D, and
  concatenate to 384D. This is the exact V3 region descriptor.
- `FrozenRGB.train()` remains evaluation mode, all parameters are frozen, and
  BatchNorm statistics cannot change when a parent model enters training.

## Downstream models

All newly trained downstream heads start fresh under the run seed. The common
head is `Linear(input_dim,192) -> ReLU -> Linear(192,2)`. No separate branch
velocity predictions, uncertainty head, geometric loss, or auxiliary velocity
loss is present. Every model receives `pair_dt / (1/12 second)` as four ordered
head inputs. The output retains m/s units.

| ID | Flow feature | RGB feature | Head input width |
|---|---|---|---|
| R00 | none | 384D frozen descriptor | 388 |
| R01 | signed `(u,v)` + valid, per-pair 4x8 average pool, ordered flatten, Linear(384,64)+ReLU | same 384D | 452 |
| R02 | R03 flow encoder | none | 68 |
| R03 | spatial/temporal flow encoder, no FB information | same 384D | 452 |
| R04 | R03 plus binary FB hint | same 384D | 452 |
| R05 | R04 and unreliable numerical flow set to zero | same 384D | 452 |

The independent flow encoder receives `[B,4,4,72,128]`: signed u/v displacement
pixels at the model resolution, valid, and hint. It applies per-flow 3x3 stride-2
Conv2d layers with widths 16/32/64 and ReLU, retaining a 9x16 spatial map. A
64-channel Conv3d with kernel `(3,1,1)` and temporal padding 1 precedes adaptive
pooling to `(1,2,3)` and Linear(384,64)+ReLU. All four pairs lie at or before the
current image; the temporal kernel does not access any future image.

R03 (and R02) uses a constant-one hint at valid positions, zero elsewhere. R04
and R05 use exactly the same supplied binary FB hint intersected with validity.
Their parameter shapes and operations are identical, except R05 additionally
zeros u/v where the hint is zero. R01 has no hint channel. Flow is not converted
to colors, normalized per image, clipped by magnitude, or absolute-differenced.

The upstream flow preparer supplies endpoint/warp legality and the documented
FB rule. Regardless of experiment, the model intersects validity with finite
u/v checks and uses `torch.where` to replace both components of invalid vectors
before arithmetic. Zero motion remains distinguishable from invalid flow through
the validity channel. Non-finite descriptors, timestamps, nonbinary masks, and
unexpected fields are rejected. R03-R05 are structurally comparable; R01/R03
still differ in trainable capacity, which must be reported.

## Required checks

CPU tests exercise all six forward/backward paths, temporal-convolution gradients,
strict inference fields and tensor shapes, signed-flow temporal ordering, NaN/Inf
sanitization, state save/load, frozen RGB selective loading and eval behavior,
exact RGB preprocessing, region order, and R04/R05 equivalence with all-reliable
flow. An encoder-input hook verifies their unreliable-region difference changes
only the numerical flow channels. A full actual RGB/cache smoke and task-level
training/resume checks belong to the preparation and training integration.
