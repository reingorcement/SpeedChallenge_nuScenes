# S23 geometry supervision A/B v1

Only four seed42 runs are authorized: pilot A/B, independent full A, and full B
conditional on the one pilot. The original S23 build_six initialization,
48 tokens,2x3 features, frozen R03 encoders and original PLANAR_PROXY targets
are retained. Every run starts from the original initialization with new
AdamW/cosine20/RNG, not from a trained parent or pilot checkpoint.

`fusion_geometry_training.py` has independent configuration, epoch-state and
recovery schemas. It inherits the historical whole-epoch writer/evaluation
behavior without changing the interpretation of old grid/full checkpoints.
The CPU short check verifies A's actual first update is exactly the historical
velocity-only update, B has an actual nonzero changed update, and full-state
epoch-boundary resume works. Partial epochs replay; no weights-only resume claim.

## Sources and finite motion

Only the last forward pair is constrained. K and both camera extrinsics come
from the pair's actual calibrated_sensor records. Pixel centers follow the
existing align_corners=False resize chain. Rotation comes from source/target
ego_pose orientations, as uncertain offline training supervision.

The nuScenes maintainer clarified that the ego origin is the rear axle center
projected onto the ground:
https://github.com/nutonomy/nuscenes-devkit/issues/161#issuecomment-508015958
The actual per-record camera translation supplies metric scale; no simulation
height or CG lever is imported. Road planes approximate a local tangent through
this source ego ground projection. Their normal is estimated from visual road
homography, selected using the observed rotation and upward-sign/local-tilt
plausibility, rather than equated to gravity.

With E_s/E_t mapping ego to world, C_s/C_t camera to ego, and l_s/l_t the
calibrated camera lever arms:

```
R21 = C_t.T @ E_t.T @ E_s @ C_s
base = C_t.T @ (E_t.T @ E_s @ l_s - l_t)
velocity_map = -dt * C_t.T @ E_t.T @ Yaw(E_t)[:, :2]
X_camera_target = R21 @ X_camera_source + base + velocity_map @ [vx,vy]
```

Thus predictions remain endpoint yaw-plane velocities, with their relationship
to the actual body frame explicit. The last-pair displacement approximation is
endpoint velocity times actual dt. Offline ego translations are used only to
diagnose/gate that endpoint-vs-interval approximation at the fixed0.2m/s limit;
they are not substituted for the unchanged target or used as model inputs.
The old RobustVisualTwistSolver twist_dof3 denotes three camera translations,
not a planar[vx,vy,yaw_rate] solver. This experiment adds no rotation head.

## Static support and fixed applicability

One pretrained Cityscapes SegFormer-B0 is frozen:
`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`, revision
`21b3847fae21ddee674abd31129307b6a1235bd9`, weight SHA256
`027ed78d8ff9c535df5a66361c92b9673cca3cb923cbeb8c5802385b6b93194c`.
Its fixed RGB input is1024x576 with ImageNet normalization; road probability
at the128x72 lattice must be>=0.9 and survive boundary erosion, source/warped
target agreement, finite cached RAFT valid, and independent planar coherence.
No per-batch images, RAFT, segmenter, encoder or Python solver is used in training.

All thresholds and rejection reasons are fixed in
`fusion_geometry_prepare.POLICY`. At most one point per16x16 spatial stratum
is selected, for at most256 points. Invalid windows remain in every velocity
batch. Padded unit-depth entries are explicitly masked absent support, not
replacement depth measurements. No learned velocity or its residual selects
support. Semantics/homographies/poses/planes remain imperfect priors.

The only bounded real short check selected128 pilot train windows, four per
scene: slowest, fastest, greatest turn, largest temporal-approximation error.
70 passed fixed support, with every failure reason retained. Synthetic
longitudinal/lateral/rotation/lever/combined-motion and fixed depth/rotation/time
perturbations pass. Actual S23 output and speed-head gradients are nonzero.

Historical input dt subtracts large epoch-second floats before FP32 conversion.
Geometric dt uses integer-microsecond differences. Both are validated exactly
against their own timestamp constructions; the observed maximum difference
is about0.23 microseconds. No input dt changes or tolerance relaxation is used.

## Loss, calibration and automatic queue

Velocity loss is the original SmoothL1(pred/[5,1],target/[5,1],beta1,mean).
Geometry compares projected and observed forward flow, divided componentwise
by(fx*dt,fy*dt), giving normalized image motion rate in s^-1. SmoothL1 beta0.01
is averaged over valid points/components per window, then all batch windows.
Every geometric input is detached; only existing S23 speed predictions carry
gradients. A uses only velocity; B adds the single fixed lambda.

Lambda0.4452454566121366 was calibrated once as the median of
`0.1*||dL_vel/dprediction||/||dL_geo/dprediction||` on the first four prescribed
epoch0 pilot train batches. Batch identities, raw norms and source hashes are
in the published lambda proof. No dev selection, model update or weight search.

Both stages use20 complete epochs, batch64, FP32, AdamW1e-3/1e-4,
betas(0.9,0.999),eps1e-8,global clip2 and validation-then-cosine step.
Pilot uses7324/1826 windows and2300 updates per run; full uses132978/23480
and41560 updates, last batch50. Whole dev equal-window EPE selects best,
earlier on ties. Best train inference is once and dev predictions are reused.

The live `fusion_geometry_pipeline.py` queue starts full A independently.
Full B requires completed pilot A/B, passed numerical/provenance/nonzero
coverage checks, pilot B best EPE<=A*1.01, complete full train geometry and
verified GitHub publication of both the gate and cache sources. This1% is
development tolerance, not proof of improvement. Rejection starts no alternative
lambda, seed or extended budget and preserves/cancels only its own speculative
cache. No other task is stopped or resumed.

The coordinator publishes completed/dev-ready events, retries failed network
publication, and reuses5min server,30min Astra and hourly publisher monitoring.
Only this task's schedules are removed after its final report is verified.

## Reproduction artifacts and limits

Task-local auxiliary Python dependencies are pinned in
`configs/next/geometry_dependencies.txt`; base training environment is unchanged.
`prepare_fusion_geometry.py`, `verify_fusion_geometry.py`,
`calibrate_fusion_geometry.py` and `verify_fusion_geometry_training.py` provide
the bounded preparation/verification entry points. The published immutable plan
records exact arguments, four configurations, all IDs, sample/cache/checkpoint
sources, lambda proof, applicability contract and actual GPU receipts.

All dev windows remain official. Only training needs a geometric cache; dev
geometry-valid-region attribution is unavailable without extra preparation.
Report all axes/speeds/scenes, gain concentration and historical side-motion
adequacy rules. Full dev has only31 nonzero-lateral windows/5 scenes and11
above15m/s windows. Single seed42 cannot establish seed stability, CG velocity
or true sideslip acceptance. Historical S23/TAIL_FT are references and never
replace the new matched A or enter this A/B's seed statistics.
