# V5 numeric flow and FB contract

This implementation is S2 infrastructure. The checks below are not trained
velocity results, flow ground-truth endpoint errors, or full-dataset validation.

## Source and preprocessing decision

- Local `../PureVision_V4/weights/raft_large_C_T_SKHT_V2.pth` is used without
  downloading or updating weights. SHA256:
  `ff5fadd56d26b40647388883af1547351ea17868b765c05b27231e72dd16a322`.
- `torchvision.models.optical_flow.raft_large(weights=None)` loads that state
  strictly, runs in evaluation/inference mode, and freezes every parameter.
  Each direction explicitly requests 12 recurrent updates. Computation is
  float32, without AMP or TF32.
- nuScenes 1600×900 RGB is resized to 512×288 with bilinear interpolation,
  preserving its 16:9 aspect ratio. Different frame dimensions/aspect ratios
  fail rather than silently changing the contract. The official C_T_SKHT_V2
  transform maps RGB from [0,1] to [-1,1].
- **Change from V4:** V4 uses antialias=True. V5 uses antialias=False because
  torchvision's official RAFT example explicitly says this matches model
  training. This is an audited preprocessing change; V4 caches are not reused.
  Source: [official torchvision RAFT example](https://docs.pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html)
  and [raft_large weight documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_large.html).
  Installed implementations were also checked in torchvision 0.25.0+cu128,
  `transforms/_presets.py:OpticalFlow` and `models/optical_flow/raft.py:RAFT.forward`.

## Numeric meanings and ablation boundary

Native `forward[0]` is horizontal displacement and `forward[1]` is vertical
displacement, in **pixels on the 512×288 RAFT image**, from source to target.
Backward is a separately estimated target-to-source field, not `-forward`.
No absolute temporal differencing, flow color conversion, per-image
normalization, or division by dt changes these numeric flows. Timestamps and
positive `dt_s` are recorded separately; variable dt remains the temporal
model's responsibility.

A forward pixel is legal only if both components are finite and its displaced
endpoint is inside the target image. Nonfinite values are stored as zero with
their legality mask false. Finite out-of-bounds displacements remain visible in
the native audit tensor, but are zeroed in the model view after mask resizing.
FB reliability is computed on native float32 values before any resizing:

```
q = p + F(p)
Bq = bilinear_sample(B, q), align_corners=True
reliable = legal(F) and finite(Bq neighbours)
           and |F + Bq|² <= 0.01 * (|F|² + |Bq|²) + 0.5 px²
```

Flow is resized to 128×72 with bilinear interpolation; its x/y components are
then multiplied by 128/512 and 72/288 respectively. Binary masks use nearest
interpolation. Model views store float16 numeric flow and boolean masks; the
loader returns float32 flow and boolean masks. Original native values remain
float32 for auditing. Legal flow with failed FB remains present in R04 inputs;
only R05's model masks it. R04/R05 consume the exact same hint/mask definition.

## Cache and public interface

- `cache_path(root, source_token, target_token)` gives a pair-specific file
  under `root/v5_flow_v1/<token-prefix>/`. Tokens cannot contain path syntax.
- `load_model_pair(path, require_fb=False, expected_source=None,
  expected_target=None, expected_times=None)` returns
  `(flow[2,72,128], valid[1,72,128], fb_mask[1,72,128])`.
  It reads only the compact file, validates shape/dtype/finiteness, provenance
  identity, pairing/time when supplied, and the FB subset/availability rules.
- `<pair>.native.pt` retains native forward and, when available, backward
  tensors, directional legality and FB masks. `<pair>.pt` stores the compact
  view plus the native file's SHA256. Native save precedes compact save, each
  with a unique temporary file, fsync and atomic rename.
- Provenance includes ordered image tokens, timestamps, dt, absolute filenames,
  source byte hashes/sizes, image dimensions, checkpoint hash, preprocessing,
  direction mode, requested batch size and torch/torchvision/Pillow/CUDA versions.
  Native execution metadata records actual per-direction batch sizes, device,
  and the reused forward packet identity/checksum. Resuming an existing FB cache
  also checks that it uses exactly the currently requested frozen forward root.
  Preparation checks
  the native identity/checksum on reuse. A missing compact commit marker can be
  recovered from a validated native packet; mismatched caches fail explicitly.
- `--direction forward` writes `fb_available=false` and an all-zero placeholder
  FB mask. R01/R03 do not wait for backward estimation. R04/R05 must request
  `require_fb=True`, which rejects those placeholders.
- Forward and bidirectional output roots are separate. With
  `--direction bidirectional --reuse-forward-root <forward-root>`, the exact
  frozen native forward tensor is reused, only backward is computed, and
  compact forward/valid equality is asserted. Original nonfinite indicators
  survive this operation. Every requested forward pair must exist.
- `prepare_flow.py` uses only causal adjacent image pairs from selected
  five-frame windows. Default partitions are train_core/dev; labels are never
  passed into the flow engine. Scene hashes are checked when present. Full
  preparation (`--limit 0`) requires a clean detached, verified-pushed commit
  through the shared `code_version` guard.
- GPU inference is batched (`--batch-size 16` by default), while files remain
  pair-specific. Resizing a shared frame is reused within each batch.
  `--shard-index K --shard-count N` selects disjoint scenes by the ordered
  selected-scene index modulo N. All shards must use the same manifest and
  partitions. Sharding does not change image/flow pairing.

## Checked on this machine, 2026-10-30

The `speedest` interpreter, torch 2.10.0+cu128 and torchvision 0.25.0+cu128 were
used. Thirteen unit checks cover signed direction/bounds, the squared relative FB
threshold, nonfinite interpolation, vector unit scaling, nearest masks,
forward-only rejection by FB consumers, compact-only loading, corruption,
atomic recovery, strict checkpoint identity, causal manifest enumeration,
batch ordering/cache reuse, limits, disjoint scene sharding and rejection of
a changed forward source when resuming a bidirectional cache.

Three real pairs from scene-0001 were computed on GPU0 (RTX PRO 6000) and then
upgraded in a separate root by reusing native forward values. All native flow
was finite; all three compact forward tensors and validity masks were bitwise
equal across the two roots. Actual pair dt was approximately 0.1000, 0.1000 and
0.0500 seconds.

| Pair source prefix | Native valid fraction | Native FB fraction | F-warp image MAE | No-warp MAE | Wrong-sign MAE |
|---|---:|---:|---:|---:|---:|
| 020d7b4f | 0.909756 | 0.908061 | 0.021038 | 0.084446 | 0.118832 |
| caa2bfad | 0.909980 | 0.904019 | 0.020026 | 0.086759 | 0.120314 |
| b0e02e16 | 0.950446 | 0.950439 | 0.015669 | 0.068081 | 0.092273 |

Photometric MAE is measured on the FB-reliable native pixels using RGB in [0,1];
it checks direction plausibility and is not a ground-truth accuracy metric.
Largest float16 model-view rounding error was 0.003901 model pixels. Each
compact file occupied 59,233 bytes; each bidirectional native file occupied
2,953,621 bytes in this check.

Forward startup plus first pair took 1.726 s; the next two took 0.145 and 0.132 s.
Backward-only additions after startup took 0.168 and 0.195 s. The actual manifest
contains 134,724 train_core pairs and 23,789 dev pairs (158,513 total), so a
rough scalar extrapolation is six GPU-hours for forward plus eight for backward.
Batching subsequently improved forward throughput as measured below. Separate native
and compact roots require approximately 0.72 TB for both stages at this count.

A subsequent 32-pair benchmark on GPU0 compared batch sizes 8 and 16:

| Operation | Total, including startup | Final warm batch | Peak allocated GPU memory |
|---|---:|---:|---:|
| Forward, batch 8 | 4.199 s | 0.587 s / 8 | 1.63 GB |
| Forward, batch 16 | 3.435 s | 1.086 s / 16 | 3.22 GB |
| Add backward to saved forward, batch 16 | 5.464 s | 2.273 s / 16 | 3.16 GB |

Batch 16 is the default based on these observations. Its warm forward rate
extrapolates to roughly three hours on GPU0 for the selected manifest; this is
not a promise of sustained runtime. Disjoint scene shards can run on GPU0 and
GPU2 while GPU1 is reserved for image features/velocity training. Backward
preparation also includes native reads, checksums, two FB evaluations and larger
atomic writes, so its end-to-end time is longer than forward-only generation.
The 32-pair backward addition asserts unchanged forward/valid tensors. Batch
arithmetic is recorded and is not claimed bitwise equivalent to scalar RAFT.
Across these 32 pairs, batch-8 versus batch-16 native flow differed by at most
0.000359 pixels; none of the 294,912 model legality-mask pixels changed.

Artifacts are outside Git under
`/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/smoke/`:
`flow_forward/`, `flow_bidirectional/`, and `flow_audit.json`.
The batch benchmark uses `batch8_forward/`, `batch16_forward/`,
`batch16_bidirectional/` and `flow_batch_audit.json` under the same smoke root.
The early preparation JSONs have an empty commit field because an ownership
protection issue was discovered during smoke; `flow_audit.json` records the
correct development base `367db9662ff880c7a362bb55182fe6766f3c3332` and dirty
state. The CLI now uses the common scoped-safe Git version helper. These are
development smoke artifacts, not pinned formal runs. No full cache or formal
training was started as part of these checks.

Example commands, to run from an approved pinned V5 checkout:

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B -m pytest -q -p no:cacheprovider tests/test_flow.py
/home/zhangyu/.conda/envs/speedest/bin/python -B prepare_flow.py --manifest /path/to/dataset.json --output /path/to/flows/forward --direction forward --device cuda:0
/home/zhangyu/.conda/envs/speedest/bin/python -B prepare_flow.py --manifest /path/to/dataset.json --output /path/to/flows/bidirectional --direction bidirectional --reuse-forward-root /path/to/flows/forward --device cuda:0
```

The two full-cache commands belong in logged tmux jobs. The second follows
forward preparation and can overlap velocity training on other GPUs.
