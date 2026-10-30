# Evidence-driven first-round adjustments

| Original suggestion | Evidence/problem | Implementation | Affected comparison |
|---|---|---|---|
| Prefer ImageXY checkpoint | Actual checkpoint uses centred T13 gray windows; its official-val selection overlaps later holdouts | Reuse causal T5 V3 direct RGB encoder for all image-containing groups | All six; claims restricted to adding flow to this frozen representation |
| Existing V3 fusion as a possible simple baseline | V3/V4 fuse geometry and image speed predictions | New R01 pooled-flow feature projection and one shared speed head | R01 versus R03 |
| Current-yaw two-endpoint difference | Actual camera intervals alternate and are unequal; synthetic no-slip circle gets spurious vy | Nonuniform three-point derivative at current time, yaw-only PLANAR_PROXY | All six; new label version, no old optimizer state |
| V3 three cached flow pairs | Five images span four adjacent pairs | Four causal pairs and measured dt in every model | All flow comparisons |
| Reuse old confidence/cache | Native backward flow, explicit legality and full provenance missing | Versioned numeric flow; separate legal/FB masks; forward cache can start plain models before FB | R01--R05 |
| V4 RAFT resize | antialias=True differs from torchvision official RAFT example | Explicit antialias=False, 288x512 RAFT raster, separate 72x128 downstream raster | New flow cache only; never mix old caches |
| Existing history as independent test evidence | Historical stage1 evaluation and ImageXY official-val selection found | Keep prior results separate and holdouts sealed; disclose scene/log limits | Interpretation and later S4 |

The three-point label was approved as the first candidate and passed numerical,
reference-point, time and finite-value checks. A bounded sensitivity check still
shows differentiation amplifies position noise; the assumed 1 cm noise is not a
measured sensor error. This limits physical lateral-speed claims and motivates
reporting zero-vy and later independently validated labels.
