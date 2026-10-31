# Full nuScenes CAM_FRONT optical-flow cache

Status: **complete**.

This report covers preprocessing and storage. It contains no model-training or accuracy result.

| Scope | Groups/scenes | Frames | Adjacent pairs |
|---|---:|---:|---:|
| v1.0-test | 150 | 34924 | 34774 |
| v1.0-trainval | 850 | 198315 | 197465 |
| official_test_cache_only | 150 | 34924 | 34774 |
| train_core | 582 | 135888 | 135306 |
| final_system_test | 75 | 17482 | 17407 |
| dev | 103 | 23995 | 23892 |
| stage1_test | 75 | 17466 | 17391 |
| unassigned_cache_only | 15 | 3484 | 3469 |

Generation commit: `87d7468f911da5579bbbc39f79ec6bc85fa16dd6`.
Native + compact bytes present: 1,384,739,436,318; bytes written in this invocation: 1,384,739,436,318.
Generation wall time: 45649.8 s; wall throughput: 5.087 pairs/s.
Outcome counts: {'created': 222969, 'reused': 0, 'recovered': 0, 'adopted': 9270}
Packed original train_core/dev pairs: 158,513.
Packed array bytes hashed: 8,765,135,232; packed validation took 21.5 s.

Validation:

- The complete index, group checksums, every shard's ordered pair digest, and every expected packet's presence/size were checked.
- Fresh generation validates packet construction and binds native bytes by SHA; adoption/resume also checks source image provenance and exact native-to-compact derivation per pair.
- Final native/compact checksum and CPU tensor reload covers 18 distinct first/last samples; the entire native corpus was not read again.
- Every packed array checksum and the complete original train/dev pair set and timestamps were checked.
- Holdout, unassigned and official test caching does not authorize model selection or training on those scenes.

Approved reuse origins:
- `be327b7f91f5f7e5ba46b70dede116a21c551652`: /home/zhangyu/.local/share/speedest-v5-next/flows/20261030_pilot_s42_retry1_raft1024_native
