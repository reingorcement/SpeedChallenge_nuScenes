# S2 implementation verification — 2026-10-30

These checks establish executable interfaces, not scientific performance.

Additional real packed-cache integration used four train and four dev windows:
R03/R04/R05 each completed 20 CPU steps, and independent evaluation matched the
selected metrics and prediction CSV exactly. Forward/FB packing preserved all
14 numerical flow/valid pairs exactly. Full recovery states loaded strictly.
Evidence: `smoke/packed_real_cpu_20261030/verification.json`; these remain smoke
results and are excluded from formal reports.

- Six model paths passed CPU forward/backward and state roundtrip checks. R03--R05
  have identical parameter shapes; with all-reliable masks they agree exactly.
  R04/R05 differ only in masking unreliable numerical flow.
- Actual nuScenes RGB descriptors and eight real flow pairs per train/dev scene
  were used for GPU integration. R00--R05 each completed a 20-step smoke run and
  saved last/best checkpoints. These tiny samples are not formal results.
- Interrupted CPU training resumed to the same two-step model, optimizer,
  scheduler, scaler, random state and predictions as uninterrupted training.
- RAFT direction/warp checks, native-to-model scaling, invalid values, mask
  semantics and exact forward reuse in FB caches passed. Packed mmap flow tests
  verify bit-for-bit copying and reject mismatched provenance or mutation.
- Scheduling tests exercise independent preparation, true versus ordering-only
  dependencies, GPU occupancy, terminal failures, idempotency and output checks.
- A real cron-triggered gpt-6-astra probe read host artifacts and saved its report
  successfully at 11:24 Shanghai time. Formal schedule installation is recorded
  separately, after jobs have been registered.

Evidence lives outside Git under
`/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5/`:
`smoke/R00_gpu_v1` through `smoke/R05_gpu_v1`, `smoke/integration_forward`,
`smoke/integration_bidirectional`, and `monitoring/astra/`. The source command is:

```bash
/home/zhangyu/.conda/envs/speedest/bin/python -B -m pytest -q -p no:cacheprovider tests
```

The first full feature launch from `eb2b268` failed during remote Git HTTPS
verification, before descriptor computation. Its logs remain. The bounded SSH
verification fix was pushed as `847abbe`; a new, separately identified feature
job uses that fixed detached checkout. No running code was edited.

Formal experiment state and metrics must come from run/evaluation records;
successful smoke tests or queued jobs cannot fill the results table.
