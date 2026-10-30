# Dependency-state incident — 2026-10-30

The first real hourly Astra patrol (12:00 Shanghai) found eight queued jobs
marked `blocked`, while both forward-cache workers were alive and advancing.
The earliest retained downstream transition is `forward_pack` at 03:59:33 UTC,
with `blocked_by=["forward1"]`; `fb1` and dependent training jobs followed.

The 11:55 and 12:00 host snapshots and subsequent `/proc` reads agree on the same
wrapper PID/start time (3231712 / 84669323) and child PID/start time
(3231878 / 84669754). No persistent worker disappearance, restart, or PID reuse
was found. The original coordinator did not preserve its transient probe error,
so the precise low-level observation at 03:59:33 **cannot be established**.

Source review found a reproducible design defect: one negative process probe,
or a missing state interpreted as `interrupted`, can immediately and irreversibly
block downstream jobs. A live worker subsequently writes `running` again, hiding
the temporary cause while descendants remain blocked. Process-stat errors were
indistinguishable from confirmed exits. PID parsing and atomic temporary-file
names were checked; neither explains the evidence.

Only the coordinator was stopped after its PID/start identity was verified.
RGB and both RAFT workers continue under their unchanged fixed commits. No
formal training had started, so no trained model, loss, label or optimizer state
is affected. Recovery must use a newly pushed orchestration fix and new run IDs,
adopting the already-running preparation jobs through explicit parent-plan
provenance. It must not mutate either immutable checkout or regenerate identical
flow values merely to repair queue state.

The implemented fix distinguishes alive/dead/unknown observations, preserves
worker-owned status, and requires two consistent observations of absent worker,
child and tmux plus a released job lock before propagating an inferred exit.
Missing/invalid observations remain uncertain. Launch-time coordinator PID
backfilling was also removed to avoid overwriting a new worker heartbeat; this
additional race was found by review, not established as the original cause.

Recovery prefix: `20261030_s42_r1`. `--preparation-plan` adopts only the old two
forward jobs after matching artifact root, manifest and numerical command
arguments. The new plan contains ten locally executable jobs, including six
new training run IDs. Model/RGB/data/training/evaluation/flow code and all six
configurations remain byte-identical to `9b859c2`; this is an orchestration fix.
Exact recovery commit and commands are recorded in the new immutable plan.

An independent `startup_watch.py` verifies actual steps 1/2/3, first dev
validation, and CPU loading of full last/best recovery state. It then requests
one Astra patrol of that specific run. This watcher changes no training state,
continues alongside hourly patrols, and exits when all declared jobs terminate.

Evidence outside Git: `ART/monitoring/incidents/` and
`ART/monitoring/astra/20261030T040001Z_3237516.*`, where
`ART=/mnt/sdb2/Workspace/zhangyu/01_speedest_vision_artifacts/v5`.
