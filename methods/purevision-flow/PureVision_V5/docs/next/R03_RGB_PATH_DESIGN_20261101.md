# R03 RGB information path: one appearance bypass, 18 complete epochs

Authorized comparison: old `20261031_r03_flex_s42_R03_FT` first 18 complete epochs against one new `R03_APPEAR_FT_s42_18ep`. Old FT remains user-paused. Its `last.pt` contains 18 complete epochs plus 41 batches and is not an epoch-18 endpoint. Its best is epoch 1 (EPE 0.2067490669); independent train_core/dev evaluation uses that best only. No old training is resumed.

## Design and self-review

Actual CMA uses absolute adjacent-image differences with a zero first time slice, then two 3D convolutions/BN/ReLU. Its output is `[B,64,5,72,128]`. The new branch is a single `Conv3d(3,64,kernel=(1,3,3),padding=(0,1,1)) + ReLU` directly on normalized RGB `[B,3,5,72,128]`. It preserves all five ordered frames and static appearance, with temporal receptive field 1. It adds 1,793 parameters including a scalar alpha. No additional BN or spatial downsampling is introduced.

`M=CMA(images); A=appearance_stem(images); X=M+alpha*A; Y=AFT(x=X,motion=M)`. Existing SFE/MFE/DIG, regional projection, 384-dimensional descriptor, FLEX flow encoder and 452→192→2 concatenation head remain. This changes only the RGB information path; no new flow gates, masks, spatial weights or geometry losses.

Common parameters load exactly from the same seed42 R02 source and V3 RGB source used by old FT. The stem uses PyTorch's default Conv initialization under an isolated seed42 RNG; alpha is zero. Only the three known new tensors may be absent from V3 loading. Flow/dt head columns, biases and last layer copy the R02 head; RGB columns are zero. Alpha-zero RGB and common initial velocity predictions must match old initialization. RGB, new stem/alpha, flow and head all train; original BN train/eval policy remains.

`epochs=18` is the stopping budget, `scheduler_epochs=40` is the cosine period. AdamW, groups, AMP, clipping, batch64/no accumulation, sample order and loss match old FT. Verify each group's first18 epoch LR against old metrics, not merely a nominal config. Expected processed batches=37,404; optimizer updates/AMP skips are reported separately.

## Validation and publication

Finite real checks cover 64 windows across scenes, original-file versus pixel-cache alignment, static appearance, exact common initialization and alpha-zero outputs, continuous updates through the staged zero initialization, and complete state round trips. Existing trainer smoke checks are reused with only this candidate. Full original 582/103 scenes, 132978/23480 windows; no cache regeneration.

Independent best reloads evaluate both train_core and dev in eval mode/FP32 with batch8. Matching train/dev EPE and vx/vy errors distinguish fit from generalization; training SmoothL1 is not substituted for train EPE. Full per-window predictions, scene deltas, first18 learning curves, counters, model cost and lateral acceptance will be published. Side subset remains31 windows/5scenes and insufficient_data under100/5.

Risks: two zero initialization stages delay stem gradients; finite nonzero gradients plus actual updates must be observed after alpha moves. Adding appearance can overfit; no benefit is assumed. Old FT completed only the selected18-epoch budget; its partial epoch19 is excluded from learning/cost/update comparisons. Accuracy priority, single seed, dev selection is not held-out generalization.

## Verified implementation, 2026-11-01

Real preflight passed:64windows; alpha-zero descriptor exactly equal; initial velocity max error0; all shared pretrained tensors/head exact. Four accepted updates, no AMP skips: alpha0 →7.4402e-5 →1.36052e-4 →1.95268e-4; stem gradient first nonzero on third update. All common RGB modules, FLEX and head updated. Batch64 peak45,827,707,904 bytes; warm real steps approximately0.72–0.73seconds (preflight compute, not end-to-end training cost).

Trainer interruption at step3, full-state restoration to step5 and independent16-window eval passed in `verification/rgb_path_trainer_v2_20261101`. First smoke attempt exposed a checker coupling scheduler period to stopping budget; corrected to `scheduler_epochs`, retaining default behavior for existing configs. This was a verification issue; no formal run was started or restarted. Existing FT remains paused.

Operator plan contains exactly one training job on PRO6000/GPU0, one old-best dev+train_core evaluation job on GPU2, and a CPU comparison blocked until both jobs finish successfully. Both best checkpoints receive independent dev and train_core evaluation under the same new evaluation code. Formal initialization remains fresh from the common original sources.
