# Task Plan: VKD Fusion Refactor

## Goal
Update `FlexiNet_ChassisFusion` to support vision, kinematic, and dynamics branch fusion while preserving the existing velocity-XY residual pipeline.

## Phases
- [x] Phase 1: Read the provided refactor plan and inspect project structure
- [x] Phase 2: Read required context docs and core code paths
- [x] Phase 3: Extract implementation scope, risks, and acceptance criteria
- [x] Phase 4: Present staged execution plan for user approval
- [x] Phase 5: Implement approved stages with tests and verification

## Key Questions
1. Does the current code already implement any part of `residual_gated` or branch flags?
2. Are the plan's assumptions about dataset outputs, chassis feature order, and model APIs consistent with code?
3. Which changes are safe for a first minimal implementation without altering data format or dependencies?

## Decisions Made
- Use the existing velocity-XY batch contract as the compatibility boundary.
- Do not modify production code before the user approves the staged execution plan.
- Keep the first implementation pass compatible with default `fusion_mode: residual`.
- Prefer model-internal dynamics proxy construction over adding a new dataset batch field.
- User approved implementing Stages 1-4 and deferring the knowledge loss scaffold.
- Stage 1/2 are implemented with passing targeted tests for branch switches and `residual_gated` gate behavior.
- Stage 3 is implemented with five new VKD ablation/smoke configs and passing config-build tests.
- Stage 4 is implemented with passing tests for proxy scenario metrics, prediction CSV proxy/gate columns, and postprocess scenario aggregation.
- Review feedback was addressed: dynamics proxy feature names now match the 8-D model input, and `save_gate_weights` controls gate retention.
- Final verification passed: 27 pytest tests, compileall, default smoke, flow smoke, and diff whitespace check.

## Errors Encountered
- PowerShell rejected `&&` in the git worktree verification command. Resolution: rerun with PowerShell-compatible command separators.
- `git diff --check` found trailing whitespace in generated patch lines. Resolution: strip trailing whitespace and rerun checks.

## Status
**Completed** - Approved Stages 1-4 are implemented and verified; knowledge loss scaffold remains deferred by decision.
