# Figure Catalog

## Figure 01: Clean Existing-Checkpoint Comparison

- Files: `figures/figure-01-clean-comparison.pdf` and `.png`.
- Purpose: Describe clean EPE for all checkpoints present in the server run.
- Data: Mean scene EPE over 75 scenes; bars show scene-bootstrap 95% intervals.
- Observation: The old V2 visual checkpoint and current V2 joint checkpoint are not a valid generation-matched pair.
- Implication: Use this figure descriptively; do not claim a fair architecture ranking.

## Figure 02: Current Joint Fault Degradation

- Files: `figures/figure-02-joint-fault-degradation.pdf` and `.png`.
- Purpose: Compare each synthetic fault against the current joint clean baseline.
- Data: Paired scene fault-minus-clean EPE with 95% scene-bootstrap intervals.
- Observation: Camera removal dominates; CAN removal is also materially harmful; yaw and wheel-low changes are near zero.
- Implication: Camera fallback is the immediate design priority. Near-zero yaw/wheel effects require pathway audits, not robustness claims.

## Figure 03: Scene-Level Joint Degradation

- Files: `figures/figure-03-joint-scene-degradation.pdf` and `.png`.
- Purpose: Show heterogeneity of fault effects across 75 scenes.
- Data: Paired scene fault-minus-clean EPE distributions.
- Observation: Camera and CAN failures worsen many scenes; mild perturbations cluster near zero.
- Implication: Aggregate camera/CAN degradation is not caused by a single isolated scene.
