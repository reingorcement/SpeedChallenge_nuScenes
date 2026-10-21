# Figure Catalog

## Figure 01: Clean Comparison

- Files: `figures/figure-01-clean-comparison.pdf` and `.png`.
- Purpose: Compare clean endpoint error for base velocity, KD, and VKD.
- Data: Mean scene EPE over 75 scenes; error bars are 95% scene-bootstrap intervals.
- Observation: Both learned models outperform base velocity; VKD and KD intervals overlap strongly.
- Implication: Existing checkpoints support a small VKD trend, not a decisive clean-set advantage.

## Figure 02: Paired Scene Deltas

- Files: `figures/figure-02-paired-scene-deltas.pdf` and `.png`.
- Purpose: Show the distribution of scene-level VKD-KD differences for every variant.
- Data: 75 paired scene means per variant; negative values favor VKD.
- Observation: Deltas cluster close to zero and contain mixed scene-level outcomes.
- Implication: Aggregate improvements are small and heterogeneous across scenes.

## Figure 03: Stress Degradation

- Files: `figures/figure-03-stress-degradation.pdf` and `.png`.
- Purpose: Separate ordinary synthetic perturbations from catastrophic complete CAN removal.
- Data: Scene-level variant-minus-clean degradation; error bars are 95% scene-bootstrap intervals.
- Observation: Wheel scaling causes modest degradation; camera removal modestly degrades VKD; CAN removal dominates all other faults.
- Implication: The retained VKD checkpoint does not provide meaningful CAN-failure robustness.
