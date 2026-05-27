# Notes: Pure Vision nuScenes Evaluation

## Findings
- `FlexiNet-main` originally only supported KITTI and nuImages testing in `test.py`.
- The pure-vision comparison must load `pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth` into the original `FlexiNet` architecture.
- The validation x-axis must be the ordered validation sample index from `DataLoader(shuffle=False)`.
- To align with `FlexiNet_ChassisFusion`, the pure-vision dataset filters validation scenes by CAN bus availability but does not use CAN features as model input.
- The default local Python interpreter does not currently have `torch` installed, so runtime loading could not be verified in this environment. Syntax checks passed.

## Expected Alignment Check
- Run pure-vision eval and confirm the sample count matches the fusion validation result, expected `33148` for the current fusion output.
- Compare the first `scene_name`, `center_token`, and `center_timestamp` rows in both prediction CSVs.
