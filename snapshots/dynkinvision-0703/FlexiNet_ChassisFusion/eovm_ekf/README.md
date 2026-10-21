# EOVM EKF Reproduction

This directory reproduces the estimator core from:

Seo, D. and Kang, J. (2025), "Dynamic-model-free vehicle velocity estimation using extended Kalman filter with IMU, steering angle, and wheel speed sensors."

## Scope

Implemented:

- RVST/EOVM formula layer from Eq. (9)-(18) and Eq. (14)-(15).
- 13-state EKF prediction and measurement model from Eq. (19)-(25).
- Steering-angle linear Kalman filter used to obtain `delta` and `delta_dot`.
- A minimal replay example that feeds synthetic sensor samples through the filters.

Not implemented:

- The 7-DOF vehicle dynamics simulation from Eq. (28)-(30).
- Magic Formula tire model.
- MATLAB/Simulink or IPG CarMaker scenarios.
- KME/DME baseline estimators.

## State Order

The EKF state follows Eq. (19):

```text
[v_x, v_y, gamma, omega_1, omega_2, omega_3, omega_4, a_x, a_y, rho_1, rho_2, rho_3, rho_4]
```

The output velocity estimate is `v_x, v_y` from the updated EKF state.

## Sensor Input

Each `SensorSample` contains:

- `steering_angle`: steering angle sensor value.
- `yaw_rate`: IMU yaw rate measurement.
- `wheel_speeds`: four wheel angular speeds `[omega_1, omega_2, omega_3, omega_4]`.
- `ax`, `ay`: IMU acceleration in the body frame.

The measurement vector uses the paper's rear-wheel pseudo-measurement:

```text
z_vx = r_t * (z_omega3 + z_omega4) / 2
```

## Jacobian Note

Eq. (21) requires the Jacobian of the discrete prediction model. This implementation evaluates it with centered finite differences. During this computation, the load-transfer weights are held fixed with respect to `a_x` and `a_y`, matching the paper's simplification that `theta_i` are treated as parameters in the prediction step.

## Run

From the vault root:

```powershell
D:\Miniconda3\envs\speedXY_py310\python.exe -m pytest reproduction/eovm_ekf/tests -v
D:\Miniconda3\envs\speedXY_py310\python.exe reproduction/eovm_ekf/examples/minimal_replay.py
```

If `speedXY_py310` is available through `conda run`, the equivalent commands are:

```powershell
conda run -n speedXY_py310 python -m pytest reproduction/eovm_ekf/tests -v
conda run -n speedXY_py310 python reproduction/eovm_ekf/examples/minimal_replay.py
```

On this Windows workspace, direct interpreter execution avoids Conda's console encoding issue in paths containing non-ASCII characters.
