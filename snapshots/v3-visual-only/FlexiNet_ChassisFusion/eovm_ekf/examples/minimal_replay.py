from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eovm_ekf import EovmEkf, SensorSample, SteeringKalmanFilter  # noqa: E402


def main() -> None:
    sampling_time = 0.01
    steering_filter = SteeringKalmanFilter(sampling_time=sampling_time)
    estimator = EovmEkf(sampling_time=sampling_time)

    samples = [
        SensorSample(
            steering_angle=0.02 + 0.002 * index,
            yaw_rate=0.08 + 0.005 * index,
            wheel_speeds=np.array(
                [
                    39.0 + 0.1 * index,
                    39.2 + 0.1 * index,
                    38.8 + 0.1 * index,
                    39.0 + 0.1 * index,
                ]
            ),
            ax=0.2,
            ay=0.05,
        )
        for index in range(20)
    ]

    result = None
    for sample in samples:
        delta, delta_dot = steering_filter.step(sample.steering_angle)
        result = estimator.step(sample, delta=delta, delta_dot=delta_dot)

    if result is None:
        raise RuntimeError("No samples were processed.")

    print("Final EOVM EKF estimate")
    print(f"vx     : {result.vx:.6f} m/s")
    print(f"vy     : {result.vy:.6f} m/s")
    print(f"gamma  : {result.gamma:.6f} rad/s")
    print(f"delta  : {result.delta:.6f} rad")
    print(f"d_delta: {result.delta_dot:.6f} rad/s")


if __name__ == "__main__":
    main()
