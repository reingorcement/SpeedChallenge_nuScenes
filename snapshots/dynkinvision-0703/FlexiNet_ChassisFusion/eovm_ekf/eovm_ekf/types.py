from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorSample:
    """Sensor values used by the paper's measurement vector z_k."""

    steering_angle: float
    yaw_rate: float
    wheel_speeds: np.ndarray
    ax: float
    ay: float

    def wheel_speed_array(self) -> np.ndarray:
        values = np.asarray(self.wheel_speeds, dtype=float)
        if values.shape != (4,):
            raise ValueError("wheel_speeds must contain four wheel angular speeds.")
        return values


@dataclass(frozen=True)
class EstimateResult:
    """Updated EKF state and commonly used vehicle estimates."""

    state: np.ndarray
    covariance: np.ndarray
    vx: float
    vy: float
    gamma: float
    delta: float
    delta_dot: float
