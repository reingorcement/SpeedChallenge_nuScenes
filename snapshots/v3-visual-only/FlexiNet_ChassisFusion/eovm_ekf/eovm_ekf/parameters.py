from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _paper_q() -> np.ndarray:
    return np.diag(
        [
            1.0e-4,
            1.0e-7,
            5.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            1.0e-4,
            5.0e-3,
            5.0e-3,
            1.0e-1,
            1.0e-1,
            1.0e-1,
            1.0e-1,
        ]
    )


def _paper_r() -> np.ndarray:
    return np.diag(
        [
            9.8e-3,
            3.046e-4,
            1.0,
            1.0,
            1.0,
            1.0,
            2.5e-1,
            2.5e-1,
        ]
    )


@dataclass(frozen=True)
class VehicleGeometry:
    """Vehicle geometry parameters used by Seo and Kang 2025.

    Defaults are from Table 2 of the paper.
    """

    l: float = 2.106
    w: float = 1.634
    tire_radius: float = 0.3205
    h: float = 0.654
    g: float = 9.81


@dataclass(frozen=True)
class EstimatorTuning:
    """Estimator tuning constants from Table 4 and Eq. (17)-(18)."""

    xi: float = 100.0
    k_theta: float = 11.0
    lambda_: float = 200.0
    lambda_prime: float = 8.0
    epsilon_a: float = 0.01
    q: np.ndarray = field(default_factory=_paper_q)
    r: np.ndarray = field(default_factory=_paper_r)

    def __post_init__(self) -> None:
        if self.q.shape != (13, 13):
            raise ValueError("Process noise covariance q must have shape (13, 13).")
        if self.r.shape != (8, 8):
            raise ValueError("Measurement noise covariance r must have shape (8, 8).")


@dataclass(frozen=True)
class SteeringFilterTuning:
    """Steering-angle Kalman filter noise values reported in Section 4.1."""

    process_variance_delta: float = 1.0e-4
    process_variance_delta_dot: float = 1.0
    measurement_variance: float = 0.3337
