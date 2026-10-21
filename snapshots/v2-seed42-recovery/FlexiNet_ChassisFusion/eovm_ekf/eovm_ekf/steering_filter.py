from __future__ import annotations

import numpy as np

from .parameters import SteeringFilterTuning


class SteeringKalmanFilter:
    """Linear Kalman filter for steering angle and steering-angle rate.

    The paper treats delta and dot(delta) as known values for the EOVM EKF
    after filtering the steering-angle sensor, as shown in Fig. 5.
    """

    def __init__(
        self,
        sampling_time: float,
        tuning: SteeringFilterTuning | None = None,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
    ) -> None:
        self.sampling_time = float(sampling_time)
        self.tuning = tuning or SteeringFilterTuning()
        self.state = (
            np.zeros(2, dtype=float)
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        if self.state.shape != (2,):
            raise ValueError("initial_state must have shape (2,).")

        self.covariance = (
            np.eye(2, dtype=float)
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=float).copy()
        )
        if self.covariance.shape != (2, 2):
            raise ValueError("initial_covariance must have shape (2, 2).")

    @property
    def transition_matrix(self) -> np.ndarray:
        return np.array(
            [[1.0, self.sampling_time], [0.0, 1.0]],
            dtype=float,
        )

    @property
    def process_noise(self) -> np.ndarray:
        return np.diag(
            [
                self.tuning.process_variance_delta,
                self.tuning.process_variance_delta_dot,
            ]
        )

    @property
    def measurement_matrix(self) -> np.ndarray:
        return np.array([[1.0, 0.0]], dtype=float)

    def predict(self) -> tuple[float, float]:
        f = self.transition_matrix
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + self.process_noise
        self.covariance = _symmetrize(self.covariance)
        return float(self.state[0]), float(self.state[1])

    def update(self, measured_delta: float) -> tuple[float, float]:
        h = self.measurement_matrix
        residual = np.array([float(measured_delta)], dtype=float) - h @ self.state
        innovation_covariance = (
            h @ self.covariance @ h.T + self.tuning.measurement_variance
        )
        gain = self.covariance @ h.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + gain @ residual
        self.covariance = self.covariance - gain @ h @ self.covariance
        self.covariance = _symmetrize(self.covariance)
        return float(self.state[0]), float(self.state[1])

    def step(self, measured_delta: float) -> tuple[float, float]:
        self.predict()
        return self.update(measured_delta)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
