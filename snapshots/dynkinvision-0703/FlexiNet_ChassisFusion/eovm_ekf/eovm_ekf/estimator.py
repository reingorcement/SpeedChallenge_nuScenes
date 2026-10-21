from __future__ import annotations

import numpy as np

from .eovm import build_eovm_matrices
from .parameters import EstimatorTuning, VehicleGeometry
from .types import EstimateResult, SensorSample


class EovmEkf:
    """EOVM-based 13-state EKF from Seo and Kang 2025.

    State order follows paper Eq. (19):
    [v_x, v_y, gamma, omega_1..omega_4, a_x, a_y, rho_1..rho_4].
    """

    def __init__(
        self,
        sampling_time: float,
        geometry: VehicleGeometry | None = None,
        tuning: EstimatorTuning | None = None,
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        jacobian_step: float = 1.0e-5,
    ) -> None:
        self.sampling_time = float(sampling_time)
        self.geometry = geometry or VehicleGeometry()
        self.tuning = tuning or EstimatorTuning()
        self.jacobian_step = float(jacobian_step)
        self.state = (
            np.zeros(13, dtype=float)
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        if self.state.shape != (13,):
            raise ValueError("initial_state must have shape (13,).")

        self.covariance = (
            np.eye(13, dtype=float)
            if initial_covariance is None
            else np.asarray(initial_covariance, dtype=float).copy()
        )
        if self.covariance.shape != (13, 13):
            raise ValueError("initial_covariance must have shape (13, 13).")

    @staticmethod
    def measurement_matrix() -> np.ndarray:
        """Measurement matrix H from paper Eq. (22)."""

        h = np.zeros((8, 13), dtype=float)
        h[0, 0] = 1.0
        h[1, 2] = 1.0
        h[2:6, 3:7] = np.eye(4, dtype=float)
        h[6:8, 7:9] = np.eye(2, dtype=float)
        return h

    @staticmethod
    def measurement_vector(
        sample: SensorSample,
        geometry: VehicleGeometry,
    ) -> np.ndarray:
        """Measurement vector z_k from paper Eq. (22).

        The pseudo-measurement z_vx is computed from rear wheel speeds:
        z_vx = r_t * (z_omega3 + z_omega4) / 2.
        """

        wheel_speeds = sample.wheel_speed_array()
        z_vx = geometry.tire_radius * (wheel_speeds[2] + wheel_speeds[3]) / 2.0
        return np.array(
            [
                z_vx,
                sample.yaw_rate,
                *wheel_speeds,
                sample.ax,
                sample.ay,
            ],
            dtype=float,
        )

    def transition(
        self,
        state: np.ndarray,
        delta: float,
        delta_dot: float,
        weight_accel: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Discrete prediction model from paper Eq. (20)."""

        state = np.asarray(state, dtype=float)
        if state.shape != (13,):
            raise ValueError("state must have shape (13,).")

        ax, ay = weight_accel if weight_accel is not None else (state[7], state[8])
        matrices = build_eovm_matrices(
            gamma=float(state[2]),
            delta=float(delta),
            delta_dot=float(delta_dot),
            ax=float(ax),
            ay=float(ay),
            geometry=self.geometry,
            tuning=self.tuning,
        )
        return (np.eye(13, dtype=float) + self.sampling_time * matrices.g) @ state

    def transition_jacobian(
        self,
        state: np.ndarray,
        delta: float,
        delta_dot: float,
    ) -> np.ndarray:
        """Numerical Jacobian F for paper Eq. (21).

        The load-transfer weights are held fixed with respect to a_x and a_y,
        matching the paper's simplification that theta_i are parameters during
        the prediction step.
        """

        state = np.asarray(state, dtype=float)
        fixed_accel = (float(state[7]), float(state[8]))
        jacobian = np.zeros((13, 13), dtype=float)

        for column in range(13):
            step = np.zeros(13, dtype=float)
            step[column] = self.jacobian_step
            forward = self.transition(
                state + step,
                delta,
                delta_dot,
                weight_accel=fixed_accel,
            )
            backward = self.transition(
                state - step,
                delta,
                delta_dot,
                weight_accel=fixed_accel,
            )
            jacobian[:, column] = (forward - backward) / (2.0 * self.jacobian_step)

        return jacobian

    def predict(self, delta: float, delta_dot: float) -> np.ndarray:
        f = self.transition_jacobian(self.state, delta, delta_dot)
        self.state = self.transition(self.state, delta, delta_dot)
        self.covariance = f @ self.covariance @ f.T + self.tuning.q
        self.covariance = _symmetrize(self.covariance)
        return self.state.copy()

    def update(self, sample: SensorSample) -> np.ndarray:
        h = self.measurement_matrix()
        z = self.measurement_vector(sample, self.geometry)
        residual = z - h @ self.state
        innovation_covariance = h @ self.covariance @ h.T + self.tuning.r
        gain = self.covariance @ h.T @ np.linalg.inv(innovation_covariance)

        self.state = self.state + gain @ residual
        self.covariance = self.covariance - gain @ h @ self.covariance
        self.covariance = _symmetrize(self.covariance)
        return self.state.copy()

    def step(
        self,
        sample: SensorSample,
        delta: float | None = None,
        delta_dot: float = 0.0,
    ) -> EstimateResult:
        delta_value = sample.steering_angle if delta is None else float(delta)
        self.predict(delta_value, delta_dot)
        self.update(sample)
        return EstimateResult(
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            vx=float(self.state[0]),
            vy=float(self.state[1]),
            gamma=float(self.state[2]),
            delta=delta_value,
            delta_dot=float(delta_dot),
        )


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)
