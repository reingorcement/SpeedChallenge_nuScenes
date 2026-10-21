from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .parameters import EstimatorTuning, VehicleGeometry


@dataclass(frozen=True)
class ConstraintMatrices:
    dv: np.ndarray
    domega: np.ndarray
    dgamma: np.ndarray


@dataclass(frozen=True)
class ConstraintDerivatives:
    dv_dot: np.ndarray
    dgamma_dot: np.ndarray


@dataclass(frozen=True)
class LoadTransferWeights:
    normal_accel: np.ndarray
    eta: np.ndarray
    theta: np.ndarray
    theta_prime: float
    s: np.ndarray
    lambda_matrix: np.ndarray
    pi: np.ndarray


@dataclass(frozen=True)
class EovmMatrices:
    m: np.ndarray
    a: np.ndarray
    b_omega: np.ndarray
    b_accel: np.ndarray
    b_rho: np.ndarray
    a_hat: np.ndarray
    b_omega_hat: np.ndarray
    b_accel_hat: np.ndarray
    b_rho_hat: np.ndarray
    g: np.ndarray


def switching_epsilon(delta: float, tuning: EstimatorTuning) -> float:
    """Switching function epsilon(delta), paper Eq. (9)."""

    return float(np.exp(-tuning.xi * delta**2))


def constraint_matrices(
    delta: float,
    geometry: VehicleGeometry,
    tuning: EstimatorTuning,
) -> ConstraintMatrices:
    """Build D_v, D_omega, and d_gamma from paper Eq. (10)-(11)."""

    c_delta = np.cos(delta)
    s_delta = np.sin(delta)
    epsilon = switching_epsilon(delta, tuning)

    dv = np.array(
        [
            [c_delta, s_delta],
            [c_delta, s_delta],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, epsilon],
        ],
        dtype=float,
    )
    domega = geometry.tire_radius * np.vstack(
        [np.eye(4, dtype=float), np.zeros((1, 4), dtype=float)]
    )
    dgamma = 0.5 * np.array(
        [
            geometry.w * c_delta - geometry.l * s_delta,
            -geometry.w * c_delta - geometry.l * s_delta,
            geometry.w,
            -geometry.w,
            geometry.l * epsilon,
        ],
        dtype=float,
    )

    return ConstraintMatrices(dv=dv, domega=domega, dgamma=dgamma)


def constraint_matrix_derivatives(
    delta: float,
    delta_dot: float,
    geometry: VehicleGeometry,
    tuning: EstimatorTuning,
) -> ConstraintDerivatives:
    """Build dot(D_v) and dot(d_gamma) terms used by paper Eq. (13)."""

    c_delta = np.cos(delta)
    s_delta = np.sin(delta)
    epsilon = switching_epsilon(delta, tuning)
    epsilon_dot = -2.0 * tuning.xi * delta * delta_dot * epsilon

    dv_dot = np.array(
        [
            [-s_delta * delta_dot, c_delta * delta_dot],
            [-s_delta * delta_dot, c_delta * delta_dot],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, epsilon_dot],
        ],
        dtype=float,
    )
    dgamma_dot = 0.5 * np.array(
        [
            -geometry.w * s_delta * delta_dot
            - geometry.l * c_delta * delta_dot,
            geometry.w * s_delta * delta_dot
            - geometry.l * c_delta * delta_dot,
            0.0,
            0.0,
            geometry.l * epsilon_dot,
        ],
        dtype=float,
    )

    return ConstraintDerivatives(dv_dot=dv_dot, dgamma_dot=dgamma_dot)


def load_transfer_weights(
    ax: float,
    ay: float,
    geometry: VehicleGeometry,
    tuning: EstimatorTuning,
) -> LoadTransferWeights:
    """Load transfer ratio and weights from paper Eq. (16)-(18)."""

    normal_accel = np.array(
        [
            geometry.g / 4.0
            - geometry.h / (2.0 * geometry.l) * ax
            - geometry.h / (2.0 * geometry.w) * ay,
            geometry.g / 4.0
            - geometry.h / (2.0 * geometry.l) * ax
            + geometry.h / (2.0 * geometry.w) * ay,
            geometry.g / 4.0
            + geometry.h / (2.0 * geometry.l) * ax
            - geometry.h / (2.0 * geometry.w) * ay,
            geometry.g / 4.0
            + geometry.h / (2.0 * geometry.l) * ax
            + geometry.h / (2.0 * geometry.w) * ay,
        ],
        dtype=float,
    )
    denominator = max(tuning.epsilon_a, *normal_accel)
    eta = normal_accel / denominator
    theta = np.exp(tuning.k_theta * eta)
    theta_prime = float(np.e)
    s = np.diag(np.append(theta, theta_prime))
    lambda_matrix = np.diag(
        [
            tuning.lambda_,
            tuning.lambda_,
            tuning.lambda_,
            tuning.lambda_,
            tuning.lambda_prime,
        ]
    )
    pi = np.linalg.solve(s, lambda_matrix)

    return LoadTransferWeights(
        normal_accel=normal_accel,
        eta=eta,
        theta=theta,
        theta_prime=theta_prime,
        s=s,
        lambda_matrix=lambda_matrix,
        pi=pi,
    )


def gamma_matrix(gamma: float) -> np.ndarray:
    """Yaw-rate skew matrix Gamma from paper Eq. (2)."""

    return np.array([[0.0, -gamma], [gamma, 0.0]], dtype=float)


def build_eovm_matrices(
    gamma: float,
    delta: float,
    delta_dot: float,
    ax: float,
    ay: float,
    geometry: VehicleGeometry,
    tuning: EstimatorTuning,
) -> EovmMatrices:
    """Build paper Eq. (14)-(15) and embed it in the Eq. (19) G matrix."""

    constraints = constraint_matrices(delta, geometry, tuning)
    derivatives = constraint_matrix_derivatives(delta, delta_dot, geometry, tuning)
    weights = load_transfer_weights(ax, ay, geometry, tuning)

    dgamma_col = constraints.dgamma.reshape(5, 1)
    dgamma_dot_col = derivatives.dgamma_dot.reshape(5, 1)

    m = np.vstack(
        [
            np.hstack([constraints.dv, -dgamma_col]),
            np.hstack([np.eye(2, dtype=float), np.zeros((2, 1), dtype=float)]),
        ]
    )
    a = np.vstack(
        [
            np.hstack(
                [
                    -derivatives.dv_dot - weights.pi @ constraints.dv,
                    dgamma_dot_col + weights.pi @ dgamma_col,
                ]
            ),
            np.hstack(
                [
                    -gamma_matrix(gamma),
                    np.zeros((2, 1), dtype=float),
                ]
            ),
        ]
    )
    b_omega = np.vstack(
        [
            weights.pi @ constraints.domega,
            np.zeros((2, 4), dtype=float),
        ]
    )
    b_accel = np.vstack(
        [
            np.zeros((5, 2), dtype=float),
            np.eye(2, dtype=float),
        ]
    )
    b_rho = np.vstack(
        [
            constraints.domega,
            np.zeros((2, 4), dtype=float),
        ]
    )

    m_pinv = np.linalg.pinv(m)
    a_hat = m_pinv @ a
    b_omega_hat = m_pinv @ b_omega
    b_accel_hat = m_pinv @ b_accel
    b_rho_hat = m_pinv @ b_rho
    g_matrix = assemble_state_matrix(
        a_hat=a_hat,
        b_omega_hat=b_omega_hat,
        b_accel_hat=b_accel_hat,
        b_rho_hat=b_rho_hat,
    )

    return EovmMatrices(
        m=m,
        a=a,
        b_omega=b_omega,
        b_accel=b_accel,
        b_rho=b_rho,
        a_hat=a_hat,
        b_omega_hat=b_omega_hat,
        b_accel_hat=b_accel_hat,
        b_rho_hat=b_rho_hat,
        g=g_matrix,
    )


def assemble_state_matrix(
    a_hat: np.ndarray,
    b_omega_hat: np.ndarray,
    b_accel_hat: np.ndarray,
    b_rho_hat: np.ndarray,
) -> np.ndarray:
    """Assemble the 13-by-13 continuous prediction matrix G, paper Eq. (19)."""

    g_matrix = np.zeros((13, 13), dtype=float)
    g_matrix[0:3, 0:3] = a_hat
    g_matrix[0:3, 3:7] = b_omega_hat
    g_matrix[0:3, 7:9] = b_accel_hat
    g_matrix[0:3, 9:13] = b_rho_hat
    g_matrix[3:7, 9:13] = np.eye(4, dtype=float)
    return g_matrix
