import numpy as np

from eovm_ekf.eovm import (
    build_eovm_matrices,
    constraint_matrices,
    constraint_matrix_derivatives,
    load_transfer_weights,
    switching_epsilon,
)
from eovm_ekf.parameters import EstimatorTuning, VehicleGeometry


def test_constraint_matrices_match_paper_eq_9_to_11():
    geometry = VehicleGeometry()
    tuning = EstimatorTuning()
    delta = 0.2

    matrices = constraint_matrices(delta, geometry, tuning)

    expected_epsilon = np.exp(-tuning.xi * delta**2)
    assert np.isclose(switching_epsilon(delta, tuning), expected_epsilon)

    expected_dv = np.array(
        [
            [np.cos(delta), np.sin(delta)],
            [np.cos(delta), np.sin(delta)],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, expected_epsilon],
        ]
    )
    expected_domega = geometry.tire_radius * np.vstack(
        [np.eye(4), np.zeros((1, 4))]
    )
    expected_dgamma = 0.5 * np.array(
        [
            geometry.w * np.cos(delta) - geometry.l * np.sin(delta),
            -geometry.w * np.cos(delta) - geometry.l * np.sin(delta),
            geometry.w,
            -geometry.w,
            geometry.l * expected_epsilon,
        ]
    )

    np.testing.assert_allclose(matrices.dv, expected_dv)
    np.testing.assert_allclose(matrices.domega, expected_domega)
    np.testing.assert_allclose(matrices.dgamma, expected_dgamma)


def test_constraint_derivatives_match_paper_eq_13_terms():
    geometry = VehicleGeometry()
    tuning = EstimatorTuning()
    delta = -0.12
    delta_dot = 0.35

    derivatives = constraint_matrix_derivatives(delta, delta_dot, geometry, tuning)
    eps = np.exp(-tuning.xi * delta**2)
    eps_dot = -2.0 * tuning.xi * delta * delta_dot * eps

    expected_dv_dot = np.array(
        [
            [-np.sin(delta) * delta_dot, np.cos(delta) * delta_dot],
            [-np.sin(delta) * delta_dot, np.cos(delta) * delta_dot],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, eps_dot],
        ]
    )
    expected_dgamma_dot = 0.5 * np.array(
        [
            -geometry.w * np.sin(delta) * delta_dot
            - geometry.l * np.cos(delta) * delta_dot,
            geometry.w * np.sin(delta) * delta_dot
            - geometry.l * np.cos(delta) * delta_dot,
            0.0,
            0.0,
            geometry.l * eps_dot,
        ]
    )

    np.testing.assert_allclose(derivatives.dv_dot, expected_dv_dot)
    np.testing.assert_allclose(derivatives.dgamma_dot, expected_dgamma_dot)


def test_load_transfer_weights_match_paper_eq_16_to_18():
    geometry = VehicleGeometry()
    tuning = EstimatorTuning()
    ax = 1.2
    ay = -0.7

    weights = load_transfer_weights(ax, ay, geometry, tuning)

    expected_normal_accel = np.array(
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
        ]
    )
    expected_eta = expected_normal_accel / max(
        tuning.epsilon_a, *expected_normal_accel
    )
    expected_theta = np.exp(tuning.k_theta * expected_eta)

    np.testing.assert_allclose(weights.normal_accel, expected_normal_accel)
    np.testing.assert_allclose(weights.eta, expected_eta)
    np.testing.assert_allclose(weights.theta, expected_theta)
    assert np.isclose(weights.theta_prime, np.e)


def test_eovm_matrices_have_paper_dimensions_and_finite_values():
    geometry = VehicleGeometry()
    tuning = EstimatorTuning()

    matrices = build_eovm_matrices(
        gamma=0.18,
        delta=0.15,
        delta_dot=-0.25,
        ax=0.8,
        ay=0.4,
        geometry=geometry,
        tuning=tuning,
    )

    assert matrices.m.shape == (7, 3)
    assert matrices.a.shape == (7, 3)
    assert matrices.b_omega.shape == (7, 4)
    assert matrices.b_accel.shape == (7, 2)
    assert matrices.b_rho.shape == (7, 4)
    assert matrices.a_hat.shape == (3, 3)
    assert matrices.b_omega_hat.shape == (3, 4)
    assert matrices.b_accel_hat.shape == (3, 2)
    assert matrices.b_rho_hat.shape == (3, 4)
    assert matrices.g.shape == (13, 13)

    for value in (
        matrices.m,
        matrices.a,
        matrices.b_omega,
        matrices.b_accel,
        matrices.b_rho,
        matrices.g,
    ):
        assert np.all(np.isfinite(value))
