import numpy as np

from eovm_ekf.estimator import EovmEkf
from eovm_ekf.parameters import VehicleGeometry
from eovm_ekf.steering_filter import SteeringKalmanFilter
from eovm_ekf.types import SensorSample


def test_steering_filter_tracks_constant_angle_measurement():
    steering_filter = SteeringKalmanFilter(sampling_time=0.01)

    for _ in range(40):
        delta, delta_dot = steering_filter.step(0.12)

    assert np.isclose(delta, 0.12, atol=2.0e-2)
    assert abs(delta_dot) < 0.2
    assert steering_filter.covariance.shape == (2, 2)
    np.testing.assert_allclose(
        steering_filter.covariance,
        steering_filter.covariance.T,
        atol=1.0e-12,
    )


def test_measurement_matrix_matches_paper_eq_22_state_selection():
    h = EovmEkf.measurement_matrix()

    assert h.shape == (8, 13)
    np.testing.assert_array_equal(h[0], np.eye(13)[0])
    np.testing.assert_array_equal(h[1], np.eye(13)[2])
    np.testing.assert_array_equal(h[2:6], np.eye(13)[3:7])
    np.testing.assert_array_equal(h[6:8], np.eye(13)[7:9])


def test_measurement_vector_uses_rear_wheel_pseudo_vx_from_paper_eq_22_note():
    geometry = VehicleGeometry()
    sample = SensorSample(
        steering_angle=0.05,
        yaw_rate=0.3,
        wheel_speeds=np.array([10.0, 12.0, 20.0, 22.0]),
        ax=1.5,
        ay=-0.2,
    )

    z = EovmEkf.measurement_vector(sample, geometry)

    expected_vx = geometry.tire_radius * (20.0 + 22.0) / 2.0
    np.testing.assert_allclose(
        z,
        np.array([expected_vx, 0.3, 10.0, 12.0, 20.0, 22.0, 1.5, -0.2]),
    )


def test_ekf_step_returns_valid_state_and_symmetric_covariance():
    estimator = EovmEkf(sampling_time=0.01)
    sample = SensorSample(
        steering_angle=0.08,
        yaw_rate=0.15,
        wheel_speeds=np.array([42.0, 41.5, 40.0, 40.5]),
        ax=0.4,
        ay=0.1,
    )

    result = estimator.step(sample, delta=0.08, delta_dot=0.0)

    assert result.state.shape == (13,)
    assert result.covariance.shape == (13, 13)
    assert np.all(np.isfinite(result.state))
    assert np.all(np.isfinite(result.covariance))
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1.0e-10)
    assert np.isclose(result.vx, result.state[0])
    assert np.isclose(result.vy, result.state[1])
    assert np.isclose(result.gamma, result.state[2])
