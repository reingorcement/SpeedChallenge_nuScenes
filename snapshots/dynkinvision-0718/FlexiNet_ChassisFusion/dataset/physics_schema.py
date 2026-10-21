from __future__ import annotations


PHYSICS_INPUT_FEATURES = [
    "base_vx",
    "yaw_rate",
    "steering_command",
    "steering_feedback",
    "longitudinal_accel",
    "lateral_accel",
    "wheel_speed_fl",
    "wheel_speed_fr",
    "wheel_speed_rl",
    "wheel_speed_rr",
    "wheel_torque_fl",
    "wheel_torque_fr",
    "wheel_torque_rl",
    "wheel_torque_rr",
    "tire_fx_fl",
    "tire_fx_fr",
    "tire_fx_rl",
    "tire_fx_rr",
    "normal_load_fl",
    "normal_load_fr",
    "normal_load_rl",
    "normal_load_rr",
    "reverse_enable",
    "rear_drive_enable",
    "road_friction",
]
PHYSICS_INPUT_DIM = len(PHYSICS_INPUT_FEATURES)
PHYSICS_INPUT_INDEX = {name: index for index, name in enumerate(PHYSICS_INPUT_FEATURES)}

PHYSICS_FEATURE_NAMES = [
    "vx_predict",
    "vx_wheel_observation",
    "vx_estimate",
    "vy_kinematic",
    "vy_dynamic",
    "vy_estimate",
    "sideslip_angle",
    "yaw_rate",
    "steering_angle",
    "ax",
    "ay",
    "wheel_speed_mean",
    "wheel_speed_std",
    "slip_score",
    "vx_innovation",
    "lateral_excitation",
    "confidence_vx",
    "confidence_vy",
]
PHYSICS_FEATURE_DIM = len(PHYSICS_FEATURE_NAMES)

FAIR_PHYSICS_FEATURE_COUNT = 10
PHYSICS_EXTRA_FEATURES = PHYSICS_INPUT_FEATURES[FAIR_PHYSICS_FEATURE_COUNT:]
