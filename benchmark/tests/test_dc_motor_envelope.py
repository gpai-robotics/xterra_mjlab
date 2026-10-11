"""Isaac Lab DCMotor torque-speed envelope declared by a policy manifest.

The go2_blind policy was trained against a DCMotor: 23.5 Nm at rest, falling
linearly to zero torque at 30 rad/s. Clipping only to the MJCF ctrlrange gives
it torque it never had in training.
"""

import pytest

from robustness_lab.observation.action import compile_action, dc_motor_torque

MOTOR = {"saturation_effort": 23.5, "effort_limit": 23.5, "velocity_limit": 30.0}


def _action(**extra):
    config = {
        "space": "joint_position_offset",
        "size": 2,
        "scale": 0.25,
        "offset": [0.0, 0.0],
        "stiffness": 25.0,
        "damping": 0.5,
    }
    config.update(extra)
    return config


def test_full_torque_at_rest_is_clipped_to_the_effort_limit():
    assert dc_motor_torque(100.0, 0.0, MOTOR) == pytest.approx(23.5)
    assert dc_motor_torque(-100.0, 0.0, MOTOR) == pytest.approx(-23.5)
    assert dc_motor_torque(10.0, 0.0, MOTOR) == pytest.approx(10.0)


def test_torque_in_the_direction_of_motion_droops_with_speed():
    # At 15 rad/s only half the saturation torque is left to push further.
    assert dc_motor_torque(100.0, 15.0, MOTOR) == pytest.approx(11.75)
    # Braking against the motion still gets the full effort limit.
    assert dc_motor_torque(-100.0, 15.0, MOTOR) == pytest.approx(-23.5)
    # At the velocity limit nothing more can be added in that direction.
    assert dc_motor_torque(100.0, 30.0, MOTOR) == pytest.approx(0.0)


def test_matches_isaac_lab_dc_motor_formula():
    for torque in (-40.0, -5.0, 0.0, 5.0, 40.0):
        for velocity in (-45.0, -30.0, -12.0, 0.0, 12.0, 30.0, 45.0):
            # DCMotor._clip_effort: each bound is also clamped at zero, so
            # past the velocity limit the motor cannot push the wrong way.
            upper = min(max(23.5 * (1.0 - velocity / 30.0), 0.0), 23.5)
            lower = max(min(23.5 * (-1.0 - velocity / 30.0), 0.0), -23.5)
            expected = min(max(torque, lower), upper)
            assert dc_motor_torque(torque, velocity, MOTOR) == pytest.approx(expected)


def test_manifest_motor_is_compiled_onto_the_decoder():
    decoder = compile_action(_action(motor=dict(MOTOR, model="dc_motor")))
    assert decoder.motor == MOTOR


def test_no_motor_means_plain_ctrlrange_clip():
    assert compile_action(_action()).motor is None


@pytest.mark.parametrize(
    "motor",
    [
        {"model": "ideal", **MOTOR},
        {"model": "dc_motor", "saturation_effort": 23.5, "effort_limit": 23.5},
        {"model": "dc_motor", "saturation_effort": 0.0, "effort_limit": 23.5, "velocity_limit": 30.0},
        "dc_motor",
    ],
)
def test_malformed_motor_is_refused(motor):
    with pytest.raises(ValueError, match="action.motor"):
        compile_action(_action(motor=motor))


def test_motor_without_gains_is_refused():
    config = _action(motor=dict(MOTOR, model="dc_motor"))
    del config["stiffness"], config["damping"]
    with pytest.raises(ValueError, match="action.motor"):
        compile_action(config)
