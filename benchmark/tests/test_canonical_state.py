"""Golden-value tests for the canonical state bus.

The constants here were derived by hand from the convention:
world gravity direction is [0, 0, -1]; the body-frame value is R_wb.T @ g_w.
A stub or placeholder implementation cannot reproduce them. If one of these
fails, the rotation convention is wrong -- most often R used instead of R.T,
or xyzw quaternion order assumed instead of wxyz.
"""

import math

import pytest

from robustness_lab.canonical_state import projected_gravity_from_quaternion


def test_level_base_projects_gravity_straight_down():
    assert projected_gravity_from_quaternion([1.0, 0.0, 0.0, 0.0]) == pytest.approx(
        [0.0, 0.0, -1.0], abs=1e-9)


def test_thirty_degree_pitch():
    half = math.radians(30.0) / 2.0
    result = projected_gravity_from_quaternion([math.cos(half), 0.0, math.sin(half), 0.0])
    assert result == pytest.approx([0.5, 0.0, -0.8660254037844387], abs=1e-9)


def test_ninety_degree_roll():
    half = math.radians(90.0) / 2.0
    result = projected_gravity_from_quaternion([math.cos(half), math.sin(half), 0.0, 0.0])
    assert result == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


def test_result_is_always_unit_length():
    result = projected_gravity_from_quaternion([0.8079, 0.3106, 0.4132, 0.2807])
    assert math.sqrt(sum(v * v for v in result)) == pytest.approx(1.0, abs=1e-6)


def test_yaw_alone_does_not_change_projected_gravity():
    """Gravity is yaw-invariant. Catches an xyzw/wxyz mix-up that happens to
    pass the level case."""
    half = math.radians(37.0) / 2.0
    result = projected_gravity_from_quaternion([math.cos(half), 0.0, 0.0, math.sin(half)])
    assert result == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
