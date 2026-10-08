"""A declared size must be checked AFTER transforms, not before.

Size was validated against the raw canonical field, so every size-changing
transform was unusable:

    {"source": "velocity_command", "size": 2,
     "transform": [{"select": {"start": 0, "count": 2}}]}
    -> ValueError: canonical field 'velocity_command' has length 3,
       declared size is 2

That is the transform's whole purpose. A researcher whose policy takes only
(vx, wz), or who pads a term to a fixed width, could not express it -- they
would have to change our code. `size` describes what the NETWORK receives, so
it must be measured at the end of the chain.
"""

import pytest


def test_select_narrows_a_source_to_the_declared_size():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2,
                           "transform": [{"select": {"start": 0, "count": 2}}]}])
    assert plan.size == 2
    assert plan.build({"velocity_command": [1.0, 2.0, 3.0]}) == pytest.approx([1.0, 2.0])


def test_select_by_index_list_also_narrows():
    """A policy taking only forward velocity and yaw rate: indices 0 and 2."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2,
                           "transform": [{"select": [0, 2]}]}])
    assert plan.build({"velocity_command": [1.0, 2.0, 3.0]}) == pytest.approx([1.0, 3.0])


def test_pad_widens_a_source_to_the_declared_size():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 5,
                           "transform": [{"pad": {"size": 5, "value": 0.0}}]}])
    assert plan.build({"velocity_command": [1.0, 2.0, 3.0]}) == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0])


def test_repeat_expands_to_the_declared_size():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 6,
                           "transform": [{"repeat": 2}]}])
    assert plan.size == 6


def test_one_hot_changes_width_and_is_accepted():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "gait_clock", "size": 4,
                           "transform": [{"one_hot": {"classes": 4}}]}])
    assert plan.size == 4


def test_size_still_rejects_a_genuine_mismatch():
    """Flexibility must not become silence: a chain that does NOT reach the
    declared size is still an error."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2,
                           "transform": [{"select": {"start": 0, "count": 3}}]}])
    with pytest.raises(ValueError, match="2"):
        plan.build({"velocity_command": [1.0, 2.0, 3.0]})


def test_mismatch_without_any_transform_is_still_rejected():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "velocity_command", "size": 2}])
    with pytest.raises(ValueError):
        plan.build({"velocity_command": [1.0, 2.0, 3.0]})


def test_a_subset_of_joints_can_be_selected():
    """A policy driving only the knees of a 12-joint robot."""
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "joint_position", "size": 4,
                           "transform": [{"select": [2, 5, 8, 11]}]}])
    values = [float(i) for i in range(12)]
    assert plan.build({"joint_position": values}) == pytest.approx([2.0, 5.0, 8.0, 11.0])


def test_chained_size_changes_are_measured_at_the_end():
    from robustness_lab.observation.compiler import compile_terms
    plan = compile_terms([{"source": "joint_position", "size": 6,
                           "transform": [{"select": {"start": 0, "count": 3}},
                                         {"repeat": 2}]}])
    values = [float(i) for i in range(12)]
    assert plan.build({"joint_position": values}) == pytest.approx(
        [0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
