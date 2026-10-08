"""A researcher must be able to override ONE source's width without rewriting.

Found by walking four real policies through the benchmark as a submitter.
Three compiled from a plain source list with no widths typed. The fourth takes
47 where the standard list gives 48 -- one value different. The natural fix is
to override that one entry:

    sources:
      - base_linear_velocity_b
      - {source: velocity_command, size: 2, transform: [{select: [0, 2]}]}
      - joint_position
      ...

but that raised "source 3 must not declare size; source-list sizes are
derived", forcing a full rewrite into explicit terms. Derived-by-default with a
per-entry override is what anyone would expect, and without it the easy form
collapses the moment a policy is slightly non-standard.
"""

import pytest

JOINTS12 = ["j%d" % i for i in range(12)]


def test_one_entry_may_override_its_width():
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(
        ["projected_gravity_b",
         {"source": "velocity_command", "size": 2, "transform": [{"select": [0, 2]}]},
         "joint_position"],
        joint_order=JOINTS12)
    assert plan.size == 3 + 2 + 12


def test_overridden_entry_still_applies_its_transform():
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(
        [{"source": "velocity_command", "size": 2, "transform": [{"select": [0, 2]}]}],
        joint_order=JOINTS12)
    assert plan.build({"velocity_command": [1.0, 2.0, 3.0]}) == pytest.approx([1.0, 3.0])


def test_underived_entries_keep_deriving_around_an_override():
    """The point of mixing: everything else still sizes itself."""
    from robustness_lab.observation.compiler import compile_source_list
    joints18 = ["j%d" % i for i in range(18)]
    plan = compile_source_list(
        ["joint_position",
         {"source": "velocity_command", "size": 2, "transform": [{"select": [0, 2]}]},
         "joint_velocity"],
        joint_order=joints18)
    assert plan.size == 18 + 2 + 18


def test_override_that_disagrees_with_its_transform_is_rejected():
    """Mixing must not weaken validation."""
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(
        [{"source": "velocity_command", "size": 2, "transform": [{"select": [0, 1, 2]}]}],
        joint_order=JOINTS12)
    with pytest.raises(ValueError):
        plan.build({"velocity_command": [1.0, 2.0, 3.0]})


def test_override_without_a_transform_must_match_the_source():
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(
        [{"source": "velocity_command", "size": 2}], joint_order=JOINTS12)
    with pytest.raises(ValueError):
        plan.build({"velocity_command": [1.0, 2.0, 3.0]})


def test_the_47_case_from_the_walkthrough():
    """The concrete case that exposed this: 48 standard, artifact wants 47."""
    from robustness_lab.observation.compiler import compile_source_list
    plan = compile_source_list(
        ["base_linear_velocity_b", "base_angular_velocity_b",
         {"source": "projected_gravity_b", "size": 2, "transform": [{"select": [0, 2]}]},
         "velocity_command", "joint_position", "joint_velocity", "previous_action"],
        joint_order=JOINTS12)
    assert plan.size == 47
