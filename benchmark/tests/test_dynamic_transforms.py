"""Researchers must be able to do arbitrary preprocessing on any source.

Two levels:

1. An expanded built-in vocabulary covering the operations real policies use.
   These are verifiable by the platform because their semantics are known.
2. A `custom` escape hatch that runs a researcher-supplied callable, for
   anything the vocabulary cannot express. It is subject to exactly the same
   verification gates, so generality never buys unverified results.
"""

import math

import pytest


# ---------------------------------------------------------------- built-ins

def test_add_offset():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["add"]([1.0, 2.0], [10.0, 20.0]) == pytest.approx([11.0, 22.0])


def test_negate():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["negate"]([1.0, -2.0], None) == pytest.approx([-1.0, 2.0])


def test_abs_values():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["abs"]([-1.5, 2.0], None) == pytest.approx([1.5, 2.0])


def test_power():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["power"]([2.0, 3.0], 2) == pytest.approx([4.0, 9.0])


def test_tanh_squash():
    from robustness_lab.observation.transforms import TRANSFORMS
    out = TRANSFORMS["tanh"]([0.0, 1.0], None)
    assert out == pytest.approx([0.0, math.tanh(1.0)], abs=1e-9)


def test_log1p_for_range_compression():
    from robustness_lab.observation.transforms import TRANSFORMS
    out = TRANSFORMS["log1p"]([0.0, math.e - 1.0], None)
    assert out == pytest.approx([0.0, 1.0], abs=1e-9)


def test_reciprocal_guards_division_by_zero():
    from robustness_lab.observation.transforms import TRANSFORMS
    with pytest.raises(ValueError):
        TRANSFORMS["reciprocal"]([1.0, 0.0], None)


def test_pad_to_width():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["pad"]([1.0, 2.0], {"size": 5, "value": 0.0}) == pytest.approx(
        [1.0, 2.0, 0.0, 0.0, 0.0])


def test_repeat_tiles_values():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["repeat"]([1.0, 2.0], 3) == pytest.approx(
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0])


def test_clamp_norm_limits_vector_magnitude():
    from robustness_lab.observation.transforms import TRANSFORMS
    out = TRANSFORMS["clamp_norm"]([3.0, 4.0], 1.0)          # norm 5 -> 1
    assert math.sqrt(sum(v * v for v in out)) == pytest.approx(1.0, abs=1e-9)
    assert out == pytest.approx([0.6, 0.8], abs=1e-9)


def test_clamp_norm_leaves_short_vectors_untouched():
    from robustness_lab.observation.transforms import TRANSFORMS
    assert TRANSFORMS["clamp_norm"]([0.3, 0.4], 1.0) == pytest.approx([0.3, 0.4])


def test_quat_to_rotmat_of_identity_is_row_major_identity():
    from robustness_lab.observation.transforms import TRANSFORMS
    out = TRANSFORMS["quat_to_rotmat"]([1.0, 0.0, 0.0, 0.0], None)
    assert out == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1], abs=1e-9)


def test_rotate_by_yaw_ninety_degrees():
    """A vector along +x rotated by +90 deg yaw lands on +y."""
    from robustness_lab.observation.transforms import TRANSFORMS
    out = TRANSFORMS["rotate_by_yaw"]([1.0, 0.0, 0.0], math.radians(90.0))
    assert out == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


# ------------------------------------------------------------ escape hatch

def test_custom_transform_runs_a_researcher_supplied_callable():
    from robustness_lab.observation.transforms import apply_transform
    out = apply_transform("custom", [1.0, 2.0, 3.0],
                          {"target": "builtins:reversed"}, allow_custom=True)
    assert list(out) == pytest.approx([3.0, 2.0, 1.0])


def test_custom_transform_is_refused_when_not_explicitly_enabled():
    """Arbitrary code must be opt-in, never silently available."""
    from robustness_lab.observation.transforms import apply_transform
    with pytest.raises(ValueError, match="custom"):
        apply_transform("custom", [1.0], {"target": "builtins:reversed"},
                        allow_custom=False)


def test_custom_transform_rejects_a_target_that_cannot_be_resolved():
    from robustness_lab.observation.transforms import apply_transform
    with pytest.raises(ValueError, match="nope"):
        apply_transform("custom", [1.0], {"target": "nope:missing"},
                        allow_custom=True)


def test_custom_transform_output_must_be_finite():
    from robustness_lab.observation.transforms import apply_transform
    with pytest.raises(ValueError, match="finite"):
        apply_transform("custom", [1.0], {"target": "math:inf"}, allow_custom=True)


def test_unknown_transform_still_lists_the_valid_names():
    from robustness_lab.observation.transforms import apply_transform
    with pytest.raises(ValueError, match="scale"):
        apply_transform("wobble", [1.0], None)
