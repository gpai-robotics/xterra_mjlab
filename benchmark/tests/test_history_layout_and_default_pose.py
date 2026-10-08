"""Three silent-wrong-number defects found by mapping a real exported bundle.

Source: rma_go2_lab/policies/exported/go2_blind_rough_combined_asymppo_steps_v1
        -- a two-input blind history policy (45 + 4500 -> 12) exported from
        mjlab/Isaac Lab, whose `deploy.yaml` declares:

            history_layout: isaaclab_term_major
            policy_history_length: 100
            policy_history_dim: 4500

Each defect below produces a PLAUSIBLE NUMBER rather than an error, which is
the worst failure mode a benchmark has.

1. TERM-MAJOR HISTORY.  The compiler stacked history frame-major only:
   [frame_t, frame_t-1, ...].  Isaac Lab / mjlab concatenate each term's whole
   history contiguously instead: 100 frames of base_ang_vel, then 100 of
   projected_gravity, and so on.  Both total 4500, so the G1 dimension gate
   PASSES and the run proceeds on a permuted observation.

2. UNDECLARED HISTORY DIRECTION.  `order` silently defaulted to newest_first.
   A bundle that stacks oldest-first is then fed time-reversed, again at the
   correct width.  With frames > 1 the direction must be stated.

3. SILENTLY ZEROED DEFAULT POSE.  `_reference_values` injected
   default_joint_position = [0.0] * action_size whenever the manifest omitted
   it -- so `subtract: default_joint_position` subtracted nothing and
   joint_pos_rel was fed as raw absolute joint angles.  The bundle's stance
   (0.1/-0.1, 0.8/1.0, -1.5) is far from zero, so this is a large, silent,
   constant offset on every joint.
"""

import pytest
import yaml


# --------------------------------------------------------------------------
# 1. Term-major history layout
# --------------------------------------------------------------------------

def test_term_major_history_groups_each_term_across_frames():
    """Frame-major would give [1,2,3,4,5,6]; term-major regroups by term."""
    from robustness_lab.observation.compiler import compile_terms

    plan = compile_terms(
        [{"source": "velocity_command", "size": 2},
         {"source": "previous_action", "size": 1}],
        history={"frames": 2, "order": "oldest_first", "layout": "term_major"},
    )
    plan.build({"velocity_command": [1.0, 2.0], "previous_action": [3.0]})
    out = plan.build({"velocity_command": [4.0, 5.0], "previous_action": [6.0]})

    assert out == pytest.approx([1.0, 2.0, 4.0, 5.0, 3.0, 6.0], abs=1e-9)


def test_term_major_still_honours_newest_first_within_each_term():
    from robustness_lab.observation.compiler import compile_terms

    plan = compile_terms(
        [{"source": "velocity_command", "size": 2},
         {"source": "previous_action", "size": 1}],
        history={"frames": 2, "order": "newest_first", "layout": "term_major"},
    )
    plan.build({"velocity_command": [1.0, 2.0], "previous_action": [3.0]})
    out = plan.build({"velocity_command": [4.0, 5.0], "previous_action": [6.0]})

    assert out == pytest.approx([4.0, 5.0, 1.0, 2.0, 6.0, 3.0], abs=1e-9)


def test_frame_major_remains_the_default_layout():
    from robustness_lab.observation.compiler import compile_terms

    plan = compile_terms(
        [{"source": "velocity_command", "size": 2},
         {"source": "previous_action", "size": 1}],
        history={"frames": 2, "order": "oldest_first"},
    )
    plan.build({"velocity_command": [1.0, 2.0], "previous_action": [3.0]})
    out = plan.build({"velocity_command": [4.0, 5.0], "previous_action": [6.0]})

    assert out == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], abs=1e-9)


def test_unknown_history_layout_is_rejected_and_lists_the_valid_names():
    from robustness_lab.observation.compiler import compile_terms

    with pytest.raises(ValueError, match="term_major"):
        compile_terms([{"source": "velocity_command", "size": 3}],
                      history={"frames": 2, "order": "newest_first",
                               "layout": "diagonal"})


def test_go2_bundle_history_is_4500_and_grouped_by_term():
    """The exported bundle's policy_history input, reproduced exactly.

    6 terms of 3/3/3/12/12/12 over 100 frames.  Term-major means the first 300
    values are base_ang_vel's whole history and nothing else.  The first build
    after a reset fills all 100 frames with the live one (Isaac Lab's
    CircularBuffer); after that oldest_first puts each term's newest values at
    the END of that term's block.
    """
    from robustness_lab.observation.compiler import compile_terms

    plan = compile_terms(
        [{"source": "base_angular_velocity_b", "size": 3},
         {"source": "projected_gravity_b", "size": 3},
         {"source": "velocity_command", "size": 3},
         {"source": "joint_position", "size": 12},
         {"source": "joint_velocity", "size": 12},
         {"source": "previous_action", "size": 12}],
        history={"frames": 100, "order": "oldest_first", "layout": "term_major"},
    )
    assert plan.size == 4500

    out = plan.build({
        "base_angular_velocity_b": [1.0, 2.0, 3.0],
        "projected_gravity_b": [4.0, 5.0, 6.0],
        "velocity_command": [7.0, 8.0, 9.0],
        "joint_position": [10.0] * 12,
        "joint_velocity": [11.0] * 12,
        "previous_action": [12.0] * 12,
    })

    assert len(out) == 4500
    # The first frame is backfilled into every slot, never zero-padded.
    assert out[:300] == pytest.approx([1.0, 2.0, 3.0] * 100, abs=1e-9)

    out = plan.build({
        "base_angular_velocity_b": [-1.0, -2.0, -3.0],
        "projected_gravity_b": [4.0, 5.0, 6.0],
        "velocity_command": [7.0, 8.0, 9.0],
        "joint_position": [10.0] * 12,
        "joint_velocity": [11.0] * 12,
        "previous_action": [12.0] * 12,
    })
    # base_ang_vel occupies [0, 300); its newest frame is the last one in there.
    assert out[297:300] == pytest.approx([-1.0, -2.0, -3.0], abs=1e-9)
    assert out[:297] == pytest.approx([1.0, 2.0, 3.0] * 99, abs=1e-9)
    # projected_gravity occupies [300, 600).
    assert out[597:600] == pytest.approx([4.0, 5.0, 6.0], abs=1e-9)
    # velocity_command [600, 900), then joint_position [900, 2100).
    assert out[897:900] == pytest.approx([7.0, 8.0, 9.0], abs=1e-9)
    assert out[2088:2100] == pytest.approx([10.0] * 12, abs=1e-9)
    assert out[4488:4500] == pytest.approx([12.0] * 12, abs=1e-9)


# --------------------------------------------------------------------------
# 2. History direction must be declared
# --------------------------------------------------------------------------

def test_stacking_frames_without_declaring_an_order_is_refused():
    from robustness_lab.observation.compiler import compile_terms

    with pytest.raises(ValueError, match="order"):
        compile_terms([{"source": "velocity_command", "size": 3}],
                      history={"frames": 100})


def test_a_single_frame_needs_no_order():
    from robustness_lab.observation.compiler import compile_terms

    plan = compile_terms([{"source": "velocity_command", "size": 3}],
                         history={"frames": 1})
    assert plan.size == 3


# --------------------------------------------------------------------------
# 3. Default pose must come from somewhere real
# --------------------------------------------------------------------------

JOINTS = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
          "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
          "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"]
STANCE = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]


def _manifest(tmp_path, *, references=None, robot_profile=None):
    body = {
        "schema_version": "2.0", "policy_id": "pose",
        "runtime": {"format": "torchscript", "path": "p.pt",
                    "control_frequency_hz": 50},
        "canonical_joint_order": JOINTS, "joint_order_policy": JOINTS,
        "observation": {
            "terms": [{"source": "joint_position", "size": 12,
                       "transform": [{"subtract": "default_joint_position"}]}],
            "history": {"frames": 1},
            "expected_size": 12,
        },
        "action": {"space": "joint_position_offset", "scale": 0.25,
                   "offset": "default_joint_position", "size": 12},
    }
    if references is not None:
        body["references"] = references
    if robot_profile is not None:
        body["robot_profile"] = robot_profile
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


def _profile(stance):
    return {"name": "go2", "source_path": "go2.urdf", "source_format": "urdf",
            "source_sha256": "0" * 64,
            "controller_joint_order": list(JOINTS),
            "default_joint_positions": dict(zip(JOINTS, stance))}


def test_named_default_pose_with_no_source_is_refused_not_silently_zero(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest

    with pytest.raises(ValueError, match="default_joint_position"):
        load_policy_manifest(_manifest(tmp_path), check_files=False)


def test_default_pose_is_taken_from_the_bound_robot_profile(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest

    path = _manifest(tmp_path, robot_profile=_profile(STANCE))
    manifest = load_policy_manifest(path, check_files=False)

    measured = [value + 0.05 for value in STANCE]
    built = manifest.observation_plan.build({"joint_position": measured})
    assert built == pytest.approx([0.05] * 12, abs=1e-9)


def test_a_permuted_stance_is_refused_as_a_joint_order_error(tmp_path):
    """Same angles, different order, is never a design choice.

    A reference pose holding exactly the robot's angles in a different order
    means the manifest was written against a different joint ordering, which
    scrambles every leg while staying the right width.
    """
    from robustness_lab.policy_manifest import load_policy_manifest

    permuted = list(reversed(STANCE))
    assert sorted(permuted) == sorted(STANCE) and permuted != STANCE
    path = _manifest(tmp_path,
                     references={"default_joint_position": permuted},
                     robot_profile=_profile(STANCE))

    with pytest.raises(ValueError, match="different joint ordering"):
        load_policy_manifest(path, check_files=False)


def test_a_genuinely_different_reference_pose_is_recorded_not_refused(tmp_path):
    """The two shipped go2 profiles stand differently: 0.1/0.8/-1.5 vs 0.0/0.9/-1.8.

    The pose a policy was trained to offset from belongs to the POLICY, not to
    the robot, so the two legitimately differ. The manifest value wins and the
    difference is reported rather than blocked -- but a permutation, which
    cannot be a legitimate difference, is still refused above.
    """
    from robustness_lab.policy_manifest import load_policy_manifest

    mjcf_stance = [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, -1.8, -1.8, -1.8, -1.8]
    assert sorted(mjcf_stance) != sorted(STANCE)
    path = _manifest(tmp_path,
                     references={"default_joint_position": STANCE},
                     robot_profile=_profile(mjcf_stance))

    manifest = load_policy_manifest(path, check_files=False)
    assert any("differs from the robot profile stance" in note
               for note in manifest.warnings)
    # The manifest's own pose is what is actually applied.
    built = manifest.observation_plan.build({"joint_position": list(STANCE)})
    assert built == pytest.approx([0.0] * 12, abs=1e-9)


def test_an_explicit_stance_matching_the_profile_is_accepted(tmp_path):
    from robustness_lab.policy_manifest import load_policy_manifest

    path = _manifest(tmp_path,
                     references={"default_joint_position": STANCE},
                     robot_profile=_profile(STANCE))
    manifest = load_policy_manifest(path, check_files=False)
    built = manifest.observation_plan.build({"joint_position": list(STANCE)})
    assert built == pytest.approx([0.0] * 12, abs=1e-9)


def test_a_policy_that_never_names_the_default_pose_still_loads(tmp_path):
    """Only manifests that actually USE the reference require it."""
    from robustness_lab.policy_manifest import load_policy_manifest

    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "2.0", "policy_id": "no-pose",
        "runtime": {"format": "torchscript", "path": "p.pt",
                    "control_frequency_hz": 50},
        "canonical_joint_order": JOINTS, "joint_order_policy": JOINTS,
        "observation": {"terms": [{"source": "joint_position", "size": 12}],
                        "expected_size": 12},
        "action": {"space": "joint_position_offset", "scale": 0.25,
                   "clip": [-100.0, 100.0], "size": 12},
    }), encoding="utf-8")
    manifest = load_policy_manifest(str(path), check_files=False)
    assert manifest.observation_plan.size == 12


# --------------------------------------------------------------------------
# 4. The named default pose must survive a joint permutation
# --------------------------------------------------------------------------
#
# `default_joint_position` is resolved once but consumed in two different joint
# orders:
#
#   observation -- transforms run BEFORE the permutation, so `subtract` sees
#                  values in CANONICAL order
#   action      -- offset runs BEFORE the inverse permutation, so it is applied
#                  in POLICY order
#
# With a hip-major policy on a per-leg robot the two disagree, and a zero action
# commanded hips to 0.8 and -1.5 rad. Gains were already reordered here
# (_resolve_gain takes the permutation); the named offset was not.

CANON_ORDER = ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
               "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
               "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
               "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
CANON_STANCE = [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5]
# The same stance as the exported bundle spells it, in ITS hip-major order.
BUNDLE_STANCE = [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]


def _decoder(offset):
    from robustness_lab.observation.action import compile_action
    return compile_action(
        {"space": "joint_position_offset", "scale": 0.25, "offset": offset,
         "clip": [-100.0, 100.0], "size": 12, "joint_order": JOINTS},
        references={"default_joint_position": CANON_STANCE},
        canonical_joint_order=CANON_ORDER,
    )


def test_zero_action_commands_the_default_stance_under_a_joint_permutation():
    """The whole point of an offset action space: no action means stand still."""
    decoded = _decoder("default_joint_position").decode([0.0] * 12)
    assert decoded == pytest.approx(CANON_STANCE, abs=1e-9)


def test_an_explicit_offset_list_is_still_read_in_policy_order():
    """Backwards compatible, and the two spellings must agree.

    An explicit list follows the same convention as `stiffness`/`damping`: it is
    written in the policy's own joint order.
    """
    decoded = _decoder(BUNDLE_STANCE).decode([0.0] * 12)
    assert decoded == pytest.approx(CANON_STANCE, abs=1e-9)
