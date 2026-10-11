"""Independent trials must be able to run concurrently.

`run_discrete_examination` is a single cursor: `_next_condition` picks one
condition, `_execute_condition` runs one trial, the state file is rewritten, and
the loop repeats. One episode at a time, on a machine with 56 cores.

The trials are independent -- each has its own seed, world file and output
directory, and they do not communicate. The 100 L1 single-terrain conditions in
particular have no interdependencies at all.

What is NOT independent, and must survive:

  * L0 qualification gates everything after it;
  * a condition is promoted from 3 screening trials to 10 confirmation trials
    based on the screening result, so dispatching 10 at once would run trials
    the serial scheduler would never have run;
  * L2 pairs are pruned by their L1 parents' verdicts, so no L2 trial may start
    before those parents are decided.

The criterion that matters most is determinism: a parallel run must reach the
same verdicts as a serial one. Trial ORDER may differ; outcomes may not. Every
trial's seed is derived from its condition index and attempt index, so this is
achievable -- but only if attempt indices are assigned when the batch is
planned, not derived from len(attempts) at execution time, which would collide
between concurrent trials of the same condition.
"""

import json
import time
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/cache/mujoco_menagerie/unitree_go2/go2.xml"
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"
SKILL = ROOT / "submissions/go2_blind/skill.yaml"


def _requirements():
    pytest.importorskip("mujoco")
    pytest.importorskip("onnxruntime")
    for path in (ASSET, PROFILE, SKILL):
        if not path.is_file():
            pytest.skip("missing %s" % path.name)


def _config(tmp_path, name, parallel):
    """A deliberately small examination: one terrain, few repetitions."""
    import yaml
    body = {
        "schema_version": "2.0",
        "examiner_id": "parallel-%s" % name,
        "scheduler": "discrete_pruned_v2",
        "backend": "mujoco",
        "robot_profile": str(PROFILE),
        "robot_asset": str(ASSET),
        "robot_format": "mjcf",
        "skill": str(SKILL),
        "runtimes": str(ROOT / "configs/runtimes.yaml"),
        "output_dir": str(tmp_path / name),
        "terrain_codes": ["rs1"],
        "seed": 1701,
        "discrete": {
            "protocol_variant_id": "parallel-test",
            "screening_repetitions": 2,
            "confirmation_repetitions": 3,
            "confirmation_successes": 2,
            "parallel_trials": parallel,
        },
    }
    path = tmp_path / ("%s.yaml" % name)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return str(path)


def _run(tmp_path, name, parallel, max_trials=8):
    """max_trials=None runs the examination to completion."""
    from robustness_lab.discrete_examiner import (
        initialize_discrete_examination, run_discrete_examination)
    initialize_discrete_examination(_config(tmp_path, name, parallel))
    state_path = tmp_path / name / "examination_state.json"
    started = time.perf_counter()
    run_discrete_examination(str(state_path), max_trials=max_trials)
    elapsed = time.perf_counter() - started
    return json.loads(state_path.read_text()), elapsed


def _verdicts(state):
    return {name: (item["status"], item.get("evidence_grade"))
            for name, item in state["conditions"].items()}


def _trial_outcomes(state):
    return sorted(
        (t.get("condition_id"), t.get("attempt_index"), t.get("success"),
         t.get("terminal_reason"))
        for t in state["trials"])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_parallelism_is_opt_in():
    """An existing config with no parallel_trials key keeps running serially."""
    from robustness_lab.discrete_examiner import DEFAULT_PARALLEL_TRIALS
    assert DEFAULT_PARALLEL_TRIALS == 1


def test_the_worker_count_is_bounded():
    from robustness_lab.discrete_examiner import resolve_worker_count
    assert resolve_worker_count(1) == 1
    assert resolve_worker_count(8) <= 8
    assert resolve_worker_count(10_000) < 10_000, "must be capped by the machine"
    for bad in (0, -3):
        with pytest.raises(ValueError):
            resolve_worker_count(bad)


# ---------------------------------------------------------------------------
# What may run at once
# ---------------------------------------------------------------------------

def test_a_batch_never_overshoots_a_conditions_target_repetitions():
    """Dispatching 10 at once would run trials the scheduler never asked for."""
    from robustness_lab.discrete_examiner import plan_ready_trials

    condition = {"id": "L1-a", "status": "pending", "attempts": [],
                 "target_repetitions": 2, "level": "L1", "parents": []}
    state = {"conditions": {"L1-a": condition}, "condition_order": ["L1-a"],
             "settings": {}, "trials": []}
    planned = plan_ready_trials(state, workers=8)
    assert len(planned) <= 2, "planned %d trials for a 2-repetition target" % len(planned)


def test_planned_attempt_indices_are_unique():
    """Concurrent trials of one condition must not derive the same seed."""
    from robustness_lab.discrete_examiner import plan_ready_trials

    condition = {"id": "L1-a", "status": "pending", "attempts": [],
                 "target_repetitions": 3, "level": "L1", "parents": []}
    state = {"conditions": {"L1-a": condition}, "condition_order": ["L1-a"],
             "settings": {}, "trials": []}
    planned = plan_ready_trials(state, workers=8)
    indices = [item["attempt_index"] for item in planned]
    assert len(indices) == len(set(indices)), indices


def test_a_condition_with_undecided_parents_is_never_planned():
    """L2 pairs are pruned by their L1 parents, so they must wait."""
    from robustness_lab.discrete_examiner import plan_ready_trials

    parent = {"id": "L1-a", "status": "pending", "attempts": [],
              "target_repetitions": 2, "level": "L1", "parents": []}
    child = {"id": "L2-ab", "status": "pending", "attempts": [],
             "target_repetitions": 2, "level": "L2", "parents": ["L1-a"]}
    state = {"conditions": {"L1-a": parent, "L2-ab": child},
             "condition_order": ["L1-a", "L2-ab"], "settings": {}, "trials": []}
    planned = plan_ready_trials(state, workers=8)
    assert all(item["condition_id"] != "L2-ab" for item in planned), planned


# ---------------------------------------------------------------------------
# The determinism criterion
# ---------------------------------------------------------------------------

def test_parallel_and_serial_reach_identical_verdicts(tmp_path):
    """Run to COMPLETION, not to a trial budget.

    Under `max_trials` the two runs are truncated at the same trial COUNT but
    reach different points in the schedule, because which trials finish first
    depends on timing. That is inherent to any parallel scheduler and is
    recovered by resuming. The real invariant -- and the stronger claim -- is
    that a completed examination is identical either way.
    """
    _requirements()
    serial, _ = _run(tmp_path, "serial", parallel=1, max_trials=None)
    parallel, _ = _run(tmp_path, "parallel", parallel=4, max_trials=None)
    assert _verdicts(parallel) == _verdicts(serial)


def test_parallel_and_serial_produce_identical_trial_outcomes(tmp_path):
    """Order may differ; the set of (condition, attempt, outcome) may not."""
    _requirements()
    serial, _ = _run(tmp_path, "serial2", parallel=1, max_trials=None)
    parallel, _ = _run(tmp_path, "parallel2", parallel=4, max_trials=None)
    assert _trial_outcomes(parallel) == _trial_outcomes(serial)


# ---------------------------------------------------------------------------
# It has to actually be concurrent, and isolated
# ---------------------------------------------------------------------------

def test_trials_actually_overlap_in_time(tmp_path):
    """Evidence of concurrency: some trial starts before another finishes."""
    _requirements()
    state, _ = _run(tmp_path, "overlap", parallel=4, max_trials=6)
    spans = [(t["started_at"], t["finished_at"]) for t in state["trials"]
             if t.get("started_at") and t.get("finished_at")]
    if len(spans) < 2:
        pytest.skip("not enough completed trials to observe overlap")
    spans.sort()
    overlapped = any(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))
    assert overlapped, "no two trials overlapped; execution was still serial"


def test_every_trial_has_its_own_run_directory(tmp_path):
    _requirements()
    state, _ = _run(tmp_path, "dirs", parallel=4, max_trials=6)
    ids = [t["trial_id"] for t in state["trials"]]
    assert len(ids) == len(set(ids)), "duplicate trial ids: %r" % ids


def test_a_parallel_run_resumes_cleanly(tmp_path):
    """Killing and restarting must not lose or duplicate trials."""
    _requirements()
    from robustness_lab.discrete_examiner import (
        initialize_discrete_examination, run_discrete_examination)
    initialize_discrete_examination(_config(tmp_path, "resume", 4))
    state_path = tmp_path / "resume" / "examination_state.json"
    run_discrete_examination(str(state_path), max_trials=3)
    first = json.loads(state_path.read_text())
    run_discrete_examination(str(state_path), max_trials=3)
    second = json.loads(state_path.read_text())

    assert len(second["trials"]) >= len(first["trials"])
    ids = [t["trial_id"] for t in second["trials"]]
    assert len(ids) == len(set(ids)), "resume duplicated trials"
