"""A free worker must be refilled immediately, not at the next batch boundary.

The loop planned a batch, launched it, and then waited for EVERY trial in it:

    records = [future.result() for future in futures]

so a batch ran at the pace of its slowest trial and the pool drained to idle at
every tail. Between batches the evidence rendering ran with nothing in flight.

Measured on a 60-trial sweep at 24 workers, before this change:

    peak concurrency observed        : 24
    mean concurrency                 : 5.4
    time with ZERO trials running    : 65% of the run
    effective core use               : 4.8 of 56

It reached full width briefly and then sat idle for two thirds of the wall
clock. The fix is a rolling queue: dispatch a new trial the moment a worker
frees, and keep a barrier only where the schedule genuinely needs one -- a level
boundary, because promotion and pruning depend on the previous level's verdicts.

The idle-fraction test below is the one that measures what was actually wrong.
"""

import datetime as dt
import json
import pytest
import yaml
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


def _sweep(tmp_path, workers, trials, terrains=None):
    """trials=None runs the examination to completion."""
    from robustness_lab.discrete_examiner import (
        initialize_discrete_examination, run_discrete_examination)
    body = {
        "schema_version": "2.0", "examiner_id": "rolling",
        "scheduler": "discrete_pruned_v2", "backend": "mujoco",
        "robot_profile": str(PROFILE), "robot_asset": str(ASSET),
        "robot_format": "mjcf", "skill": str(SKILL),
        "runtimes": str(ROOT / "configs/runtimes.yaml"),
        "output_dir": str(tmp_path / "run"),
        "terrain_codes": terrains or ["rs1", "rs2", "rs5"], "seed": 1701,
        "discrete": {"protocol_variant_id": "rolling-test",
                     "screening_repetitions": 2, "confirmation_repetitions": 3,
                     "confirmation_successes": 2, "parallel_trials": workers},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "c.yaml"
    config.write_text(yaml.safe_dump(body), encoding="utf-8")
    initialize_discrete_examination(str(config))
    state_path = tmp_path / "run" / "examination_state.json"
    run_discrete_examination(str(state_path), max_trials=trials)
    return json.loads(state_path.read_text())


def _spans(state):
    def stamp(value):
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    return sorted(
        (stamp(t["started_at"]), stamp(t["finished_at"]))
        for t in state["trials"]
        if t.get("started_at") and t.get("finished_at")
    )


def _idle_fraction(spans, samples=1000):
    low = min(a for a, _ in spans)
    high = max(b for _, b in spans)
    if high <= low:
        return 1.0
    step = (high - low) / samples
    idle = sum(
        1 for i in range(samples)
        if not any(a <= low + i * step < b for a, b in spans)
    )
    return idle / samples


# ---------------------------------------------------------------------------
# Planning against work already in flight
# ---------------------------------------------------------------------------

def test_in_flight_attempts_are_not_planned_twice():
    """Without this, a refilled worker re-runs an attempt already running."""
    from robustness_lab.discrete_examiner import plan_ready_trials

    condition = {"id": "L1-a", "status": "pending", "attempts": [],
                 "target_repetitions": 3, "level": "L1", "parents": []}
    state = {"conditions": {"L1-a": condition}, "condition_order": ["L1-a"],
             "settings": {}, "trials": []}
    planned = plan_ready_trials(state, workers=8, in_flight={"L1-a": {0, 1}})
    assert all(item["attempt_index"] not in {0, 1} for item in planned), planned


def test_in_flight_work_counts_towards_the_repetition_target():
    from robustness_lab.discrete_examiner import plan_ready_trials

    condition = {"id": "L1-a", "status": "pending", "attempts": [],
                 "target_repetitions": 2, "level": "L1", "parents": []}
    state = {"conditions": {"L1-a": condition}, "condition_order": ["L1-a"],
             "settings": {}, "trials": []}
    planned = plan_ready_trials(state, workers=8, in_flight={"L1-a": {0}})
    assert len(planned) <= 1, planned


def test_a_condition_still_waits_for_its_parents():
    from robustness_lab.discrete_examiner import plan_ready_trials

    parent = {"id": "L1-a", "status": "pending", "attempts": [],
              "target_repetitions": 2, "level": "L1", "parents": []}
    child = {"id": "L2-ab", "status": "pending", "attempts": [],
             "target_repetitions": 2, "level": "L2", "parents": ["L1-a"]}
    state = {"conditions": {"L1-a": parent, "L2-ab": child},
             "condition_order": ["L1-a", "L2-ab"], "settings": {}, "trials": []}
    planned = plan_ready_trials(state, workers=8, in_flight={})
    assert all(item["condition_id"] != "L2-ab" for item in planned)


# ---------------------------------------------------------------------------
# The behaviour that was wrong
# ---------------------------------------------------------------------------

def test_the_pool_does_not_sit_idle_for_most_of_the_run(tmp_path):
    """Was 65% idle. A rolling queue should keep work in flight throughout."""
    _requirements()
    spans = _spans(_sweep(tmp_path, workers=8, trials=24))
    if len(spans) < 8:
        pytest.skip("not enough completed trials to judge occupancy")
    idle = _idle_fraction(spans)
    assert idle < 0.35, "pool was idle %.0f%% of the run" % (100 * idle)


def test_a_worker_is_refilled_before_the_whole_wave_finishes(tmp_path):
    """The signature of rolling dispatch: a trial starts while others still run
    and at least one has already finished."""
    _requirements()
    spans = _spans(_sweep(tmp_path, workers=8, trials=24))
    if len(spans) < 8:
        pytest.skip("not enough completed trials")
    refilled = any(
        any(b <= start for _, b in spans)          # something already finished
        and any(a < start < b for a, b in spans)   # something still running
        for start, _ in spans
    )
    assert refilled, "every trial started in a lock-step wave"


def test_verdicts_still_match_a_serial_run(tmp_path):
    """Rolling dispatch must not change any result.

    Run to completion: under a trial budget the two runs stop at different
    points in the schedule, which is inherent to parallel dispatch. A finished
    examination must be identical.
    """
    _requirements()
    serial = _sweep(tmp_path / "a", workers=1, trials=None, terrains=["rs1"])
    rolling = _sweep(tmp_path / "b", workers=8, trials=None, terrains=["rs1"])
    assert ({k: v["status"] for k, v in rolling["conditions"].items()}
            == {k: v["status"] for k, v in serial["conditions"].items()})
