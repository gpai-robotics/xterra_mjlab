"""Later levels must be PRUNED down to what passed, not cancelled wholesale.

A real 757-trial sweep finished with L4-L7 never created at all. The dashboard
explained why: "not created because L3 passed 53.8%, below the 70% gate".

`_next_condition` gated each progression level on the PREVIOUS level's aggregate
pass rate:

    if previous != "L2" and state["level_results"].get(previous) != "passed":
        state["materialized"][level] = True
        state["level_results"][level] = "pruned"
        continue

where `_stage_outcome` returns "passed" only at >= 70%. So one level scoring 54%
deleted four whole levels of the benchmark, including the terrain combinations
the policy had already proved it could handle.

That is a different mechanism from the pruning used at L2 and L3, which is
element-wise: `_passed_pair_library` builds each level only out of pairs that
PASSED, and drops any pair whose counterfactual later failed. That mechanism is
the right one and it already carries failures forward. The aggregate gate on top
of it is what removes work that would have succeeded.

So: keep the element-wise pruning, drop the stage gate. If nothing passed, the
library is empty and the level comes out empty on its own -- which is the honest
"nothing left to test", reached by evidence rather than by a threshold.
"""

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "assets/cache/go2_mjcf_profile.json"


def _robot():
    if not PROFILE.is_file():
        pytest.skip("go2 profile not cached")
    from robustness_lab.io import load_data
    from robustness_lab.models import RobotProfile
    return RobotProfile.from_dict(load_data(str(PROFILE)))


def _state(l3_pass_rate):
    """A state with L0-L2 done, and L3 finished at the given pass rate."""
    from robustness_lab.discrete_examiner import _condition

    conditions, order = {}, []

    def add(item):
        conditions[item["id"]] = item
        order.append(item["id"])

    qualification = _condition("L0-flat", "L0", "qualification", 1)
    qualification["status"] = "passed"
    add(qualification)

    # Four L2 pairs at one geometry level: three passed, one failed.
    pairs = [["rs1", "rs2"], ["rs2", "rs5"], ["rs5", "rs6"], ["rs6", "rs7"]]
    for index, pair in enumerate(pairs):
        item = _condition("L2-%d" % index, "L2", "pair", 10 + index,
                          pair=pair, geometry_level=0.5, geometry_label="medium")
        item["status"] = "passed" if index < 3 else "failed"
        add(item)

    # L3 finished, at the requested pass rate.
    total = 13
    passes = round(l3_pass_rate * total)
    for index in range(total):
        item = _condition("L3-%d" % index, "L3", "progression", 100 + index,
                          topology="chain", horizon="M", geometry_level=0.5,
                          geometry_label="medium", allowed_pairs=[pairs[0]],
                          covered_pairs=[pairs[0]], stressors=[], critical=True)
        item["status"] = "passed" if index < passes else "failed"
        item["counterfactual_created"] = True
        add(item)

    return {
        "conditions": conditions, "condition_order": order,
        "materialized": {"L2": True, "L3": True, "L4": False, "L5": False,
                         "L6": False, "L7": False},
        "level_results": {}, "status": "running", "stage": "L3-composed",
        "settings": {"seed": 7, "terrain_codes": ["rs1", "rs2", "rs5", "rs6", "rs7"],
                     "screening_repetitions": 3, "confirmation_repetitions": 10,
                     "confirmation_successes": 7,
                     "geometry_levels": [{"label": "medium", "value": 0.5}],
                     "friction_levels": [{"label": "moderate", "static": 0.5, "dynamic": 0.4}]},
        "trials": [],
    }


def _levels_created(state):
    return {item["level"] for item in state["conditions"].values()}


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------

def test_l4_is_created_even_when_l3_falls_below_the_old_gate():
    """53.8% deleted four levels. It must now only prune within them."""
    from robustness_lab.discrete_examiner import _next_condition

    state = _state(l3_pass_rate=0.538)
    _next_condition(state, _robot())
    assert "L4" in _levels_created(state), (
        "L4 was still cancelled wholesale: level_results=%r"
        % state.get("level_results"))


def test_a_single_weak_level_does_not_cancel_every_later_level():
    from robustness_lab.discrete_examiner import _next_condition

    state = _state(l3_pass_rate=0.30)
    for _ in range(8):
        if _next_condition(state, _robot()) is None:
            break
    created = _levels_created(state)
    assert {"L4", "L5", "L6"} & created, (
        "no later level was created at all: %r" % sorted(created))


# ---------------------------------------------------------------------------
# The pruning that must survive
# ---------------------------------------------------------------------------

def test_later_levels_use_only_pairs_that_passed():
    from robustness_lab.discrete_examiner import _next_condition

    state = _state(l3_pass_rate=0.538)
    _next_condition(state, _robot())
    failed_pair = ["rs6", "rs7"]
    for item in state["conditions"].values():
        if item["level"] in {"L4", "L5", "L6", "L7"}:
            for pair in item.get("covered_pairs", []):
                assert list(pair) != failed_pair, (
                    "%s reuses the pair that failed at L2" % item["id"])


def test_a_pair_whose_counterfactual_failed_is_dropped():
    """Failures in composed worlds still remove that pair from later levels."""
    from robustness_lab.discrete_examiner import _condition, _next_condition

    state = _state(l3_pass_rate=0.538)
    counterfactual = _condition("CF-0", "L3", "counterfactual", 500,
                                pair=["rs1", "rs2"], geometry_level=0.5,
                                geometry_label="medium")
    counterfactual["status"] = "failed"
    state["conditions"]["CF-0"] = counterfactual
    state["condition_order"].append("CF-0")

    _next_condition(state, _robot())
    for item in state["conditions"].values():
        if item["level"] in {"L4", "L5", "L6", "L7"}:
            for pair in item.get("covered_pairs", []):
                assert list(pair) != ["rs1", "rs2"], (
                    "%s reuses a pair whose counterfactual failed" % item["id"])


def test_nothing_is_created_when_no_pair_passed():
    """Empty by evidence, not by threshold."""
    from robustness_lab.discrete_examiner import _next_condition

    state = _state(l3_pass_rate=0.538)
    for item in state["conditions"].values():
        if item["level"] == "L2":
            item["status"] = "failed"
    _next_condition(state, _robot())
    assert not ({"L4", "L5", "L6", "L7"} & _levels_created(state))


def test_the_stage_outcome_is_still_reported():
    """Still measured and shown -- it just no longer cancels the next level."""
    from robustness_lab.discrete_examiner import _stage_outcome

    state = _state(l3_pass_rate=0.538)
    assert _stage_outcome(state, "L3") in {"passed", "failed", "pruned", "running"}


def test_the_dashboard_no_longer_blames_a_gate():
    source = (ROOT / "src/robustness_lab/evidence.py").read_text()
    assert "below the ${gate}% gate" not in source, (
        "the dashboard still explains missing levels with a gate that is gone")
