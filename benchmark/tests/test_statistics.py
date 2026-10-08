import math

from robustness_lab.aggregate import _hierarchical_capability_curve
from robustness_lab.statistics import bootstrap_interval, capability_curve, kaplan_meier, wilson_interval


def test_wilson_and_bootstrap_intervals():
    low, high = wilson_interval(9, 10)
    assert 0 < low < 0.9 < high <= 1
    low, high = bootstrap_interval([1, 2, 3, 4, 5], resamples=500, seed=3)
    assert low < 3 < high


def test_capability_curve_uses_lower_bound_for_c90():
    points = [(0.1, True)] * 30 + [(0.5, True)] * 29 + [(0.5, False)] + [(0.9, False)] * 30
    result = capability_curve(points)
    assert 0 <= result["auc"] <= 1
    assert result["c50"] == 0.5
    assert math.isnan(result["c90"]) or result["c90"] <= 0.5


def test_kaplan_meier_handles_censoring():
    curve = kaplan_meier([1.0, 2.0, 2.0], [True, True, False])
    assert curve[0]["survival"] == 1.0
    assert curve[-1]["survival"] < 1.0


def test_capability_curve_keeps_controller_seed_as_outer_unit():
    records = []
    for controller_seed, outcomes in ((10, [1.0, 1.0, 0.0]), (20, [0.0, 0.0, 0.0])):
        for episode_seed, outcome in enumerate(outcomes):
            records.append({"severity": 0.5, "controller_training_seed": controller_seed, "metrics": {"success": outcome}, "seed": episode_seed})
    curve = _hierarchical_capability_curve(records)
    assert curve["curve"][0]["experimental_units"] == 2
    assert curve["curve"][0]["trials"] == 6
    assert curve["aggregation_unit"] == "controller_training_seed"
