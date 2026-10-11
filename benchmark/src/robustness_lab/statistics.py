"""Dependency-light uncertainty and capability-curve utilities."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    # Fixed common confidence levels keep this function scipy-independent.
    z_values = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_values.get(round(confidence, 2), 1.959963984540054)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * trials)) / trials) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def bootstrap_interval(values: Sequence[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0, statistic: Callable[[Sequence[float]], float] = None) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    statistic = statistic or (lambda sample: sum(sample) / len(sample))
    rng = random.Random(seed)
    samples = []
    source = list(values)
    for _ in range(resamples):
        draw = [source[rng.randrange(len(source))] for _ in source]
        samples.append(float(statistic(draw)))
    alpha = (1.0 - confidence) / 2.0
    return percentile(samples, alpha), percentile(samples, 1.0 - alpha)


def paired_bootstrap_interval(left: Sequence[float], right: Sequence[float], **kwargs: Any) -> Tuple[float, float]:
    if len(left) != len(right):
        raise ValueError("paired samples must have equal length")
    return bootstrap_interval([a - b for a, b in zip(left, right)], **kwargs)


def capability_curve(points: Sequence[Tuple[float, bool]], confidence: float = 0.95) -> Dict[str, Any]:
    grouped: Dict[float, List[bool]] = {}
    for severity, success in points:
        grouped.setdefault(float(severity), []).append(bool(success))
    curve = []
    for severity in sorted(grouped):
        outcomes = grouped[severity]
        count = sum(outcomes)
        lower, upper = wilson_interval(count, len(outcomes), confidence)
        curve.append({"severity": severity, "successes": count, "trials": len(outcomes), "success_rate": count / len(outcomes), "lower_ci": lower, "upper_ci": upper})
    auc = 0.0
    if len(curve) >= 2:
        width = curve[-1]["severity"] - curve[0]["severity"]
        if width > 0:
            auc = sum((right["severity"] - left["severity"]) * (left["success_rate"] + right["success_rate"]) / 2.0 for left, right in zip(curve, curve[1:])) / width

    def threshold(probability: float, conservative: bool) -> float:
        eligible = [point["severity"] for point in curve if (point["lower_ci"] if conservative else point["success_rate"]) >= probability]
        return max(eligible) if eligible else float("nan")

    return {"curve": curve, "auc": auc, "c50": threshold(0.5, False), "c90": threshold(0.9, True), "confidence": confidence}


def kaplan_meier(durations: Sequence[float], failed: Sequence[bool]) -> List[Dict[str, float]]:
    if len(durations) != len(failed):
        raise ValueError("durations and failed flags must align")
    records = sorted(zip(durations, failed))
    at_risk = len(records)
    survival = 1.0
    curve = [{"time_s": 0.0, "survival": 1.0, "at_risk": float(at_risk)}]
    for time_s in sorted(set(duration for duration, _ in records)):
        events = sum(1 for duration, event in records if duration == time_s and event)
        censored = sum(1 for duration, event in records if duration == time_s and not event)
        if at_risk and events:
            survival *= 1.0 - events / at_risk
            curve.append({"time_s": float(time_s), "survival": survival, "at_risk": float(at_risk)})
        at_risk -= events + censored
    return curve

