"""Run aggregation, uncertainty, capability curves, and cross-backend deltas."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .io import dump_json, load_data
from .statistics import bootstrap_interval, capability_curve, kaplan_meier, percentile, wilson_interval


def _distribution_distance(left: List[float], right: List[float]) -> Optional[float]:
    """Approximate one-dimensional Wasserstein distance on a fixed quantile grid."""
    if not left or not right:
        return None
    probabilities = [index / 100.0 for index in range(101)]
    return sum(abs(percentile(left, value) - percentile(right, value)) for value in probabilities) / len(probabilities)


def _event_sequence_divergence(left: List[str], right: List[str]) -> float:
    """Normalized Levenshtein distance for canonical event-type sequences."""
    if not left and not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def _hierarchical_capability_curve(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Treat controller-training seed, rather than episode, as the outer unit."""
    grouped: Dict[float, Dict[Any, List[float]]] = {}
    for record in records:
        grouped.setdefault(float(record["severity"]), {}).setdefault(record["controller_training_seed"], []).append(float(record["metrics"]["success"]))
    curve = []
    for severity in sorted(grouped):
        seed_rates = [sum(values) / len(values) for values in grouped[severity].values()]
        lower, upper = bootstrap_interval(seed_rates, resamples=2000, seed=31)
        curve.append({"severity": severity, "successes": sum(sum(values) for values in grouped[severity].values()), "trials": sum(len(values) for values in grouped[severity].values()), "experimental_units": len(seed_rates), "success_rate": sum(seed_rates) / len(seed_rates), "lower_ci": lower, "upper_ci": upper, "aggregation_unit": "controller_training_seed"})
    auc = 0.0
    if len(curve) >= 2 and curve[-1]["severity"] > curve[0]["severity"]:
        width = curve[-1]["severity"] - curve[0]["severity"]
        auc = sum((right["severity"] - left["severity"]) * (left["success_rate"] + right["success_rate"]) / 2.0 for left, right in zip(curve, curve[1:])) / width
    c50_values = [point["severity"] for point in curve if point["success_rate"] >= 0.5]
    c90_values = [point["severity"] for point in curve if point["lower_ci"] >= 0.9]
    return {"curve": curve, "auc": auc, "c50": max(c50_values) if c50_values else float("nan"), "c90": max(c90_values) if c90_values else float("nan"), "confidence": 0.95, "aggregation_unit": "controller_training_seed"}


def aggregate_runs(run_dirs: List[str], output_path: Optional[str] = None, conformance_path: Optional[str] = None, calibration_path: Optional[str] = None) -> Dict[str, Any]:
    records = []
    for path in run_dirs:
        root = Path(path).resolve()
        manifest = load_data(str(root / "manifest.json"))
        metrics_path = root / "metrics.json"
        metrics = load_data(str(metrics_path)) if metrics_path.exists() else {"metrics": []}
        log_path = root / "episode_log.json"
        log = load_data(str(log_path)) if log_path.exists() else {"events": []}
        episode = manifest["episode_spec"]
        metric_map = {item["name"]: item.get("value") for item in metrics.get("metrics", []) if item.get("valid") and item.get("value") is not None}
        records.append({"run_dir": str(root), "backend": episode["backend"], "seed": episode.get("seed", 0), "controller_training_seed": episode.get("metadata", {}).get("controller_training_seed"), "condition_id": episode.get("metadata", {}).get("condition_id", Path(episode["world"]).stem), "severity": episode.get("metadata", {}).get("severity"), "status": manifest["status"], "terminal_reason": log.get("terminal_reason", "unknown"), "fidelity_tier": log.get("diagnostics", {}).get("terrain_fidelity_tier"), "metrics": metric_map, "event_sequence": [item.get("event_type", "") for item in log.get("events", [])]})
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        backend = record["backend"]
        for name, value in record["metrics"].items():
            grouped.setdefault(backend, {}).setdefault(name, []).append(float(value))
    summaries = {}
    for backend, metrics in grouped.items():
        summaries[backend] = {}
        for name, values in metrics.items():
            matching_records = [record for record in records if record["backend"] == backend and name in record["metrics"]]
            uses_controller_seeds = any(record["controller_training_seed"] is not None for record in matching_records)
            if uses_controller_seeds:
                per_seed: Dict[Any, List[float]] = {}
                for record in matching_records:
                    per_seed.setdefault(record["controller_training_seed"], []).append(float(record["metrics"][name]))
                experimental_values = [sum(seed_values) / len(seed_values) for seed_values in per_seed.values()]
                mean = sum(experimental_values) / len(experimental_values)
                lower, upper = bootstrap_interval(experimental_values, resamples=2000, seed=17)
                count = len(experimental_values)
                aggregation_unit = "controller_training_seed"
            else:
                mean = sum(values) / len(values)
                if name == "success":
                    lower, upper = wilson_interval(round(sum(values)), len(values))
                else:
                    lower, upper = bootstrap_interval(values, resamples=1000, seed=17)
                count = len(values)
                aggregation_unit = "episode"
            summaries[backend][name] = {"mean": mean, "lower_ci": lower, "upper_ci": upper, "count": count, "episode_count": len(values), "aggregation_unit": aggregation_unit}
    curves = {}
    survival_curves = {}
    for backend in sorted(grouped):
        capability_records = [record for record in records if record["backend"] == backend and record["severity"] is not None and "success" in record["metrics"]]
        if capability_records:
            if any(record["controller_training_seed"] is not None for record in capability_records):
                curves[backend] = _hierarchical_capability_curve(capability_records)
            else:
                curves[backend] = capability_curve([(float(record["severity"]), bool(record["metrics"]["success"])) for record in capability_records])
        survival_records = [record for record in records if record["backend"] == backend and "completion_time" in record["metrics"] and record["status"] not in {"skipped", "error"}]
        if survival_records:
            survival_curves[backend] = kaplan_meier(
                [record["metrics"]["completion_time"] for record in survival_records],
                [record["terminal_reason"] in {"fall", "stuck", "route_incomplete", "behavioral_failure"} for record in survival_records],
            )
    comparability = {"status": "blocked", "reasons": []}
    if conformance_path:
        conformance = load_data(conformance_path)
        if not conformance.get("comparable"):
            comparability["reasons"].append("robot conformance failed")
    else:
        comparability["reasons"].append("robot conformance evidence missing")
    if calibration_path:
        calibration = load_data(calibration_path)
        if not calibration.get("complete"):
            comparability["reasons"].append("backend calibration incomplete")
        required_calibration_backends = {record["backend"] for record in records if record["backend"] != "fake"}
        missing_calibration_backends = sorted(required_calibration_backends - set(calibration.get("backends", {})))
        if missing_calibration_backends:
            comparability["reasons"].append("calibration evidence missing for: %s" % ", ".join(missing_calibration_backends))
    else:
        comparability["reasons"].append("backend calibration evidence missing")
    if not comparability["reasons"]:
        comparability["status"] = "comparable"
    status_counts: Dict[str, Dict[str, int]] = {}
    for record in records:
        backend_status = status_counts.setdefault(record["backend"], {})
        backend_status[record["status"]] = backend_status.get(record["status"], 0) + 1
    cross_backend = []
    by_backend = {name: [record for record in records if record["backend"] == name] for name in sorted({record["backend"] for record in records})}
    for left_name, right_name in itertools.combinations(sorted(by_backend), 2):
        left_index = {(item["condition_id"], item["seed"]): item for item in by_backend[left_name]}
        right_index = {(item["condition_id"], item["seed"]): item for item in by_backend[right_name]}
        keys = sorted(set(left_index) & set(right_index))
        disagreements = []
        event_divergences = []
        fidelity_mismatches = []
        metric_deltas: Dict[str, List[float]] = {}
        for key in keys:
            left, right = left_index[key], right_index[key]
            if left.get("fidelity_tier") and right.get("fidelity_tier") and left["fidelity_tier"] != right["fidelity_tier"]:
                fidelity_mismatches.append({"condition_id": key[0], "seed": key[1], "left": left["fidelity_tier"], "right": right["fidelity_tier"]})
            event_divergences.append(_event_sequence_divergence(left["event_sequence"], right["event_sequence"]))
            if "success" in left["metrics"] and "success" in right["metrics"]:
                disagreements.append(left["metrics"]["success"] != right["metrics"]["success"])
            for name in set(left["metrics"]) & set(right["metrics"]):
                metric_deltas.setdefault(name, []).append(left["metrics"][name] - right["metrics"][name])
        common_metrics = set(grouped.get(left_name, {})) & set(grouped.get(right_name, {}))
        distribution_distances = {
            name: _distribution_distance(grouped[left_name][name], grouped[right_name][name])
            for name in sorted(common_metrics)
        }
        delta_intervals = {}
        for name, values in metric_deltas.items():
            lower, upper = bootstrap_interval(values, resamples=1000, seed=23)
            delta_intervals[name] = {"mean": sum(values) / len(values), "lower_ci": lower, "upper_ci": upper, "count": len(values)}
        threshold_delta = {}
        if left_name in curves and right_name in curves:
            for threshold in ("c50", "c90"):
                left_threshold, right_threshold = curves[left_name][threshold], curves[right_name][threshold]
                threshold_delta[threshold] = left_threshold - right_threshold if math.isfinite(left_threshold) and math.isfinite(right_threshold) else None
        cross_backend.append({"left": left_name, "right": right_name, "matched_trials": len(keys), "success_disagreement_rate": sum(disagreements) / len(disagreements) if disagreements else None, "mean_event_sequence_divergence": sum(event_divergences) / len(event_divergences) if event_divergences else None, "metric_delta_ci": delta_intervals, "distribution_distance": distribution_distances, "capability_threshold_delta": threshold_delta, "fidelity_mismatches": fidelity_mismatches, "valid_comparison": comparability["status"] == "comparable" and not fidelity_mismatches})
    result = {"schema_version": "1.0", "run_count": len(records), "status_counts": status_counts, "backend_summaries": summaries, "capability_curves": curves, "survival_curves": survival_curves, "cross_backend": cross_backend, "comparability": comparability, "runs": records}
    if output_path:
        dump_json(output_path, result)
    return result
