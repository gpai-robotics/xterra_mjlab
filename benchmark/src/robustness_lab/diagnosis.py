"""Auditable rule-based failure chains and matched evidence summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import load_data


def _metric_map(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return {
        item["name"]: float(item["value"])
        for item in load_data(str(path)).get("metrics", [])
        if item.get("valid") and isinstance(item.get("value"), (int, float))
    }


def diagnose_run(run_dir: str) -> Dict[str, Any]:
    root = Path(run_dir).resolve()
    log_path = root / "episode_log.json"
    if not log_path.exists():
        return {"status": "unavailable", "run_dir": str(root), "reason": "episode_log.json missing"}
    log = load_data(str(log_path))
    metrics = _metric_map(root)
    events = log.get("events", [])
    types = [item.get("event_type", "") for item in events]
    terminal = log.get("terminal_reason", "unknown")
    chain: List[str] = []
    factors: List[Dict[str, Any]] = []

    first_slip = next((index for index, value in enumerate(types) if value == "slip_start"), None)
    first_fall = next((index for index, value in enumerate(types) if value == "fall"), None)
    if first_slip is not None:
        chain.append("stance-foot slip")
        factors.append({"factor": "surface traction or foothold stability", "evidence": "slip event preceded termination", "strength": "strong" if first_fall is not None and first_slip < first_fall else "moderate"})
    if "body_contact" in types:
        chain.append("forbidden body contact")
        factors.append({"factor": "insufficient body clearance or obstacle collision", "evidence": "non-foot contact detected", "strength": "strong"})
    if first_fall is not None or terminal == "fall":
        chain.append("base instability / fall")
    if "stuck" in types or terminal in {"stuck", "timeout"}:
        chain.append("insufficient forward progress")
        if metrics.get("torque_saturation_fraction", 0.0) > 0.1:
            factors.append({"factor": "actuator saturation", "evidence": "torque saturation exceeded 10% of samples", "strength": "strong"})
        elif metrics.get("velocity_tracking_rmse", 0.0) > 0.2:
            factors.append({"factor": "tracking/control breakdown", "evidence": "large velocity tracking error", "strength": "moderate"})
    if "disturbance_start" in types and "recovered" not in types:
        chain.insert(0, "external disturbance")
        factors.append({"factor": "insufficient disturbance recovery", "evidence": "disturbance occurred without a recovery event", "strength": "strong"})
    if metrics.get("observation_age_mean", 0.0) > 0.04:
        factors.append({"factor": "stale perception", "evidence": "mean observation age exceeded 40 ms", "strength": "moderate"})
    if metrics.get("inference_latency_p95", 0.0) > 0.02:
        factors.append({"factor": "inference latency", "evidence": "p95 inference latency exceeded 20 ms", "strength": "moderate"})
    if terminal == "route_incomplete":
        chain.append("mandatory region or route missed")
        factors.append({"factor": "route selection or waypoint tracking", "evidence": "goal reached without mandatory route completion", "strength": "strong"})
    if terminal in {"simulator_error", "backend_unavailable"} or log.get("status") in {"error", "skipped"}:
        classification = "infrastructure"
        chain = [terminal]
    elif terminal == "goal" and log.get("status") == "completed":
        classification = "success"
        chain = ["mandatory regions traversed", "goal reached"]
        # Slip or contact observations may still be useful telemetry in a
        # successful episode, but they are not failure-cause evidence.
        factors = []
    elif first_fall is not None or terminal == "fall":
        classification = "behavioral/fall"
    elif "stuck" in types or terminal in {"stuck", "timeout"}:
        classification = "behavioral/progress"
    elif terminal == "route_incomplete":
        classification = "behavioral/route"
    else:
        classification = "behavioral/other"
    if classification == "infrastructure":
        failure_category = "infrastructure"
    elif classification == "success":
        failure_category = "success"
    elif first_slip is not None and (first_fall is None or first_slip < first_fall):
        failure_category = "traction/slip"
    elif "body_contact" in types:
        failure_category = "collision/clearance"
    elif metrics.get("torque_saturation_fraction", 0.0) > 0.1:
        failure_category = "actuation saturation"
    elif terminal == "route_incomplete":
        failure_category = "route error"
    elif metrics.get("observation_age_mean", 0.0) > 0.04 or metrics.get("inference_latency_p95", 0.0) > 0.02:
        failure_category = "perception/latency"
    elif first_fall is not None or terminal == "fall":
        failure_category = "fall"
    elif "stuck" in types or terminal in {"stuck", "timeout"}:
        failure_category = "stuck/progress"
    elif metrics.get("velocity_tracking_rmse", 0.0) > 0.2:
        failure_category = "control tracking"
    else:
        failure_category = "unknown"
    return {
        "status": "ok",
        "run_dir": str(root),
        "episode_status": log.get("status"),
        "terminal_reason": terminal,
        "classification": classification,
        "failure_category": failure_category,
        "event_chain": chain,
        "likely_factors": factors,
        "event_sequence": types,
        "official_method": "events-v1 deterministic rules",
        "causal_claim": "descriptive only until matched counterfactual evidence is attached",
    }


def cluster_diagnoses(records: Iterable[Dict[str, Any]], counterfactuals: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        diagnosis = record.get("diagnosis", {})
        grouped[diagnosis.get("classification", "unclassified")].append(record)
    evidence = counterfactuals or []
    clusters = []
    for classification, items in sorted(grouped.items()):
        terminals = Counter(item.get("terminal_reason", "unknown") for item in items)
        factors = Counter(
            factor.get("factor", "unknown")
            for item in items
            for factor in item.get("diagnosis", {}).get("likely_factors", [])
        )
        matching = [item for item in evidence if item.get("classification") == classification]
        clusters.append(
            {
                "classification": classification,
                "count": len(items),
                "terminal_reasons": dict(terminals),
                "common_factors": [{"factor": name, "support": count} for name, count in factors.most_common()],
                "counterfactual_evidence": matching,
                "confidence": "supported" if matching else ("repeated_descriptive" if len(items) >= 2 else "single_observation"),
                "representative_run": items[0].get("run_dir"),
            }
        )
    return {"schema_version": "1.0", "cluster_count": len(clusters), "clusters": clusters}
