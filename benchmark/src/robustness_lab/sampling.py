"""Reproducible multi-factor stressor sampling."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple


def latin_hypercube(bounds: Dict[str, Tuple[float, float]], count: int, seed: int = 0) -> List[Dict[str, float]]:
    if count < 1:
        raise ValueError("sample count must be positive")
    rng = random.Random(seed)
    columns = {}
    for name, (lower, upper) in bounds.items():
        values = [lower + (upper - lower) * (index + rng.random()) / count for index in range(count)]
        rng.shuffle(values)
        columns[name] = values
    return [{name: columns[name][index] for name in bounds} for index in range(count)]


def sobol(bounds: Dict[str, Tuple[float, float]], count: int, seed: int = 0) -> List[Dict[str, float]]:
    if count < 1:
        raise ValueError("sample count must be positive")
    names = list(bounds)
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
        points = sampler.random(count)
        rows = []
        for point in points:
            rows.append({name: bounds[name][0] + (bounds[name][1] - bounds[name][0]) * float(point[index]) for index, name in enumerate(names)})
        return rows
    except (ImportError, ModuleNotFoundError):
        return latin_hypercube(bounds, count, seed)


def one_factor_sweep(name: str, values: Sequence[float], baseline: Dict[str, float]) -> List[Dict[str, float]]:
    rows = []
    for value in values:
        row = dict(baseline)
        row[name] = float(value)
        rows.append(row)
    return rows

