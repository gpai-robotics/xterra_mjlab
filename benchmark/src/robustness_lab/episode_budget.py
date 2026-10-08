"""Per-episode time budgets derived from the route a robot must actually walk.

A single global ``timeout_s`` cannot serve both a 2.8 m single terrain and a
70 m composed world. Measured against the route follower's 0.6 m/s, the shipped
20 s budget was ample for RS terrains and impossible for composed ones: covering
68 m in 20 s needs 3.4 m/s, past what the reference policy declares. Every
L3-L7 level therefore failed by construction, and recorded that failure as
``timeout`` -- the same label a policy that cannot walk receives.

The budget scales with the route and is clamped at both ends. The floor keeps
short worlds generous rather than tight; the ceiling bounds the cost of a robot
that gets stuck but keeps inching forward, which no progress threshold can
reliably detect. At the measured ~167x realtime a 180 s episode costs about
1.4 s of wall clock, so a generous ceiling is close to free.
"""

from __future__ import annotations

import math
from typing import Any


DEFAULT_REFERENCE_SPEED_MPS = 0.6
DEFAULT_SLACK = 2.0
DEFAULT_MINIMUM_S = 40.0
DEFAULT_MAXIMUM_S = 180.0


def route_length_m(world: Any) -> float:
    """Planar length of the world's primary route, in metres."""

    routes = getattr(world, "routes", None)
    if not routes:
        return 0.0
    points = [list(point) for point in routes[0].waypoints]
    if len(points) < 2:
        return 0.0
    return sum(
        math.dist(points[index][:2], points[index + 1][:2])
        for index in range(len(points) - 1)
    )


def episode_timeout_s(
    world: Any,
    reference_speed_mps: float = DEFAULT_REFERENCE_SPEED_MPS,
    slack: float = DEFAULT_SLACK,
    minimum_s: float = DEFAULT_MINIMUM_S,
    maximum_s: float = DEFAULT_MAXIMUM_S,
) -> float:
    """Simulated-second budget for one episode in ``world``.

    ``slack`` is how much slower than ``reference_speed_mps`` a policy may be and
    still finish: at 2.0 a policy walking at half the commanded speed arrives
    exactly on the budget.
    """

    speed = float(reference_speed_mps)
    slack_value = float(slack)
    floor = float(minimum_s)
    ceiling = float(maximum_s)
    for name, value in (
        ("reference_speed_mps", speed),
        ("slack", slack_value),
        ("maximum_s", ceiling),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("%s must be positive and finite" % name)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("minimum_s must be non-negative and finite")
    if ceiling < floor:
        raise ValueError(
            "maximum_s (%.3f) must not be below minimum_s (%.3f)" % (ceiling, floor)
        )

    nominal = route_length_m(world) / speed
    return min(max(slack_value * nominal, floor), ceiling)
