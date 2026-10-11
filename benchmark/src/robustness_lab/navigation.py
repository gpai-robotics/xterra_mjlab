"""Navigation assistance for policies that consume body-frame commands."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


NAVIGATION_SCOPES = {"straight_only", "fixed_route", "route_choice"}
ROUTE_MATCH_TOLERANCE_M = 0.35


def _named_routes(routes: Any) -> List[Tuple[str, Any]]:
    """Return route names and values from public or model route containers."""

    if isinstance(routes, Mapping):
        return [(str(name), route) for name, route in routes.items()]

    named = []
    for route in routes:
        if isinstance(route, Mapping):
            name = route.get("name")
        else:
            name = getattr(route, "name", None)
        if name is None:
            raise ValueError("each candidate route requires a name")
        named.append((str(name), route))
    return named


def _route_waypoints(route: Any) -> Sequence[Sequence[float]]:
    if isinstance(route, Mapping) and "waypoints" in route:
        return route["waypoints"]
    if hasattr(route, "waypoints"):
        return route.waypoints
    return route


def guidance_for_scope(scope: str, routes: Any, goal: Sequence[float]) -> Dict[str, Any]:
    """Return only the navigation guidance permitted by ``scope``."""

    if scope not in NAVIGATION_SCOPES:
        raise ValueError("unknown navigation scope %r" % scope)
    if scope in {"straight_only", "route_choice"}:
        return {"waypoints": [], "goal": goal}

    for name, route in _named_routes(routes):
        route_kind = route.get("route_kind") if isinstance(route, Mapping) else getattr(route, "route_kind", None)
        if name == "central" or route_kind == "central":
            return {"waypoints": _route_waypoints(route), "goal": goal}
    raise ValueError("fixed_route guidance requires a central route")


def _point_to_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    if length_squared == 0.0:
        return math.hypot(px - ax, py - ay)
    projection = ((px - ax) * dx + (py - ay) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    return math.hypot(px - (ax + projection * dx), py - (ay + projection * dy))


def classify_route(
    trajectory: Sequence[Sequence[float]],
    routes: Any,
    tolerance: float = ROUTE_MATCH_TOLERANCE_M,
) -> Optional[str]:
    """Return the closest route when its mean planar error is acceptable."""

    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("route classification tolerance must be finite and non-negative")
    if not trajectory:
        return None

    planar_trajectory = []
    for point in trajectory:
        if len(point) < 2:
            raise ValueError("trajectory points require planar x and y coordinates")
        planar = (float(point[0]), float(point[1]))
        if not all(math.isfinite(value) for value in planar):
            raise ValueError("trajectory points must contain finite coordinates")
        planar_trajectory.append(planar)

    best_name = None
    best_mean_distance = math.inf
    for name, route in _named_routes(routes):
        waypoints = _route_waypoints(route)
        if len(waypoints) < 2:
            continue
        segments = list(zip(waypoints, waypoints[1:]))
        mean_distance = sum(
            min(_point_to_segment_distance(point, start, end) for start, end in segments)
            for point in planar_trajectory
        ) / len(planar_trajectory)
        if mean_distance < best_mean_distance:
            best_name = name
            best_mean_distance = mean_distance

    return best_name if best_mean_distance <= tolerance else None


def navigation_claim(scope: str, route_taken: Optional[str]) -> str:
    """Describe a traversal without conflating route following with choice."""

    if scope not in NAVIGATION_SCOPES:
        raise ValueError("unknown navigation scope %r" % scope)
    if scope == "route_choice":
        if route_taken is None:
            return "Policy made no recognised route choice."
        return "Policy chose route %r." % route_taken
    if scope == "fixed_route":
        if route_taken is None:
            return "Policy did not complete the supplied route."
        return "Policy followed supplied route %r." % route_taken
    return "Policy followed the straight-only route."


class RouteFollower:
    """Turn a planar waypoint route into a differential-style command."""

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        forward_speed: float = 0.6,
        max_yaw_rate: float = 1.0,
        arrival_radius: float = 0.15,
    ) -> None:
        self.waypoints = []
        for waypoint in waypoints:
            if len(waypoint) < 2:
                raise ValueError("route waypoints require planar x and y coordinates")
            point = [float(waypoint[0]), float(waypoint[1])]
            if not all(math.isfinite(value) for value in point):
                raise ValueError("route waypoints must contain finite coordinates")
            self.waypoints.append(point)
        if not self.waypoints:
            raise ValueError("route follower requires at least one waypoint")

        self.forward_speed = float(forward_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.arrival_radius = float(arrival_radius)
        if not math.isfinite(self.forward_speed):
            raise ValueError("forward_speed must be finite")
        if not math.isfinite(self.max_yaw_rate) or self.max_yaw_rate < 0.0:
            raise ValueError("max_yaw_rate must be finite and non-negative")
        if not math.isfinite(self.arrival_radius) or self.arrival_radius < 0.0:
            raise ValueError("arrival_radius must be finite and non-negative")

        self.index = 0
        self._last_heading = None
        self._terminal_heading = self._route_terminal_heading()

    def _route_terminal_heading(self):
        final_x, final_y = self.waypoints[-1]
        for prior_x, prior_y in reversed(self.waypoints[:-1]):
            dx = final_x - prior_x
            dy = final_y - prior_y
            if math.hypot(dx, dy) > 0.0:
                return math.atan2(dy, dx)
        return None

    def command(self, position: Sequence[float], yaw: float) -> List[float]:
        """Return ``[forward, lateral, yaw_rate]`` in the robot body frame."""

        if len(position) < 2:
            raise ValueError("position requires planar x and y coordinates")
        x, y = float(position[0]), float(position[1])
        yaw = float(yaw)
        if not all(math.isfinite(value) for value in (x, y, yaw)):
            raise ValueError("position and yaw must be finite")

        desired_heading = None
        while self.index < len(self.waypoints):
            target_x, target_y = self.waypoints[self.index]
            dx = target_x - x
            dy = target_y - y
            distance = math.hypot(dx, dy)
            passed = False
            if self.index > 0:
                prior_x, prior_y = self.waypoints[self.index - 1]
                segment_x, segment_y = target_x - prior_x, target_y - prior_y
                segment_norm = math.hypot(segment_x, segment_y)
                if segment_norm > 1e-9:
                    along = (
                        (x - target_x) * segment_x
                        + (y - target_y) * segment_y
                    ) / segment_norm
                    lateral = abs(
                        (x - target_x) * segment_y
                        - (y - target_y) * segment_x
                    ) / segment_norm
                    passed = along >= 0.0 and lateral <= max(
                        2.0 * self.arrival_radius, 0.35
                    )
            if distance > self.arrival_radius and not passed:
                desired_heading = math.atan2(dy, dx)
                self._last_heading = desired_heading
                break
            if distance > 0.0:
                self._last_heading = math.atan2(dy, dx)
            self.index += 1

        if desired_heading is None:
            desired_heading = self._terminal_heading
            if desired_heading is None:
                desired_heading = self._last_heading
            if desired_heading is None:
                desired_heading = yaw

        heading_error = math.atan2(
            math.sin(desired_heading - yaw), math.cos(desired_heading - yaw)
        )
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, heading_error))
        # Do not drive off a junction while trying to make a sharp turn.  This
        # matters for edge-covering routes, which legitimately contain 90 and
        # 180 degree turns when returning from a tree branch.  The command is
        # still a normal body-frame velocity command; only its forward part is
        # reduced until the robot faces the next waypoint.
        alignment = max(0.0, math.cos(heading_error))
        return [self.forward_speed * alignment, 0.0, yaw_rate]

    @property
    def completed(self) -> bool:
        """Whether every waypoint has been reached in sequence."""

        return self.index >= len(self.waypoints)
