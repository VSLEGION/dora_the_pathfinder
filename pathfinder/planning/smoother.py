"""
Path smoothing module for realistic flight paths.

STATUS: Beta - under active development.

Converts jagged A* paths into smooth, flyable trajectories
that respect no-fly zone constraints using tangent-arc geometry.

Constraints to be set for this:
Bank angle influenced turning radius. (dynamic)
Flight track and no-fly zone proximity optimization
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from scipy.interpolate import splprep, splev
from pathfinder.models.environment import Environment, NoFlyZone


def _point_to_segment_distance(point: Tuple[float, float],
                               p1: Tuple[float, float],
                               p2: Tuple[float, float]) -> float:
    """Perpendicular distance from a point to a line segment."""
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.sqrt((px - x1)**2 + (py - y1)**2)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class PathSmoother:
    """
    Smooths discrete A* paths into continuous, flyable trajectories.

    When an environment with no-fly zones is provided, builds a
    piecewise path of straight segments and tangent arcs (like a
    CATIA spline sketch), then optionally applies a final gentle
    smoothing pass for C2 continuity.
    """

    def __init__(self, environment: Optional[Environment] = None,
                 smoothing_factor: float = 0.0,
                 num_points: int = 100):
        """
        Initialize path smoother.

        Args:
            environment: Optional environment for collision checking
            smoothing_factor: Spline smoothing (0=interpolate, >0=smooth)
            num_points: Number of points in smoothed path
        """
        self.environment = environment
        self.smoothing_factor = smoothing_factor
        self.num_points = num_points

    def smooth_path(self, path: List[Tuple[float, float]],
                   method: str = 'spline') -> List[Tuple[float, float]]:
        """
        Smooth a discrete path into a continuous trajectory.

        Args:
            path: Original path from A* (jagged)
            method: Smoothing method ('spline', 'bezier', 'simple', 'multi')

        Returns:
            Smoothed path with more points
        """
        if len(path) < 3:
            return path

        if method == 'spline':
            return self._smooth_spline(path)
        elif method == 'bezier':
            return self._smooth_bezier(path)
        elif method == 'simple':
            return self._smooth_simple(path)
        elif method == 'multi':
            return self._smooth_multi_pass(path)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    def _smooth_multi_pass(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Multi-pass smoothing: simplify then spline."""
        simplified = self._simplify_path(path, tolerance=1.0)
        if len(simplified) >= 4:
            return self._smooth_spline(simplified)
        return simplified

    def _simplify_path(self, path: List[Tuple[float, float]],
                      tolerance: float = 1.0) -> List[Tuple[float, float]]:
        """
        Simplify path by removing nearly-collinear points.
        Uses Ramer-Douglas-Peucker concept.
        """
        if len(path) < 3:
            return path

        simplified = [path[0]]

        for i in range(1, len(path) - 1):
            prev = np.array(simplified[-1])
            curr = np.array(path[i])
            next_pt = np.array(path[i + 1])

            line_vec = next_pt - prev
            line_len = np.linalg.norm(line_vec)

            if line_len < 1e-6:
                continue

            point_vec = curr - prev
            cross = abs(np.cross(line_vec, point_vec))
            distance = cross / line_len

            if distance > tolerance:
                simplified.append(path[i])

        simplified.append(path[-1])
        return simplified

    # ------------------------------------------------------------------
    # Core smoothing: obstacle-aware vs plain spline
    # ------------------------------------------------------------------

    def _smooth_spline(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Smooth path using cubic B-spline.

        When obstacles are present, uses the tangent-arc approach to
        build a geometrically correct skeleton, then applies a gentle
        smoothing spline over the skeleton for C2 continuity.
        """
        if self.environment and self.environment.no_fly_zones:
            return self._smooth_obstacle_aware(path)
        return self._fit_spline(path)

    def _fit_spline(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Fit a cubic B-spline through control points (no obstacle checks)."""
        path_array = np.array(path)
        x = path_array[:, 0]
        y = path_array[:, 1]

        x, y = self._remove_duplicates(x, y)

        if len(x) < 4:
            return path

        try:
            distances = np.zeros(len(x))
            distances[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            distances = np.cumsum(distances)

            if distances[-1] > 0:
                distances = distances / distances[-1]

            tck, u = splprep(
                [x, y],
                u=distances,
                s=self.smoothing_factor,
                k=min(3, len(x) - 1)
            )

            u_new = np.linspace(0, 1, self.num_points)
            x_new, y_new = splev(u_new, tck)

            valid_mask = np.isfinite(x_new) & np.isfinite(y_new)
            x_new = x_new[valid_mask]
            y_new = y_new[valid_mask]

            smoothed_path = list(zip(x_new.tolist(), y_new.tolist()))

            if len(smoothed_path) < len(path) // 2:
                return path

            return smoothed_path

        except Exception:
            return path

    # ------------------------------------------------------------------
    # Tangent-arc obstacle avoidance (CATIA-style)
    # ------------------------------------------------------------------
    #
    # HOW IT WORKS (the geometry lesson):
    #
    # Given two waypoints A and B with a circular no-fly zone O
    # between them (center C, radius R, clearance radius R' = R + margin):
    #
    #   1. The A* path already tells us WHICH SIDE to go around
    #      (left or right of the obstacle).
    #
    #   2. We compute the two TANGENT POINTS T1, T2 on the clearance
    #      circle where:
    #      - Line A->T1 is tangent to the circle (smooth entry)
    #      - Line T2->B is tangent to the circle (smooth exit)
    #
    #   3. Between T1 and T2, we trace a CIRCULAR ARC along the
    #      clearance circle. This arc has constant curvature = 1/R',
    #      which means a constant bank angle for the UAV.
    #
    #   4. The final path is: A -> straight -> T1 -> arc -> T2 -> straight -> B
    #      with smooth heading transitions at T1 and T2 (tangent = no kink).
    #
    #   5. Finally, we apply a GENTLE smoothing spline over the whole
    #      skeleton to get C2 continuity (smooth acceleration), which
    #      softens the tangent points slightly.
    #
    # This is exactly how CATIA, SolidWorks, and real flight planners
    # construct smooth paths around obstacles.
    # ------------------------------------------------------------------

    def _smooth_obstacle_aware(self, path: List[Tuple[float, float]],
                                safety_margin: float = 8.0) -> List[Tuple[float, float]]:
        """
        Build a CATIA-style smooth path: straight segments connected
        by tangent circular arcs around obstacles.
        """
        # Step 1: Simplify the dense A* path to key waypoints
        simplified = self._simplify_path(path, tolerance=2.0)
        if len(simplified) < 3:
            simplified = [path[0], path[len(path) // 2], path[-1]]

        # Step 2: Remove simplified waypoints that are INSIDE no-fly zones
        # (these are artifacts of simplification that would create invalid arcs)
        if self.environment:
            filtered = [simplified[0]]  # always keep start
            for i in range(1, len(simplified) - 1):
                pt = simplified[i]
                inside = False
                for zone in self.environment.no_fly_zones:
                    if zone.contains_point(pt):
                        inside = True
                        break
                if not inside:
                    filtered.append(pt)
            filtered.append(simplified[-1])  # always keep end
            if len(filtered) >= 3:
                simplified = filtered

        # Step 3: Build the piecewise tangent-arc skeleton
        # Pass the ORIGINAL dense A* path for accurate side detection
        skeleton = self._build_tangent_arc_path(simplified, safety_margin, path)

        # Step 3: Validate skeleton against obstacles
        skeleton = self._enforce_clearance(skeleton, safety_margin)

        # Step 4: Apply gentle smoothing over the skeleton
        # Try spline first for C2 continuity; if it violates, use Chaikin
        if len(skeleton) >= 4:
            spline_result = self._fit_final_spline(skeleton)
            # Check if spline introduced violations
            has_violations = False
            if self.environment:
                for i in range(len(spline_result) - 1):
                    if not self.environment.is_valid_segment(spline_result[i], spline_result[i + 1]):
                        has_violations = True
                        break

            if not has_violations:
                result = spline_result
            else:
                # Spline overshoots  -- use Chaikin which preserves shape
                result = self._chaikin_smooth(skeleton, iterations=3)
        else:
            result = skeleton

        return result

    def _build_tangent_arc_path(self, waypoints: List[Tuple[float, float]],
                                 safety_margin: float,
                                 original_path: Optional[List[Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
        """
        Build a piecewise path of straight segments and tangent arcs.

        Key insight: when multiple consecutive segments are blocked by
        the SAME zone, we merge them into a single tangent arc passage.
        This prevents the double-arc loops seen in naive per-segment processing.

        Algorithm:
        1. For each segment, identify which zone (if any) blocks it
        2. Group consecutive segments blocked by the same zone
        3. For each group: compute one tangent arc from the entry waypoint
           to the exit waypoint around that zone
        4. For unblocked segments: straight line (with optional midpoint)
        """
        if not self.environment or len(waypoints) < 2:
            return list(waypoints)

        # Step 1: classify each segment
        segment_zones = []
        for i in range(len(waypoints) - 1):
            zone = self._find_blocking_zone(waypoints[i], waypoints[i + 1])
            segment_zones.append(zone)

        # Step 2: group consecutive segments by zone
        skeleton = [waypoints[0]]
        i = 0

        while i < len(segment_zones):
            zone = segment_zones[i]

            if zone is None:
                # Unblocked segment  -- straight line
                p_start = waypoints[i]
                p_end = waypoints[i + 1]
                seg_len = _dist(p_start, p_end)
                if seg_len > 30:
                    mid = ((p_start[0] + p_end[0]) / 2,
                           (p_start[1] + p_end[1]) / 2)
                    skeleton.append(mid)
                skeleton.append(p_end)
                i += 1
            else:
                # Find the full run of consecutive segments blocked by this zone
                run_start = i
                while (i < len(segment_zones) and
                       segment_zones[i] is not None and
                       segment_zones[i].center == zone.center):
                    i += 1
                run_end = i  # exclusive

                # The passage goes from waypoints[run_start] to waypoints[run_end]
                p_entry = waypoints[run_start]
                p_exit = waypoints[run_end]

                arc_points = self._compute_tangent_arc(
                    p_entry, p_exit, zone, safety_margin,
                    original_path if original_path else waypoints
                )
                skeleton.extend(arc_points)
                skeleton.append(p_exit)

        return skeleton

    def _find_blocking_zone(self, p1: Tuple[float, float],
                             p2: Tuple[float, float]) -> Optional[NoFlyZone]:
        """Find the first no-fly zone that blocks the straight line from p1 to p2."""
        if not self.environment:
            return None

        for zone in self.environment.no_fly_zones:
            if zone.intersects_segment(p1, p2):
                return zone
            # Also check if the segment passes very close
            dist = _point_to_segment_distance(zone.center, p1, p2)
            if dist < zone.radius_m + 2.0:
                return zone

        return None

    def _compute_tangent_arc(self, p_start: Tuple[float, float],
                              p_end: Tuple[float, float],
                              zone: NoFlyZone,
                              safety_margin: float,
                              full_path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Compute the tangent entry point, circular arc, and tangent exit point
        around a no-fly zone.

        GEOMETRY:
        - Clearance circle: center C, radius R' = zone.radius + safety_margin
        - From external point P to circle, the tangent point T satisfies:
          |PT| perpendicular to CT, so |PT|^2 = |PC|^2 - R'^2
        - Tangent angle: alpha = asin(R' / |PC|)

        Returns a list of points: [tangent_entry, arc_pt1, ..., arc_ptN, tangent_exit]
        """
        cx, cy = zone.center
        R = zone.radius_m + safety_margin

        # Determine which side the A* path goes around
        side = self._determine_side(p_start, p_end, zone, full_path)

        # Compute tangent point from p_start to the clearance circle
        t1 = self._tangent_point(p_start, (cx, cy), R, side)

        # Compute tangent point from p_end to the clearance circle
        # (approaching from the other direction, so flip the side logic)
        t2 = self._tangent_point(p_end, (cx, cy), R, side)

        # Generate arc between t1 and t2 on the clearance circle
        angle1 = math.atan2(t1[1] - cy, t1[0] - cx)
        angle2 = math.atan2(t2[1] - cy, t2[0] - cx)

        # Sweep in the direction consistent with the chosen side
        arc_points = self._generate_arc(
            (cx, cy), R, angle1, angle2, side
        )

        return [t1] + arc_points + [t2]

    def _determine_side(self, p_start: Tuple[float, float],
                         p_end: Tuple[float, float],
                         zone: NoFlyZone,
                         full_path: List[Tuple[float, float]]) -> int:
        """
        Determine which side of the obstacle the path goes around.

        The most reliable method: find the path point closest to the
        obstacle center, then check which side of the direct line
        (p_start -> p_end) that point is on. This uses the actual A*
        path geometry rather than assumptions.

        Returns +1 for left (CCW), -1 for right (CW).
        """
        cx, cy = zone.center

        # Find the closest point on the A* path to the zone center
        # This is the point that "wraps around" the obstacle
        closest_pt = None
        min_dist = float('inf')
        for p in full_path:
            d = _dist(p, (cx, cy))
            if d < min_dist:
                min_dist = d
                closest_pt = p

        if closest_pt is None:
            closest_pt = ((p_start[0] + p_end[0]) / 2,
                          (p_start[1] + p_end[1]) / 2)

        # Which side of the line (p_start -> p_end) is closest_pt on?
        # Cross product: (p_end - p_start) x (closest_pt - p_start)
        # Positive = left side, Negative = right side
        dx = p_end[0] - p_start[0]
        dy = p_end[1] - p_start[1]
        cpx = closest_pt[0] - p_start[0]
        cpy = closest_pt[1] - p_start[1]
        cross = dx * cpy - dy * cpx

        return 1 if cross > 0 else -1

    def _tangent_point(self, external_pt: Tuple[float, float],
                        center: Tuple[float, float],
                        radius: float,
                        side: int) -> Tuple[float, float]:
        """
        Compute the tangent point from an external point to a circle.

        GEOMETRY:
        Given external point P and circle with center C and radius R:
        - Distance d = |PC|
        - If d <= R, point is inside circle  -- return closest boundary point
        - Tangent length L = sqrt(d^2 - R^2)
        - Tangent angle alpha = atan2(R, L)  (or equivalently asin(R/d))
        - The tangent point is at angle (angle_to_center +/- alpha) on the circle

        The 'side' parameter determines which of the two tangent points to use.
        """
        px, py = external_pt
        cx, cy = center

        dx = cx - px
        dy = cy - py
        d = math.sqrt(dx * dx + dy * dy)

        if d <= radius:
            # Point is inside or on the circle  -- project outward
            if d < 1e-6:
                return (cx + radius, cy)
            return (cx - dx / d * radius, cy - dy / d * radius)

        # The tangent point on the circle, as seen from center, is at angle:
        #   angle_from_center_to_external +/- beta
        # where beta = acos(R / d) is the angle at the center between
        # the line to the external point and the line to the tangent point.
        #
        #        P (external)
        #       /|
        #      / |  L = tangent length = sqrt(d^2 - R^2)
        #     /  |
        #    /   | R
        #   C----T (tangent point)
        #     beta = acos(R/d)
        #
        angle_from_center = math.atan2(py - cy, px - cx)
        beta = math.acos(min(1.0, radius / d))

        tangent_angle_on_circle = angle_from_center + side * beta
        tx = cx + radius * math.cos(tangent_angle_on_circle)
        ty = cy + radius * math.sin(tangent_angle_on_circle)

        return (tx, ty)

    def _generate_arc(self, center: Tuple[float, float],
                       radius: float,
                       angle_start: float,
                       angle_end: float,
                       side: int,
                       points_per_radian: float = 8.0) -> List[Tuple[float, float]]:
        """
        Generate evenly-spaced points along a circular arc.

        Always takes the SHORTER arc between angle_start and angle_end
        that is on the correct side. The 'side' parameter is used to
        break ties when the arc is exactly 180 degrees.
        """
        cx, cy = center

        # Compute the shortest angular difference
        diff = _normalize_angle(angle_end - angle_start)

        # diff is in [-pi, pi], so abs(diff) <= pi = shortest arc
        # If exactly 0 or pi, use side to break the tie
        if abs(diff) < 1e-6:
            return []  # T1 and T2 are the same point

        # For the short arc, diff already gives us the right direction
        # (positive = CCW, negative = CW)

        # Number of points proportional to arc length
        num_points = max(3, int(abs(diff) * points_per_radian))

        points = []
        for k in range(1, num_points):  # exclude endpoints (added separately)
            t = k / num_points
            angle = angle_start + t * diff
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))

        return points

    def _enforce_clearance(self, path: List[Tuple[float, float]],
                           min_clearance: float) -> List[Tuple[float, float]]:
        """
        Push any path points that are too close to no-fly zones outward.
        """
        if not self.environment:
            return path

        result = list(path)
        for i in range(1, len(result) - 1):
            pt = result[i]
            for zone in self.environment.no_fly_zones:
                cx, cy = zone.center
                dx = pt[0] - cx
                dy = pt[1] - cy
                dist = math.sqrt(dx * dx + dy * dy)
                required = zone.radius_m + min_clearance

                if dist < required and dist > 1e-6:
                    scale = required / dist
                    result[i] = (cx + dx * scale, cy + dy * scale)

        return result

    def _fit_final_spline(self, skeleton: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Apply a gentle smoothing spline over the tangent-arc skeleton.

        Uses a small smoothing factor to soften tangent-point kinks
        into C2-continuous curves while staying close to the skeleton.
        The skeleton already has the right shape, so we just need to
        smooth the transitions.
        """
        path_array = np.array(skeleton)
        x = path_array[:, 0]
        y = path_array[:, 1]

        x, y = self._remove_duplicates(x, y)

        if len(x) < 4:
            return skeleton

        try:
            distances = np.zeros(len(x))
            distances[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
            distances = np.cumsum(distances)

            if distances[-1] > 0:
                distances = distances / distances[-1]

            # Smoothing factor scaled to path length.
            # Small enough to preserve the tangent-arc shape,
            # large enough to remove kinks at tangent points.
            total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            s_factor = total_length * 0.01

            tck, u = splprep(
                [x, y],
                u=distances,
                s=s_factor,
                k=min(3, len(x) - 1)
            )

            u_new = np.linspace(0, 1, self.num_points)
            x_new, y_new = splev(u_new, tck)

            valid_mask = np.isfinite(x_new) & np.isfinite(y_new)
            x_new = x_new[valid_mask]
            y_new = y_new[valid_mask]

            return list(zip(x_new.tolist(), y_new.tolist()))

        except Exception:
            return skeleton

    def _post_validate(self, smoothed: List[Tuple[float, float]],
                        original_path: List[Tuple[float, float]],
                        safety_margin: float) -> List[Tuple[float, float]]:
        """
        Final validation pass. If the smoothing spline introduced any
        violations, fall back to the Chaikin-smoothed A* path.
        """
        if not self.environment:
            return smoothed

        for i in range(len(smoothed) - 1):
            p1 = smoothed[i]
            p2 = smoothed[i + 1]
            if not self.environment.is_valid_segment(p1, p2):
                # Spline introduced a violation  -- fall back
                return self._chaikin_smooth(original_path, iterations=4)

        return smoothed

    def _chaikin_smooth(self, path: List[Tuple[float, float]],
                        iterations: int = 4) -> List[Tuple[float, float]]:
        """
        Chaikin corner-cutting as a safe fallback.

        Only interpolates between consecutive A* points, so if each
        original segment is valid, the subdivided path is also valid
        for convex obstacles (circles).
        """
        result = list(path)

        for _ in range(iterations):
            new_path = [result[0]]
            for i in range(len(result) - 1):
                p1 = result[i]
                p2 = result[i + 1]
                q = (0.75 * p1[0] + 0.25 * p2[0], 0.75 * p1[1] + 0.25 * p2[1])
                r = (0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1])
                new_path.extend([q, r])
            new_path.append(result[-1])
            result = new_path

        if self.environment:
            validated = [result[0]]
            for i in range(1, len(result)):
                if self.environment.is_valid_segment(validated[-1], result[i]):
                    validated.append(result[i])
            if len(validated) >= len(path):
                return validated

        return result

    # ------------------------------------------------------------------
    # Non-obstacle smoothing methods
    # ------------------------------------------------------------------

    def _smooth_bezier(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Smooth path using Bezier curves between waypoints."""
        if len(path) < 3:
            return path

        smoothed = []
        for i in range(len(path) - 1):
            p0 = np.array(path[i])
            p3 = np.array(path[i + 1])
            direction = p3 - p0
            p1 = p0 + direction / 3
            p2 = p0 + 2 * direction / 3
            for t in np.linspace(0, 1, self.num_points // len(path)):
                point = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
                smoothed.append(tuple(point))
        return smoothed

    def _smooth_simple(self, path: List[Tuple[float, float]],
                      window_size: int = 5) -> List[Tuple[float, float]]:
        """Simple moving average smoothing."""
        if len(path) < window_size:
            return path

        path_array = np.array(path)
        smoothed = []
        for i in range(len(path)):
            start = max(0, i - window_size // 2)
            end = min(len(path), i + window_size // 2 + 1)
            avg_point = np.mean(path_array[start:end], axis=0)
            smoothed.append(tuple(avg_point))
        return smoothed

    def _remove_duplicates(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove consecutive duplicate points."""
        unique_points = []
        for i in range(len(x)):
            if i == 0 or x[i] != x[i-1] or y[i] != y[i-1]:
                unique_points.append((x[i], y[i]))

        if len(unique_points) < 2:
            return x, y

        unique_array = np.array(unique_points)
        return unique_array[:, 0], unique_array[:, 1]

    # ------------------------------------------------------------------
    # Curvature analysis
    # ------------------------------------------------------------------

    def calculate_path_curvature(self, path: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate curvature at each point.

        Curvature k = |d_theta/ds| where theta is heading angle, s is arc length.

        Returns:
            List of curvature values (1/meters)
        """
        if len(path) < 3:
            return [0.0] * len(path)

        path_array = np.array(path)
        curvatures = []

        for i in range(1, len(path) - 1):
            p0 = path_array[i - 1]
            p1 = path_array[i]
            p2 = path_array[i + 1]

            v1 = p1 - p0
            v2 = p2 - p1

            d1 = np.linalg.norm(v1)
            d2 = np.linalg.norm(v2)

            if d1 < 1e-6 or d2 < 1e-6:
                curvatures.append(0.0)
                continue

            cos_angle = np.dot(v1, v2) / (d1 * d2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            curvature = angle / ((d1 + d2) / 2)
            curvatures.append(curvature)

        curvatures.insert(0, curvatures[0] if curvatures else 0.0)
        curvatures.append(curvatures[-1] if curvatures else 0.0)

        return curvatures

    def enforce_turn_radius(self, path: List[Tuple[float, float]],
                           min_turn_radius: float = 50.0) -> List[Tuple[float, float]]:
        """
        Future: Enforce minimum turn radius constraint.

        Args:
            path: Input path
            min_turn_radius: Minimum turn radius in meters

        Returns:
            Path with turn radius constraints enforced
        """
        # TODO: Implement turn radius constraints
        curvatures = self.calculate_path_curvature(path)
        max_curvature = 1.0 / min_turn_radius
        violations = [i for i, k in enumerate(curvatures) if k > max_curvature]
        return path


def smooth_path_simple(path: List[Tuple[float, float]],
                      num_points: int = 100) -> List[Tuple[float, float]]:
    """
    Quick path smoothing function (convenience wrapper).

    Args:
        path: Original discrete path
        num_points: Number of points in smoothed path

    Returns:
        Smoothed path
    """
    smoother = PathSmoother(num_points=num_points)
    return smoother.smooth_path(path, method='spline')
