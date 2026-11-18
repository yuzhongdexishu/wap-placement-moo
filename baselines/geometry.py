"""Shared WAP-geometry checks and repair logic for the baseline comparisons.

Ported verbatim from a_V16_EXPERIMENTS_REFACTORED_V2 experiments/core/geometry.py.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence


def point_inside_aabb(
    point: Sequence[float], aabb: dict[str, Sequence[float]], padding: float = 0.05
) -> bool:
    return all(
        float(aabb["min"][i]) - padding
        <= float(point[i])
        <= float(aabb["max"][i]) + padding
        for i in range(3)
    )


def is_valid_wap(
    point: Sequence[float],
    bounds: dict[str, Sequence[float]],
    object_bounds: Sequence[dict[str, Any]],
    obstacle_padding: float = 0.05,
) -> bool:
    lo, hi = bounds["min"], bounds["max"]
    tolerance = 1e-6
    if not all(
        float(lo[i]) - tolerance <= float(point[i]) <= float(hi[i]) + tolerance
        for i in range(3)
    ):
        return False
    return not any(
        point_inside_aabb(point, obj["bounds"], obstacle_padding)
        for obj in object_bounds
    )


def random_wap_point(
    rng: random.Random,
    bounds: dict[str, Sequence[float]],
    height: float | None,
    object_bounds: Sequence[dict[str, Any]],
    max_attempts: int = 2000,
    *,
    z_margin_m: float = 0.35,
    prefer_boundary: bool = False,
) -> list[float]:
    lo, hi = [list(v) for v in (bounds["min"], bounds["max"])]
    z_lo = float(lo[2]) + min(
        z_margin_m, max((float(hi[2]) - float(lo[2])) * 0.2, 0.0)
    )
    z_hi = float(hi[2]) - min(
        z_margin_m, max((float(hi[2]) - float(lo[2])) * 0.2, 0.0)
    )
    for _ in range(max_attempts):
        if prefer_boundary:
            face = rng.randrange(4)
            inset = 0.05
            if face == 0:
                x, y = float(lo[0]) + inset, rng.uniform(float(lo[1]), float(hi[1]))
            elif face == 1:
                x, y = float(hi[0]) - inset, rng.uniform(float(lo[1]), float(hi[1]))
            elif face == 2:
                x, y = rng.uniform(float(lo[0]), float(hi[0])), float(lo[1]) + inset
            else:
                x, y = rng.uniform(float(lo[0]), float(hi[0])), float(hi[1]) - inset
        else:
            x = rng.uniform(float(lo[0]), float(hi[0]))
            y = rng.uniform(float(lo[1]), float(hi[1]))
        z = float(height) if height is not None else rng.uniform(z_lo, z_hi)
        point = [x, y, z]
        if is_valid_wap(point, bounds, object_bounds):
            return [round(v, 6) for v in point]
    z = float(height) if height is not None else rng.uniform(z_lo, z_hi)
    return [
        round(rng.uniform(float(lo[0]), float(hi[0])), 6),
        round(rng.uniform(float(lo[1]), float(hi[1])), 6),
        round(z, 6),
    ]


def repair_layout(
    layout: Sequence[Sequence[float]],
    rng: random.Random,
    bounds: dict[str, Sequence[float]],
    height: float | None,
    object_bounds: Sequence[dict[str, Any]],
    minimum_distance: float,
    *,
    z_margin_m: float = 0.35,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    repaired: list[list[float]] = []
    events: list[dict[str, Any]] = []
    lo, hi = bounds["min"], bounds["max"]
    for index, original in enumerate(layout):
        point = [
            min(max(float(original[0]), float(lo[0])), float(hi[0])),
            min(max(float(original[1]), float(lo[1])), float(hi[1])),
            float(height)
            if height is not None
            else min(max(float(original[2]), float(lo[2])), float(hi[2])),
        ]
        needs_repair = not is_valid_wap(point, bounds, object_bounds)
        if not needs_repair:
            needs_repair = any(
                math.dist(point, other) < minimum_distance for other in repaired
            )
        if needs_repair:
            before = point[:]
            for attempt in range(2000):
                candidate = random_wap_point(
                    rng,
                    bounds,
                    height,
                    object_bounds,
                    z_margin_m=z_margin_m,
                    prefer_boundary=(index + attempt) % 2 == 0,
                )
                if all(
                    math.dist(candidate, other) >= minimum_distance
                    for other in repaired
                ):
                    point = candidate
                    break
            events.append({"wap_index": index, "before": before, "after": point})
        repaired.append([round(v, 6) for v in point])
    return repaired, events


def constraint_report(
    layout: Sequence[Sequence[float]],
    bounds: dict[str, Sequence[float]],
    object_bounds: Sequence[dict[str, Any]],
    minimum_distance: float,
    *,
    required_floors: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    invalid_points = [
        index
        for index, point in enumerate(layout)
        if not is_valid_wap(point, bounds, object_bounds)
    ]
    short_pairs: list[list[int]] = []
    for i, first in enumerate(layout):
        for j in range(i + 1, len(layout)):
            if math.dist(first, layout[j]) < minimum_distance:
                short_pairs.append([i, j])
    z_values = [float(point[2]) for point in layout]
    floor_counts: dict[str, int] = {}
    if required_floors:
        for index, floor in enumerate(required_floors):
            lower, upper = float(floor["z_min_m"]), float(floor["z_max_m"])
            floor_counts[str(floor["floor_id"])] = sum(
                lower <= float(point[2]) < upper
                or (
                    index == len(required_floors) - 1
                    and math.isclose(float(point[2]), upper)
                )
                for point in layout
            )
    missing_floors = [
        floor_id for floor_id, count in floor_counts.items() if count == 0
    ]
    return {
        "valid": not invalid_points and not short_pairs and not missing_floors,
        "invalid_wap_indices": invalid_points,
        "minimum_distance_violations": short_pairs,
        "z_min_m": min(z_values) if z_values else None,
        "z_max_m": max(z_values) if z_values else None,
        "distinct_z_levels_0_1m": len({round(value, 1) for value in z_values}),
        "floor_wap_counts": floor_counts,
        "missing_required_floors": missing_floors,
    }
