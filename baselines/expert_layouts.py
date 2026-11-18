"""EbM1 / EbM2 / EbM3 manufacturer-inspired expert deployment heuristics (V17 baselines).

EbM = Expert-based Model. The three variants are manufacturer-inspired
reproducible research heuristics: EbM1 (corridor-centred, Huawei-inspired),
EbM2 (room-entry/intersection, Akuvox-inspired), EbM3 (zonal grid / vertical
traffic, Siemens-inspired). Adapted and simplified from
a_V16_EXPERIMENTS_REFACTORED_V2 experiments/expert/layout_rules.py. V17
problem structure: ``problem["objects"]`` is top-level and scene zones use
the ``"scenario"`` key; the SC1 baseline emphasizes only the priority
scenario's regions (F-SC1 for frame, E-SC1 for enclosure), matching the PM
run and the geometric baselines.
"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Sequence

from .geometry import constraint_report, is_valid_wap, random_wap_point, repair_layout

METHOD_DESCRIPTIONS = {
    "EbM1": (
        "Manufacturer-inspired corridor-centred heuristic prioritizing longitudinal circulation, "
        "major entrances, building edges, and the SC1 priority construction scenario (F-SC1 / E-SC1). "
        "It is a reproducible research heuristic and is not an official Huawei design."),
    "EbM2": (
        "Manufacturer-inspired room-entry and corridor-intersection heuristic prioritizing scene "
        "approaches, circulation junctions, and the SC1 priority construction scenario (F-SC1 / E-SC1). "
        "It is a reproducible research heuristic and is not an official Akuvox design."),
    "EbM3": (
        "Manufacturer-inspired zonal grid and vertical-traffic heuristic using balanced spatial "
        "cells, multiple elevation bands, and vertical circulation anchors. It is a reproducible "
        "research heuristic and is not an official Siemens design."),
}

EXPERT_METHODS = ("EbM1", "EbM2", "EbM3")
DEFAULT_MINIMUM_WAP_DISTANCE_M = 4.0
DEFAULT_RANDOM_SEED = 20260718


def _fraction(lo: float, hi: float, f: float) -> float:
    return float(lo) + float(f) * (float(hi) - float(lo))


def _z_levels(bounds: dict[str, Sequence[float]], count: int = 3) -> list[float]:
    lo, hi = bounds["min"], bounds["max"]
    margin = min(0.5, max((float(hi[2]) - float(lo[2])) * 0.08, 0.1))
    if count == 1:
        return [(float(lo[2]) + float(hi[2])) / 2.0]
    return [
        _fraction(float(lo[2]) + margin, float(hi[2]) - margin, (i + 0.5) / count)
        for i in range(count)
    ]


def _zone_center(zone: dict[str, Any], z: float) -> list[float]:
    return [
        (float(zone["x_min"]) + float(zone["x_max"])) / 2.0,
        (float(zone["y_min"]) + float(zone["y_max"])) / 2.0,
        float(z),
    ]


def _hed_candidates(problem: dict[str, Any]) -> list[list[float]]:
    """Corridor-centred and entrance-oriented candidates; only the SC1 priority
    scenario's zones are used (F-SC1 / E-SC1 per construction stage)."""
    opt = problem["optimization_problem"]
    b = opt["deployable_bounds"]
    lo, hi = b["min"], b["max"]
    zs = _z_levels(b, 3)
    points: list[list[float]] = []
    # Every priority-scenario zone centre on three elevations.
    for zone in opt["spatial_zones"]:
        if zone.get("scenario") != opt.get("priority_scenario"):
            continue
        for z in zs:
            points.append(_zone_center(zone, z))
    # Main circulation axes: four longitudinal stations on three elevations.
    for layer, z in enumerate(zs):
        y = _fraction(lo[1], hi[1], 0.50 if layer == 1 else (0.36 if layer == 0 else 0.64))
        for xf in (0.12, 0.37, 0.63, 0.88):
            points.append([_fraction(lo[0], hi[0], xf), y, z])
    # Major entry and edge reinforcement.
    mid_z = zs[1]
    points.extend([
        [_fraction(lo[0], hi[0], 0.03), _fraction(lo[1], hi[1], 0.50), mid_z],
        [_fraction(lo[0], hi[0], 0.97), _fraction(lo[1], hi[1], 0.50), mid_z],
        [_fraction(lo[0], hi[0], 0.50), _fraction(lo[1], hi[1], 0.04), mid_z],
        [_fraction(lo[0], hi[0], 0.50), _fraction(lo[1], hi[1], 0.96), mid_z],
    ])
    return points


def _aed_candidates(problem: dict[str, Any]) -> list[list[float]]:
    """Scene-entry, corridor-intersection, and device-area candidates for the
    SC1 priority scenario's zones."""
    opt = problem["optimization_problem"]
    b = opt["deployable_bounds"]
    lo, hi = b["min"], b["max"]
    zs = _z_levels(b, 3)
    zones = [zone for zone in opt["spatial_zones"] if zone.get("scenario") == opt.get("priority_scenario")]
    building_mid_y = (float(lo[1]) + float(hi[1])) / 2.0
    points: list[list[float]] = []
    # Scene centre and one approach point toward the building middle, on every
    # elevation, for every priority-scenario zone.
    for zone in zones:
        for z in zs:
            points.append(_zone_center(zone, z))
            zone_mid_y = (float(zone["y_min"]) + float(zone["y_max"])) / 2.0
            approach_y = float(zone["y_max"] if building_mid_y >= zone_mid_y else zone["y_min"])
            points.append([(float(zone["x_min"]) + float(zone["x_max"])) / 2.0, approach_y, z])
    # One entrance-facing point for every priority-scenario zone.
    for idx, zone in enumerate(zones):
        z = zs[idx % len(zs)]
        zone_mid_y = (float(zone["y_min"]) + float(zone["y_max"])) / 2.0
        entry_y = float(zone["y_max"] if building_mid_y >= zone_mid_y else zone["y_min"])
        points.append([(float(zone["x_min"]) + float(zone["x_max"])) / 2.0, entry_y, z])
    # Corridor intersections and building-centre junctions.
    for z in zs:
        for xf in (0.20, 0.40, 0.60, 0.80):
            points.append([_fraction(lo[0], hi[0], xf), building_mid_y, z])
    return points


def _sed_candidates(problem: dict[str, Any]) -> list[list[float]]:
    """Balanced zonal grid with vertical-traffic anchors."""
    opt = problem["optimization_problem"]
    b = opt["deployable_bounds"]
    lo, hi = b["min"], b["max"]
    zs = _z_levels(b, 3)
    points: list[list[float]] = []
    # Balanced 4x3 zoning; z cycles to avoid a single-height deployment.
    k = 0
    for yf in (0.18, 0.50, 0.82):
        for xf in (0.12, 0.37, 0.63, 0.88):
            points.append([_fraction(lo[0], hi[0], xf), _fraction(lo[1], hi[1], yf), zs[k % 3]])
            k += 1
    # Vertical circulation anchors at two cores and the building centre.
    for xf, yf in ((0.18, 0.50), (0.82, 0.50), (0.50, 0.50)):
        for z in zs:
            points.append([_fraction(lo[0], hi[0], xf), _fraction(lo[1], hi[1], yf), z])
    return points


def _distance_ok(point: Sequence[float], selected: Iterable[Sequence[float]], minimum: float) -> bool:
    return all(math.dist(point, other) >= minimum for other in selected)


def generate_expert_layout(
    problem: dict[str, Any],
    method: str,
    wap_count: int,
    *,
    random_seed: int,
    minimum_distance_m: float | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Build an expert baseline layout from V17 problem-definition data only."""
    method = {m.upper(): m for m in EXPERT_METHODS}.get(str(method).upper())
    if method is None:
        raise ValueError(f"Unsupported expert method: {method}")
    if wap_count <= 0:
        raise ValueError("wap_count must be positive")
    opt = problem["optimization_problem"]
    bounds = opt["deployable_bounds"]
    objects = problem["objects"]
    constraints = opt.get("constraints", {})
    minimum = float(minimum_distance_m or constraints.get(
        "minimum_wap_distance_m", DEFAULT_MINIMUM_WAP_DISTANCE_M
    ))
    raw = {
        "EbM1": _hed_candidates,
        "EbM2": _aed_candidates,
        "EbM3": _sed_candidates,
    }[method](problem)
    selected: list[list[float]] = []
    rejected_invalid = 0
    rejected_distance = 0
    for point in raw:
        candidate = [round(float(v), 6) for v in point]
        if not is_valid_wap(candidate, bounds, objects):
            rejected_invalid += 1
            continue
        if not _distance_ok(candidate, selected, minimum):
            rejected_distance += 1
            continue
        selected.append(candidate)
        if len(selected) == wap_count:
            break
    rng = random.Random(int(random_seed))
    while len(selected) < wap_count:
        candidate = random_wap_point(
            rng, bounds, None, objects,
            prefer_boundary=(method == "EbM1" or (method == "EbM2" and len(selected) % 2 == 0)),
        )
        if _distance_ok(candidate, selected, minimum):
            selected.append(candidate)
    repaired, events = repair_layout(
        selected, rng, bounds, None, objects, minimum,
        z_margin_m=float(constraints.get("wap_z_margin_m", 0.35)),
    )
    audit = constraint_report(repaired, bounds, objects, minimum)
    metadata = {
        "method": method,
        "method_description": METHOD_DESCRIPTIONS[method],
        "rule_version": "1.0",
        "manufacturer_inspired_not_official_design": True,
        "requested_wap_count": int(wap_count),
        "minimum_wap_distance_m": minimum,
        "candidate_count": len(raw),
        "rejected_invalid_candidate_count": rejected_invalid,
        "rejected_minimum_distance_count": rejected_distance,
        "repair_event_count": len(events),
        "constraint_audit": audit,
        "random_seed": int(random_seed),
    }
    return repaired, metadata
