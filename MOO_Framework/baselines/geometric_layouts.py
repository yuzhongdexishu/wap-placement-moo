"""GcM1 / GcM2 geometric coverage genetic-algorithm baselines (V17).

GcM = Geometric Coverage Model. GcM1 represents each WAP service region as a
regular hexagon (hexagonal network), GcM2 as a circle (circular network).
Adapted and simplified from a_V16_EXPERIMENTS_REFACTORED_V2
experiments/geometric/{ga_optimizer, geometry_models}.py and
experiments/core/suite.py (GeometryGASettings). Changes:

- the initial positions are seeded random placements generated inside this
  module, Z cycling through the height layers — the V17 PM (MOO) layout is
  never used as an anchor, so the result is this method's own optimization;
- the weighted area grid marks only the SC1 priority-scenario regions with
  weight 2 and the rest of the space with weight 1 (the SC1 baseline: F-SC1
  zones for frame, E-SC1 zones for enclosure), matching
  tools/wap_layout_optimization/generate_candidates._weight;
- the V17 10-field sensor profile has no RSRP threshold field, so the
  link-budget coverage threshold is the module constant SP2_RSRP_THRESHOLD_DBM
  (the V16 SP2 row's rsrp_threshold_dbm).
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .geometry import constraint_report

METHOD_DESCRIPTIONS = {
    "GcM1": (
        "Each WAP service region is represented by a regular hexagon. The GA maximizes the union of "
        "weighted geometric coverage areas over five horizontal height layers."
    ),
    "GcM2": (
        "Each WAP service region is represented by a circle. The GA maximizes the union of weighted "
        "geometric coverage areas over five horizontal height layers."
    ),
}

GEOMETRIC_METHODS = ("GcM1", "GcM2")


def _canonical_method(method: str) -> str:
    """Map a case-insensitive method name to its canonical GcM key."""
    return {m.upper(): m for m in GEOMETRIC_METHODS}.get(str(method).upper(), "")


# RSRP coverage threshold for the equivalent service radius. The V17 10-field
# SensorProfile has no threshold field; this is the V16 SP2 row's value.
SP2_RSRP_THRESHOLD_DBM = -94.0


@dataclass(frozen=True)
class GeometryGASettings:
    population_size: int = 80
    generations: int = 35
    crossover_probability: float = 0.75
    mutation_probability: float = 0.30
    tournament_size: int = 4
    elite_count: int = 2
    minimum_wap_distance_m: float = 4.0
    grid_shape_xy: tuple[int, int] = (72, 48)
    height_layer_count: int = 5
    random_seed: int = 20260718
    radius_override_m: float | None = None


def equivalent_service_radius_m(
    profile: Any,
    propagation_loss_db: float,
    *,
    radius_override_m: float | None = None,
) -> float:
    """Derive the same-height service radius from the proxy RSRP link budget.

    The profile transmit power is interpreted as total-band power. RSRP is the
    per-reference-resource-element power, consistent with the main project.
    """
    if radius_override_m is not None:
        if radius_override_m <= 0:
            raise ValueError("radius_override_m must be positive")
        return float(radius_override_m)
    reference_re_count = profile.num_resource_blocks * profile.subcarriers_per_rb
    required_total_rx_dbm = SP2_RSRP_THRESHOLD_DBM + 10.0 * math.log10(reference_re_count)
    allowable_path_loss_db = (
        profile.tx_power_dbm + profile.tx_gain_dbi + profile.rx_gain_dbi
        - required_total_rx_dbm
    )
    numerator = (
        allowable_path_loss_db
        - float(propagation_loss_db)
        - 32.4
        - 20.0 * math.log10(profile.frequency_ghz)
    )
    radius = 10.0 ** (numerator / 21.0)
    return max(float(radius), 0.5)


def equal_area_hexagon_circumradius(circle_radius_m: float) -> float:
    """Return hexagon circumradius whose area equals pi*r^2."""
    return float(circle_radius_m) * math.sqrt(2.0 * math.pi / (3.0 * math.sqrt(3.0)))


def point_mask(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    circle_radius_m: float,
) -> np.ndarray:
    dx = np.abs(x - float(center_x))
    dy = np.abs(y - float(center_y))
    if method == "GcM2":
        return dx * dx + dy * dy <= float(circle_radius_m) ** 2
    if method == "GcM1":
        side = equal_area_hexagon_circumradius(circle_radius_m)
        root3 = math.sqrt(3.0)
        return (
            (dx <= side)
            & (dy <= root3 * side / 2.0)
            & (root3 * dx + dy <= root3 * side)
        )
    raise ValueError(f"Unknown geometry method: {method}")


def build_height_layers(bounds: dict[str, Sequence[float]], count: int) -> list[float]:
    if count <= 0:
        raise ValueError("height layer count must be positive")
    z0, z1 = float(bounds["min"][2]), float(bounds["max"][2])
    if count == 1:
        return [(z0 + z1) / 2.0]
    step = (z1 - z0) / count
    return [z0 + (i + 0.5) * step for i in range(count)]


def nearest_layer_index(z: float, layers: Sequence[float]) -> int:
    return min(range(len(layers)), key=lambda idx: abs(float(z) - float(layers[idx])))


def build_weighted_area_grid(
    bounds: dict[str, Sequence[float]],
    zones: Sequence[dict[str, Any]],
    priority_weight: float,
    regular_weight: float,
    grid_shape_xy: Sequence[int],
    priority_scenario: str = "SC1",
    stage: str | None = None,
) -> dict[str, Any]:
    """Grid cell weights: only the priority scenario's rects (SC1 baseline) get
    priority_weight; every other zone and the open space gets regular_weight."""
    nx, ny = int(grid_shape_xy[0]), int(grid_shape_xy[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("grid_shape_xy must contain positive integers")
    x0, y0 = float(bounds["min"][0]), float(bounds["min"][1])
    x1, y1 = float(bounds["max"][0]), float(bounds["max"][1])
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    xs = x0 + (np.arange(nx, dtype=float) + 0.5) * dx
    ys = y0 + (np.arange(ny, dtype=float) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys)
    weights = np.full(xx.shape, float(regular_weight), dtype=float)
    if stage is not None:
        import shapely
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        from tools.wap_layout_optimization.define_problem import region_data, scenario_key
        floors = region_data()["scenarios"][scenario_key(stage, priority_scenario)]["floors"]
        priority = unary_union([Polygon(p["outer"], p.get("holes", []))
                                for floor in floors.values() for p in floor["polygons"]])
        weights[shapely.contains_xy(priority, xx, yy)] = float(priority_weight)
        return {"x": xx, "y": yy, "weights": weights, "cell_area_m2": dx * dy,
                "grid_shape_xy": [nx, ny], "grid_spacing_m": [dx, dy],
                "plan_area_m2": (x1 - x0) * (y1 - y0)}
    for zone in zones:
        if str(zone.get("scenario", "")) != priority_scenario:
            continue
        mask = (
            (xx >= float(zone["x_min"]))
            & (xx <= float(zone["x_max"]))
            & (yy >= float(zone["y_min"]))
            & (yy <= float(zone["y_max"]))
        )
        weights[mask] = float(priority_weight)
    return {
        "x": xx,
        "y": yy,
        "weights": weights,
        "cell_area_m2": dx * dy,
        "grid_shape_xy": [nx, ny],
        "grid_spacing_m": [dx, dy],
        "plan_area_m2": (x1 - x0) * (y1 - y0),
    }


def _z_obstacles(
    z: float,
    objects: Sequence[dict[str, Any]],
    padding: float = 0.05,
) -> list[tuple[float, float, float, float]]:
    rows: list[tuple[float, float, float, float]] = []
    for obj in objects:
        bounds = obj["bounds"]
        if float(bounds["min"][2]) - padding <= z <= float(bounds["max"][2]) + padding:
            rows.append(
                (
                    float(bounds["min"][0]) - padding,
                    float(bounds["max"][0]) + padding,
                    float(bounds["min"][1]) - padding,
                    float(bounds["max"][1]) + padding,
                )
            )
    return rows


def _valid_xy(
    x: float,
    y: float,
    bounds: dict[str, Sequence[float]],
    obstacles: Sequence[tuple[float, float, float, float]],
) -> bool:
    if not float(bounds["min"][0]) <= x <= float(bounds["max"][0]):
        return False
    if not float(bounds["min"][1]) <= y <= float(bounds["max"][1]):
        return False
    return not any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in obstacles)


def _initial_positions(
    rng: random.Random,
    bounds: dict[str, Sequence[float]],
    objects: Sequence[dict[str, Any]],
    layers: Sequence[float],
    wap_count: int,
    max_attempts: int = 2000,
) -> list[list[float]]:
    """Seed `wap_count` legal random initial WAP positions.

    Z cycles through the height layers so every layer starts with at least one
    WAP. Independent of the PM layout: the geometric group optimizes from its
    own seeded random initial placement.
    """
    positions: list[list[float]] = []
    for index in range(wap_count):
        z = float(layers[index % len(layers)])
        obstacles = _z_obstacles(z, objects)
        point: list[float] | None = None
        for _ in range(max_attempts):
            x = rng.uniform(float(bounds["min"][0]), float(bounds["max"][0]))
            y = rng.uniform(float(bounds["min"][1]), float(bounds["max"][1]))
            if _valid_xy(x, y, bounds, obstacles):
                point = [float(x), float(y), z]
                break
        if point is None:
            raise RuntimeError(f"No legal random initial position found at z={z:.3f} m")
        positions.append([round(v, 6) for v in point])
    return positions


def _candidate_pool(
    anchor: Sequence[float],
    rng: random.Random,
    bounds: dict[str, Sequence[float]],
    obstacles: Sequence[tuple[float, float, float, float]],
    size: int = 240,
) -> list[list[float]]:
    """Build a legal candidate-site pool for one WAP at its fixed Z value."""
    z = float(anchor[2])
    x0, y0 = float(bounds["min"][0]), float(bounds["min"][1])
    x1, y1 = float(bounds["max"][0]), float(bounds["max"][1])
    pool: list[list[float]] = []
    seen: set[tuple[float, float]] = set()

    def add(x: float, y: float) -> None:
        key = (round(float(x), 5), round(float(y), 5))
        if key in seen or not _valid_xy(float(x), float(y), bounds, obstacles):
            return
        seen.add(key)
        pool.append([float(x), float(y), z])

    # The seeded random initial position is a useful feasible anchor.
    add(float(anchor[0]), float(anchor[1]))

    # Scene-wide deterministic grid provides reproducible spatial coverage.
    nx, ny = 36, 24
    for iy in range(ny):
        y = y0 + (iy + 0.5) * (y1 - y0) / ny
        for ix in range(nx):
            x = x0 + (ix + 0.5) * (x1 - x0) / nx
            add(x, y)

    # Supplement the pool near the initial anchor and throughout the scene.
    span_x, span_y = x1 - x0, y1 - y0
    for _ in range(1200):
        if len(pool) >= max(size * 2, size + 1):
            break
        if rng.random() < 0.55:
            x = float(anchor[0]) + rng.gauss(0.0, span_x * 0.18)
            y = float(anchor[1]) + rng.gauss(0.0, span_y * 0.18)
            x = min(max(x, x0), x1)
            y = min(max(y, y0), y1)
        else:
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        add(x, y)

    if not pool:
        raise RuntimeError(f"No geometrically deployable candidate was found at z={z:.3f} m")

    # Keep the anchor and shuffle the remaining legal candidates.
    anchor_point = pool[0]
    remainder = pool[1:]
    rng.shuffle(remainder)
    return [anchor_point, *remainder[: max(0, size - 1)]]


def _mask_to_int(mask: np.ndarray) -> int:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8).reshape(-1), bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)


def _build_mask_cache(
    method: str,
    pools: Sequence[Sequence[Sequence[float]]],
    layers: Sequence[float],
    area_grid: dict[str, Any],
    radius_m: float,
) -> tuple[list[list[int]], list[int], dict[float, int]]:
    """Precompute geometric coverage masks for every legal candidate site."""
    xx = np.asarray(area_grid["x"], dtype=float)
    yy = np.asarray(area_grid["y"], dtype=float)
    weights = np.asarray(area_grid["weights"], dtype=float).reshape(-1)
    masks_by_wap: list[list[int]] = []
    layer_by_wap: list[int] = []

    for pool in pools:
        layer_by_wap.append(nearest_layer_index(float(pool[0][2]), layers))
        masks_by_wap.append(
            [
                _mask_to_int(point_mask(method, xx, yy, point[0], point[1], radius_m))
                for point in pool
            ]
        )

    weight_masks: dict[float, int] = {}
    for value in np.unique(weights):
        weight_masks[float(value)] = _mask_to_int(np.isclose(weights, value))
    return masks_by_wap, layer_by_wap, weight_masks


def _repair_genes(
    genes: list[int],
    pools: Sequence[Sequence[Sequence[float]]],
    rng: random.Random,
    minimum_distance_m: float,
) -> tuple[list[int], int]:
    repaired: list[int] = []
    selected_positions: list[list[float]] = []
    repair_count = 0

    for wap_index, raw_gene in enumerate(genes):
        pool = pools[wap_index]
        gene = int(raw_gene) % len(pool)
        candidate = [float(v) for v in pool[gene]]
        valid = all(math.dist(candidate, other) >= minimum_distance_m for other in selected_positions)
        if not valid:
            repair_count += 1
            order = list(range(len(pool)))
            rng.shuffle(order)
            chosen_gene: int | None = None
            for candidate_gene in order:
                point = [float(v) for v in pool[candidate_gene]]
                if all(math.dist(point, other) >= minimum_distance_m for other in selected_positions):
                    chosen_gene = candidate_gene
                    break
            if chosen_gene is None:
                chosen_gene = max(
                    range(len(pool)),
                    key=lambda idx: min(
                        (math.dist(pool[idx], other) for other in selected_positions),
                        default=math.inf,
                    ),
                )
            gene = int(chosen_gene)
            candidate = [float(v) for v in pool[gene]]
        repaired.append(gene)
        selected_positions.append(candidate)
    return repaired, repair_count


def _genes_to_positions(
    genes: Sequence[int], pools: Sequence[Sequence[Sequence[float]]]
) -> list[list[float]]:
    return [[float(v) for v in pools[index][int(gene)]] for index, gene in enumerate(genes)]


def _evaluate_genes(
    method: str,
    genes: list[int],
    masks_by_wap: Sequence[Sequence[int]],
    layer_by_wap: Sequence[int],
    layers: Sequence[float],
    area_grid: dict[str, Any],
    radius_m: float,
    weight_masks: dict[float, int],
) -> dict[str, Any]:
    union_by_layer = [0 for _ in layers]
    wap_count_by_layer = [0 for _ in layers]
    for wap_index, gene in enumerate(genes):
        layer_index = int(layer_by_wap[wap_index])
        union_by_layer[layer_index] |= int(masks_by_wap[wap_index][int(gene)])
        wap_count_by_layer[layer_index] += 1

    cell_area = float(area_grid["cell_area_m2"])
    cells_per_layer = int(np.asarray(area_grid["weights"]).size)
    one_layer_area = cells_per_layer * cell_area
    total_area = one_layer_area * len(layers)
    one_layer_weighted = sum(
        float(weight) * int(mask).bit_count() for weight, mask in weight_masks.items()
    ) * cell_area
    total_weighted = one_layer_weighted * len(layers)

    per_layer: list[dict[str, Any]] = []
    covered_area = 0.0
    weighted_covered = 0.0
    for layer_index, (layer_z, union) in enumerate(zip(layers, union_by_layer)):
        layer_covered = int(union).bit_count() * cell_area
        layer_weighted = sum(
            float(weight) * (int(union) & int(mask)).bit_count()
            for weight, mask in weight_masks.items()
        ) * cell_area
        covered_area += layer_covered
        weighted_covered += layer_weighted
        per_layer.append(
            {
                "layer_index": layer_index + 1,
                "z_m": float(layer_z),
                "wap_count": int(wap_count_by_layer[layer_index]),
                "covered_area_m2": float(layer_covered),
                "weighted_covered_area": float(layer_weighted),
                "coverage_rate": float(layer_covered / max(one_layer_area, 1e-12)),
            }
        )

    geometry = {
        "method": method,
        "circle_equivalent_radius_m": float(radius_m),
        "hexagon_circumradius_m": equal_area_hexagon_circumradius(radius_m),
        "covered_area_m2": float(covered_area),
        "total_multilayer_area_m2": float(total_area),
        "geometric_coverage_rate": float(covered_area / max(total_area, 1e-12)),
        "weighted_covered_area": float(weighted_covered),
        "total_weighted_area": float(total_weighted),
        "weighted_geometric_coverage_rate": float(weighted_covered / max(total_weighted, 1e-12)),
        "per_layer": per_layer,
    }
    return {
        "genes": genes,
        "fitness": geometry["weighted_geometric_coverage_rate"],
        "geometry": geometry,
    }


def _tournament(
    population: list[dict[str, Any]], rng: random.Random, size: int
) -> dict[str, Any]:
    competitors = rng.sample(population, min(size, len(population)))
    return max(competitors, key=lambda row: float(row["fitness"]))


def optimize_geometry_layout(
    method: str,
    wap_count: int,
    bounds: dict[str, Sequence[float]],
    objects: Sequence[dict[str, Any]],
    layers: Sequence[float],
    area_grid: dict[str, Any],
    radius_m: float,
    settings: GeometryGASettings,
) -> dict[str, Any]:
    """Optimize GcM1/GcM2 using a discrete legal-candidate genetic algorithm.

    Every gene selects one legal XY candidate for a WAP while preserving the
    initial Z coordinate. Coverage masks are precomputed as integer bitsets,
    so the 80-population, 35-generation formal configuration can be rebuilt
    deterministically without changing the geometric objective.
    """
    method = _canonical_method(method)
    if not method:
        raise ValueError(f"Unsupported numerical method: {method}")
    if settings.population_size < 4 or settings.generations < 1:
        raise ValueError("population_size must be >= 4 and generations must be >= 1")
    if wap_count <= 0:
        raise ValueError("wap_count must be positive")

    rng = random.Random(settings.random_seed + (1000 if method == "GcM1" else 2000))
    anchor_positions = _initial_positions(rng, bounds, objects, layers, wap_count)
    obstacles_by_index = [_z_obstacles(float(point[2]), objects) for point in anchor_positions]
    pools = [
        _candidate_pool(point, rng, bounds, obstacles_by_index[index])
        for index, point in enumerate(anchor_positions)
    ]
    masks_by_wap, layer_by_wap, weight_masks = _build_mask_cache(
        method, pools, layers, area_grid, radius_m
    )

    population: list[dict[str, Any]] = []
    repair_total = 0
    for _ in range(settings.population_size):
        genes = [rng.randrange(len(pool)) for pool in pools]
        genes, repairs = _repair_genes(
            genes, pools, rng, settings.minimum_wap_distance_m
        )
        repair_total += repairs
        population.append(
            _evaluate_genes(
                method,
                genes,
                masks_by_wap,
                layer_by_wap,
                layers,
                area_grid,
                radius_m,
                weight_masks,
            )
        )
    population.sort(key=lambda row: float(row["fitness"]), reverse=True)

    history: list[dict[str, Any]] = []
    for generation in range(settings.generations + 1):
        fitnesses = [float(row["fitness"]) for row in population]
        history.append(
            {
                "generation": generation,
                "best_fitness": max(fitnesses),
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "minimum_fitness": min(fitnesses),
            }
        )
        if generation == settings.generations:
            break

        next_population = [
            deepcopy(row) for row in population[: max(1, settings.elite_count)]
        ]
        while len(next_population) < settings.population_size:
            parent_a = _tournament(population, rng, settings.tournament_size)
            parent_b = _tournament(population, rng, settings.tournament_size)
            child_a = list(parent_a["genes"])
            child_b = list(parent_b["genes"])

            if rng.random() < settings.crossover_probability:
                for index in range(len(child_a)):
                    if rng.random() < 0.5:
                        child_a[index], child_b[index] = child_b[index], child_a[index]

            for child in (child_a, child_b):
                if rng.random() < settings.mutation_probability:
                    for index, pool in enumerate(pools):
                        if rng.random() < 0.25:
                            child[index] = rng.randrange(len(pool))
                child, repairs = _repair_genes(
                    child, pools, rng, settings.minimum_wap_distance_m
                )
                repair_total += repairs
                next_population.append(
                    _evaluate_genes(
                        method,
                        child,
                        masks_by_wap,
                        layer_by_wap,
                        layers,
                        area_grid,
                        radius_m,
                        weight_masks,
                    )
                )
                if len(next_population) >= settings.population_size:
                    break
        population = sorted(
            next_population, key=lambda row: float(row["fitness"]), reverse=True
        )

    best = population[0]
    best_positions = _genes_to_positions(best["genes"], pools)
    audit = constraint_report(
        best_positions, bounds, objects, settings.minimum_wap_distance_m
    )
    return {
        "method": method,
        "encoding": "discrete_legal_candidate_index",
        "settings": settings.__dict__,
        "best_positions": best_positions,
        "best_candidate_indices": [int(value) for value in best["genes"]],
        "best_fitness": float(best["fitness"]),
        "geometric_performance": best["geometry"],
        "generation_history": history,
        "constraint_audit": audit,
        "total_repair_events": int(repair_total),
        "candidate_pool_size_per_wap": [len(pool) for pool in pools],
        "coverage_mask_cache": {
            "grid_shape_xy": list(settings.grid_shape_xy),
            "height_layer_count": len(layers),
            "representation": "integer_bitset",
        },
    }
