from __future__ import annotations

import itertools, math, random, time
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, vstack

from .define_problem import (_powers, measurement_plane_geometry, metric_arrays, quality_mask,
                             read_json, success_result, verify_region_hash, write_json)

# RF-feasible WAP-count search is part of candidate generation.
class QualitySearchError(RuntimeError):
    def __init__(self, message: str, trace: list[dict[str, Any]]):
        super().__init__(message)
        self.trace = trace


def _cover_solution(coverage: np.ndarray, distances: np.ndarray, min_distance: float,
                    *, count: int | None, rng: np.random.Generator,
                    time_limit_s: float) -> tuple[list[int] | None, str]:
    """Exact RSRP-only set cover, optionally constrained to exactly N WAPs."""
    n_rx, n_sites = coverage.shape
    pairs = np.argwhere(np.triu(distances < min_distance, 1))
    spacing = np.zeros((len(pairs), n_sites), dtype=float)
    for k, (a, b) in enumerate(pairs):
        spacing[k, a] = spacing[k, b] = 1.0
    blocks = [csr_matrix(coverage.astype(float)), csr_matrix(spacing)]
    lower = [np.ones(n_rx), np.zeros(len(pairs))]
    upper = [np.full(n_rx, np.inf), np.ones(len(pairs))]
    if count is not None:
        blocks.append(csr_matrix(np.ones((1, n_sites))))
        lower.append(np.asarray([count], dtype=float))
        upper.append(np.asarray([count], dtype=float))
    matrix = vstack(blocks)
    objective = np.ones(n_sites) if count is None else rng.uniform(0.8, 1.2, n_sites)
    result = milp(objective, integrality=np.ones(n_sites), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, np.concatenate(lower), np.concatenate(upper)),
                  options={"time_limit": time_limit_s, "mip_rel_gap": 0.0})
    if result.status == 2:
        return None, "infeasible"
    if result.status != 0 or result.x is None:
        raise RuntimeError(f"RSRP covering model was not solved: {result.message}")
    selected = np.flatnonzero(result.x > 0.5).tolist()
    if not bool(np.all(coverage[:, selected].any(axis=1))):
        raise RuntimeError("RSRP covering solution failed independent verification")
    return selected, "optimal"


def _quality_rank(gain: np.ndarray, selected: list[int], profile: dict[str, Any],
                  requirements: dict[str, float]) -> tuple[int, float]:
    metrics = metric_arrays(gain[:, selected], profile)
    good = quality_mask(gain[:, selected], profile, requirements)
    deficit = 0.0
    for field, limit, scale in (("rsrp_dbm", "min_rsrp_dbm", 30.0),
                                ("rsrq_db", "min_rsrq_db", 5.0),
                                ("sinr_db", "min_sinr_db", 15.0)):
        gap = requirements[limit] - metrics[field]
        deficit += float(np.clip(np.nan_to_num(gap, nan=1000.0, posinf=1000.0), 0.0, 1000.0).sum() / scale)
    return int(np.count_nonzero(~good)), deficit


def _improve_layout(gain: np.ndarray, coverage: np.ndarray, distances: np.ndarray,
                    selected: list[int], min_distance: float, profile: dict[str, Any],
                    requirements: dict[str, float]) -> tuple[list[int], int]:
    selected = selected.copy()
    best = _quality_rank(gain, selected, profile, requirements)
    for _ in range(5):
        improved = False
        for slot in range(len(selected)):
            others = selected[:slot] + selected[slot + 1:]
            already = coverage[:, others].any(axis=1) if others else np.zeros(len(gain), dtype=bool)
            for j in range(gain.shape[1]):
                if j in others or (others and np.any(distances[j, others] < min_distance)):
                    continue
                if not bool(np.all(already | coverage[:, j])):
                    continue
                trial = selected.copy(); trial[slot] = j
                rank = _quality_rank(gain, trial, profile, requirements)
                if rank < best:
                    selected, best, improved = trial, rank, True
                    if best[0] == 0:
                        return selected, 0
        if not improved:
            break
    return selected, best[0]


def _enumerate_small_count(gain: np.ndarray, coverage: np.ndarray,
                           distances: np.ndarray, count: int, min_distance: float,
                           profile: dict[str, Any], requirements: dict[str, float]) -> tuple[list[int] | None, int]:
    """Exhaustively certify N<=3 when local search found no feasible layout."""
    if count > 3:
        raise ValueError("Exact enumeration is restricted to at most three WAPs")
    bits = []
    for j in range(gain.shape[1]):
        mask = 0
        for i in np.flatnonzero(coverage[:, j]):
            mask |= 1 << int(i)
        bits.append(mask)
    target = (1 << len(gain)) - 1
    checked = 0
    for combo in itertools.combinations(range(gain.shape[1]), count):
        if any(distances[a, b] < min_distance for a, b in itertools.combinations(combo, 2)):
            continue
        union = 0
        for j in combo:
            union |= bits[j]
        if union != target:
            continue
        checked += 1
        if bool(np.all(quality_mask(gain[:, combo], profile, requirements))):
            return list(combo), checked
    return None, checked


def _enumerate_cover_solutions(gain: np.ndarray, coverage: np.ndarray,
                               distances: np.ndarray, count: int, min_distance: float,
                               profile: dict[str, Any], requirements: dict[str, float],
                               *, max_layouts: int, time_limit_s: float) -> tuple[list[int] | None, int, bool]:
    """Enumerate exact-N RSRP covers with no-good cuts; verify full RF quality."""
    n_rx, n_sites = coverage.shape
    pairs = np.argwhere(np.triu(distances < min_distance, 1))
    spacing = np.zeros((len(pairs), n_sites), dtype=float)
    for k, (a, b) in enumerate(pairs):
        spacing[k, a] = spacing[k, b] = 1.0
    base = vstack([csr_matrix(coverage.astype(float)), csr_matrix(spacing),
                   csr_matrix(np.ones((1, n_sites)))])
    low = np.r_[np.ones(n_rx), np.zeros(len(pairs)), count]
    high = np.r_[np.full(n_rx, np.inf), np.ones(len(pairs)), count]
    cuts: list[np.ndarray] = []
    deadline = time.monotonic() + time_limit_s
    for _ in range(max_layouts):
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            return None, len(cuts), False
        matrix = vstack([base, csr_matrix(np.asarray(cuts))]) if cuts else base
        result = milp(np.ones(n_sites), integrality=np.ones(n_sites), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix,
                          np.r_[low, np.full(len(cuts), -np.inf)],
                          np.r_[high, np.full(len(cuts), count - 1)]),
                      options={"time_limit": min(10.0, remaining), "mip_rel_gap": 0.0})
        if result.status == 2:
            return None, len(cuts), True
        if result.status != 0 or result.x is None:
            return None, len(cuts), False
        selected = np.flatnonzero(result.x > 0.5).tolist()
        if bool(np.all(quality_mask(gain[:, selected], profile, requirements))):
            return selected, len(cuts) + 1, False
        cut = np.zeros(n_sites)
        cut[selected] = 1.0
        cuts.append(cut)
    return None, len(cuts), False


def minimum_quality_layout(gain: np.ndarray, sites: np.ndarray,
                           profile: dict[str, Any], requirements: dict[str, float],
                           min_distance: float, *, max_count: int = 30,
                           restarts_per_count: int = 8,
                           time_limit_s: float = 600.0) -> tuple[list[int], dict[str, Any]]:
    """Check N in order and return the first 600-point RF-feasible layout found.

    The RSRP-only model proves a count lower bound. If full RF feasibility is
    found at that bound, the count is proven minimum on the candidate set.
    Otherwise the returned count is the first verified under the stated
    search budget, with smaller counts explicitly marked unproven.
    """
    gain = np.asarray(gain, dtype=np.float64)
    sites = np.asarray(sites, dtype=np.float64)
    if gain.ndim != 2 or gain.shape[0] != 600 or gain.shape[1] != len(sites):
        raise ValueError("The count search requires 600 receivers and aligned candidate sites")
    if not np.all(np.isfinite(gain)) or np.any(gain < 0):
        raise ValueError("Sionna path gains must be finite and non-negative")
    if max_count < 1 or restarts_per_count < 1:
        raise ValueError("Count and restart limits must be positive")
    _, p_sss, _, _, _ = _powers(profile)
    min_gain = 10.0 ** (requirements["min_rsrp_dbm"] / 10.0) / p_sss
    coverage = gain >= min_gain
    if np.any(~coverage.any(axis=1)):
        raise QualitySearchError("At least one receiver cannot meet RSRP from any candidate site", [])
    distances = np.linalg.norm(sites[:, None, :] - sites[None, :, :], axis=2)
    rng = np.random.default_rng(20261111)
    deadline = time.monotonic() + time_limit_s
    seed, _ = _cover_solution(coverage, distances, min_distance,
                              count=None, rng=rng, time_limit_s=min(90.0, time_limit_s))
    if seed is None:
        raise QualitySearchError("No RSRP-covering layout obeys the spacing constraint", [])
    lower_bound = len(seed)
    trace: list[dict[str, Any]] = [
        {"N": n, "status": "proven_infeasible_by_rsrp_lower_bound"}
        for n in range(1, lower_bound)]
    for count in range(lower_bound, min(max_count, len(sites)) + 1):
        best_failed = 600
        for attempt in range(restarts_per_count):
            remaining = deadline - time.monotonic()
            if remaining <= 1.0:
                raise QualitySearchError("RF-quality count search exhausted its time budget", trace)
            if count == lower_bound and attempt == 0:
                selected = seed.copy()
            else:
                try:
                    selected, status = _cover_solution(coverage, distances, min_distance,
                        count=count, rng=rng, time_limit_s=min(30.0, remaining))
                except RuntimeError as exc:
                    raise QualitySearchError(str(exc), trace) from exc
                if selected is None:
                    trace.append({"N": count, "status": "proven_infeasible_by_rsrp_and_spacing"})
                    break
            selected, failed = _improve_layout(gain, coverage, distances, selected,
                min_distance, profile, requirements)
            best_failed = min(best_failed, failed)
            if failed == 0:
                trace.append({"N": count, "status": "feasible_verified_600_of_600",
                              "attempt": attempt + 1})
                return selected, {"selected_wap_count": count,
                    "rsrp_only_lower_bound": lower_bound,
                    "minimum_count_proven": count == lower_bound or all(
                        row["status"].startswith("proven_infeasible") for row in trace[:-1]),
                    "receiver_count": 600, "candidate_site_count": len(sites),
                    "quality_requirements": requirements, "count_trace": trace,
                    "restarts_per_count": restarts_per_count,
                    "search_time_limit_s": time_limit_s}
        else:
            if count <= 3:
                exact, checked = _enumerate_small_count(gain, coverage, distances,
                    count, min_distance, profile, requirements)
                if exact is not None:
                    trace.append({"N": count, "status": "feasible_verified_600_of_600",
                                  "basis": "exact enumeration", "layouts_checked": checked})
                    return exact, {"selected_wap_count": count,
                        "rsrp_only_lower_bound": lower_bound,
                        "minimum_count_proven": all(
                            row["status"].startswith("proven_infeasible") for row in trace[:-1]),
                        "receiver_count": 600, "candidate_site_count": len(sites),
                        "quality_requirements": requirements, "count_trace": trace,
                        "restarts_per_count": restarts_per_count,
                        "search_time_limit_s": time_limit_s}
                trace.append({"N": count, "status": "proven_infeasible_full_quality",
                              "basis": "exact enumeration", "layouts_checked": checked})
            else:
                remaining = deadline - time.monotonic()
                exact, checked, exhausted = _enumerate_cover_solutions(
                    gain, coverage, distances, count, min_distance,
                    profile, requirements, max_layouts=5000,
                    time_limit_s=max(1.0, min(120.0, remaining)))
                if exact is not None:
                    trace.append({"N": count, "status": "feasible_verified_600_of_600",
                                  "basis": "exact RSRP-cover enumeration", "layouts_checked": checked})
                    return exact, {"selected_wap_count": count,
                        "rsrp_only_lower_bound": lower_bound,
                        "minimum_count_proven": all(
                            row["status"].startswith("proven_infeasible") for row in trace[:-1]),
                        "receiver_count": 600, "candidate_site_count": len(sites),
                        "quality_requirements": requirements, "count_trace": trace,
                        "restarts_per_count": restarts_per_count,
                        "search_time_limit_s": time_limit_s}
                trace.append({"N": count,
                              "status": "proven_infeasible_full_quality" if exhausted
                                        else "not_found_within_search_budget",
                              "basis": "exact RSRP-cover enumeration" if exhausted
                                       else "enumeration budget exhausted",
                              "layouts_checked": checked,
                              "best_failed_receiver_count": best_failed})
    raise QualitySearchError(
        f"No 600-point RF-feasible layout found through N={min(max_count, len(sites))}", trace)

TOOL_NAME = "generate_and_evaluate_candidate_layouts"
SURROGATE_CELL_SIZE_M = 0.75  # radio-map cell size of the surrogate measurement planes

def point_inside_aabb(point: Sequence[float], aabb: dict[str, Sequence[float]], padding: float = .05) -> bool:
    return all(aabb["min"][i] - padding <= point[i] <= aabb["max"][i] + padding for i in range(3))


def is_valid_wap(point: Sequence[float], bounds: dict[str, Sequence[float]], objects: Sequence[dict[str, Any]]) -> bool:
    return all(bounds["min"][i] <= point[i] <= bounds["max"][i] for i in range(3)) and not any(
        point_inside_aabb(point, obj["bounds"]) for obj in objects)


def random_wap_point(rng: random.Random, bounds: dict[str, Sequence[float]], objects: Sequence[dict[str, Any]], z_margin: float) -> list[float]:
    lo, hi = bounds["min"], bounds["max"]
    zpad = min(z_margin, max((hi[2] - lo[2]) * .2, 0.0))
    for _ in range(2000):
        p = [rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]), rng.uniform(lo[2] + zpad, hi[2] - zpad)]
        if is_valid_wap(p, bounds, objects):
            return p
    raise RuntimeError("Could not generate a valid WAP point")


def repair_layout(layout: Sequence[Sequence[float]], rng: random.Random, bounds: dict[str, Sequence[float]],
                  objects: Sequence[dict[str, Any]], minimum_distance: float, z_margin: float = .35) -> list[list[float]]:
    repaired = []
    for original in layout:
        p = [min(max(float(original[a]), bounds["min"][a]), bounds["max"][a]) for a in range(3)]
        if not is_valid_wap(p, bounds, objects) or any(math.dist(p, other) < minimum_distance for other in repaired):
            for _ in range(2000):
                p = random_wap_point(rng, bounds, objects, z_margin)
                if all(math.dist(p, other) >= minimum_distance for other in repaired):
                    break
            else:
                raise RuntimeError("Could not repair WAP layout")
        repaired.append([round(v, 6) for v in p])
    return repaired


def constraint_valid(layout: Sequence[Sequence[float]], bounds: dict[str, Sequence[float]],
                     objects: Sequence[dict[str, Any]], minimum_distance: float) -> bool:
    return all(is_valid_wap(p, bounds, objects) for p in layout) and all(
        math.dist(layout[i], layout[j]) >= minimum_distance
        for i in range(len(layout)) for j in range(i + 1, len(layout)))


# ---------------------------------------------------------------------------
# Ray-Tracing-Assisted Surrogate: Sionna RadioMapSolver numeric matrices.
#
# The surrogate is built ONCE per stage by the Sionna RadioMapSolver, which directly
# outputs path_gain / rss / sinr numeric matrices over one horizontal measurement
# plane per receiver-grid z-level. Transmitters are placed on the 150 candidate-site
# dictionary positions, so any candidate WAP position is served by the nearest
# dictionary column. The GA then reads this ray-tracing data instead of any
# log-distance proxy.
# ---------------------------------------------------------------------------
class RadioMapSurrogate:
    """RadioMapSolver path_gain/rss/sinr matrices queried by the GA as its propagation source."""

    def __init__(self, path_gain: np.ndarray, rss: np.ndarray, sinr: np.ndarray,
                 cell_centers: np.ndarray, z_levels: np.ndarray, tx_positions: np.ndarray,
                 plane: dict[str, Any], metadata: dict[str, Any]):
        self.path_gain = path_gain       # float32 [levels, tx, ny, nx], linear scale
        self.rss = rss                   # optional, not used for GA scoring
        self.sinr = sinr                 # optional, not used for GA scoring
        self.cell_centers = cell_centers # float32 [levels, ny, nx, 3]
        self.z_levels = z_levels         # [levels]
        self.tx_positions = tx_positions # [tx, 3] dictionary grid (the 150 candidate WAP sites)
        self.plane = plane               # {"origin": [x0, y0], "cell_size": [cx, cy], "grid": [ny, nx]}
        self.metadata = metadata
        self._cell_cache: dict[tuple[float, float, float], tuple[int, int, int]] = {}
        self._tx_cache: dict[tuple[float, float, float], int] = {}

    @staticmethod
    def from_files(npz_path: str | Path, json_path: str | Path) -> "RadioMapSurrogate":
        data = np.load(npz_path)
        metadata = read_json(json_path)
        return RadioMapSurrogate(
            data["path_gain"], data["rss"] if "rss" in data else None,
            data["sinr"] if "sinr" in data else None, data["cell_centers"],
            data["z_levels"], data["tx_positions"], metadata["plane"], metadata)

    def tx_index(self, position: Sequence[float]) -> int:
        key = tuple(round(float(v), 6) for v in position)
        cached = self._tx_cache.get(key)
        if cached is not None:
            return cached
        index = int(np.argmin(np.linalg.norm(self.tx_positions - np.asarray(position, dtype=float), axis=1)))
        self._tx_cache[key] = index
        return index

    def cell_index(self, position: Sequence[float]) -> tuple[int, int, int]:
        """Nearest (level, iy, ix) of the measurement grid for a receiver position."""
        key = tuple(round(float(v), 6) for v in position)
        cached = self._cell_cache.get(key)
        if cached is not None:
            return cached
        level = int(np.argmin(np.abs(self.z_levels - float(position[2]))))
        origin, cell_size = self.plane["origin"], self.plane["cell_size"]
        ny, nx = self.plane["grid"]
        ix = min(max(int(math.floor((float(position[0]) - origin[0]) / cell_size[0])), 0), nx - 1)
        iy = min(max(int(math.floor((float(position[1]) - origin[1]) / cell_size[1])), 0), ny - 1)
        result = (level, iy, ix)
        self._cell_cache[key] = result
        return result

    def gain_matrix(self, receiver_positions: Sequence[Sequence[float]],
                    wap_positions: Sequence[Sequence[float]]) -> np.ndarray:
        """Linear path gains [n_rx, n_wap] from the surrogate matrices (nearest-cell lookup)."""
        cells = np.asarray([self.cell_index(receiver) for receiver in receiver_positions], dtype=int)
        tx = np.asarray([self.tx_index(wap) for wap in wap_positions], dtype=int)
        return self.path_gain[cells[:, 0, None], tx[None, :],
                              cells[:, 1, None], cells[:, 2, None]].astype(np.float64)


def _as_numpy(tensor: Any) -> np.ndarray:
    try:
        return np.asarray(tensor.numpy(), dtype=np.float32)
    except (AttributeError, TypeError):
        return np.asarray(tensor, dtype=np.float32)


def build_radio_map_surrogate(state: dict[str, Any], problem: dict[str, Any],
                              receivers: Sequence[dict[str, Any]], *,
                              plane_receivers: Sequence[dict[str, Any]],
                              max_depth: int, los: bool, specular_reflection: bool,
                              diffraction: bool, refraction: bool,
                              samples_per_tx: int, seed: int = 41) -> RadioMapSurrogate:
    """Run the Sionna RadioMapSolver once per receiver-grid z-level and return the
    stacked path_gain / rss / sinr numeric matrices (Ray-Tracing-Assisted surrogate).

    receivers: the TX dictionary (150 obstacle-aware GA points). plane_receivers: the
    600 scenario receivers whose extent and z-levels define the measurement planes, so the
    surrogate samples exactly the geometry the GA will score."""
    try:
        from sionna.rt import load_scene, PlanarArray, Transmitter
        from sionna.rt import RadioMapSolver
    except ImportError as exc:
        raise RuntimeError("Sionna RT RadioMapSolver is required for the GA surrogate") from exc

    opt = problem["optimization_problem"]
    profile = opt["sensor_profile"]
    scene = load_scene(state["paths"]["scene_xml"], merge_shapes=True)
    scene.frequency = float(profile["frequency_ghz"]) * 1e9
    scene.bandwidth = float(profile["bandwidth_mhz"]) * 1e6
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
                                 pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5, horizontal_spacing=.5,
                                 pattern="iso", polarization="V")

    # TX dictionary grid: the 150 obstacle-aware GA receiver positions. A candidate WAP
    # position is served by the nearest dictionary column.
    tx_positions = np.asarray([r["position"] for r in receivers], dtype=float)
    for i, point in enumerate(tx_positions, 1):
        tx = Transmitter(name=f"surr_tx_{i:03d}", position=[float(v) for v in point], orientation=[0, 0, 0])
        tx.power_dbm = float(profile["tx_power_dbm"])
        scene.add(tx)

    # Measurement planes on the SAME geometry as the final export (receiver extent and
    # z-levels); any deviation here shows up as surrogate-vs-audit disagreement.
    geometry = measurement_plane_geometry(plane_receivers)
    center = geometry["center"]
    size = geometry["size"]
    z_levels = geometry["z_levels"]

    solver = RadioMapSolver()
    path_gain, rss, sinr, cell_centers, grids = [], [], [], [], []
    for level in z_levels:
        radio_map = solver(
            scene, center=[center[0], center[1], level], orientation=[0, 0, 0],
            size=size, cell_size=[SURROGATE_CELL_SIZE_M, SURROGATE_CELL_SIZE_M],
            samples_per_tx=int(samples_per_tx), max_depth=max_depth, los=los,
            specular_reflection=specular_reflection, diffuse_reflection=False,
            refraction=refraction, diffraction=diffraction, edge_diffraction=diffraction,
            seed=int(seed))
        path_gain.append(_as_numpy(radio_map.path_gain))    # [tx, ny, nx]
        rss.append(_as_numpy(radio_map.rss))                # [tx, ny, nx], W
        sinr.append(_as_numpy(radio_map.sinr))              # [tx, ny, nx], linear
        cell_centers.append(_as_numpy(radio_map.cell_centers))  # [ny, nx, 3]
        grids.append([int(cell_centers[-1].shape[0]), int(cell_centers[-1].shape[1])])

    if not grids or len({tuple(g) for g in grids}) != 1:
        raise RuntimeError("RadioMapSolver planes must share one cell grid")
    path_gain = np.asarray(path_gain, dtype=np.float32)
    rss = np.asarray(rss, dtype=np.float32)
    sinr = np.asarray(sinr, dtype=np.float32)
    cell_centers = np.asarray(cell_centers, dtype=np.float32)
    if not float(path_gain.sum()) > 0.0:
        raise RuntimeError("RadioMapSolver returned an all-zero coverage map; check scene, materials, and sample count")

    plane = {
        "origin": [float(center[0] - size[0] / 2), float(center[1] - size[1] / 2)],
        "cell_size": [SURROGATE_CELL_SIZE_M, SURROGATE_CELL_SIZE_M],
        "grid": grids[0],
    }
    metadata = {"solver": {
        "name": "Sionna RT RadioMapSolver", "role": "Ray-Tracing-Assisted Surrogate for the GA",
        "max_depth": max_depth, "los": los, "specular_reflection": specular_reflection,
        "diffraction": diffraction, "refraction": refraction, "samples_per_tx": samples_per_tx,
        "seed": int(seed), "cell_size_m": SURROGATE_CELL_SIZE_M,
        "tx_dictionary": "150 legal candidate WAP sites",
        "plane_receivers": "600 valid scenario receivers",
    }, "plane": plane}
    return RadioMapSurrogate(path_gain, rss, sinr, cell_centers,
                             np.asarray(z_levels, dtype=float), tx_positions, plane, metadata)


def evaluate_fitness(receivers: Sequence[dict[str, Any]], wap_positions: Sequence[Sequence[float]],
                     surrogate: RadioMapSurrogate, opt: dict[str, Any]) -> float:
    """Weighted receiver mean of the equal-weight composite scores (数据.txt eq 13-15).

    Propagation comes exclusively from the RadioMapSolver surrogate matrices; the
    SS-RSRP/SS-RSRQ/SS-SINR formulas live in define_problem.ss_metrics_from_gains.
    Receivers without a valid link score 0 and stay in the denominator."""
    gains = surrogate.gain_matrix([r["position"] for r in receivers], wap_positions)
    metrics = metric_arrays(gains, opt["sensor_profile"])
    normalized = np.zeros(len(receivers), dtype=float)
    for values, key in ((metrics["rsrp_dbm"], "rsrp"),
                        (metrics["rsrq_db"], "rsrq"), (metrics["sinr_db"], "sinr")):
        low, high = opt["normalization"][key]
        normalized += np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    weights = np.asarray([r["weight"] for r in receivers], dtype=float)
    weighted_total = float(np.dot(weights, normalized / 3.0))
    weight_sum = float(weights.sum())
    failed = int(np.count_nonzero(~quality_mask(gains, opt["sensor_profile"],
                                                 opt["quality_requirements"])))
    raw_mean = weighted_total / weight_sum
    # Lexicographic feasibility ranking: every fully feasible layout outranks
    # an infeasible one; among infeasible layouts, fewer failed receivers wins.
    # The published composite remains the unpenalized weighted receiver mean.
    return raw_mean if failed == 0 else -float(failed) + raw_mean / (len(receivers) + 1)


def snap_to_dictionary(layout: Sequence[Sequence[float]], tx_positions: np.ndarray) -> list[list[float]]:
    """Snap every WAP to its nearest UNUSED dictionary TX, nearest-first.

    The surrogate only contains ray-traced data for the 150 dictionary TX; evaluating
    a WAP 2-4 m away from the nearest dictionary column silently substitutes that
    column's coverage, which the GA can exploit (a different WAP position and produces incorrect data — this cost the enclosure stage ~0.13 composite). Snapping
    makes the GA search the dictionary sites themselves, so the surrogate fitness is
    exactly what the final export will use. Each dictionary site is used at most
    once (two WAPs cannot occupy one point)."""
    order = sorted(range(len(layout)),
                   key=lambda i: float(np.min(np.linalg.norm(tx_positions - np.asarray(layout[i], dtype=float), axis=1))))
    chosen: list[int | None] = [None] * len(layout)
    used: set[int] = set()
    for i in order:
        dists = np.linalg.norm(tx_positions - np.asarray(layout[i], dtype=float), axis=1)
        for j in np.argsort(dists):
            j = int(j)
            if j not in used:
                used.add(j); chosen[i] = j; break
    return [list(map(float, tx_positions[j])) for j in chosen]


def _evaluate(layout: Sequence[Sequence[float]], receivers: list[dict[str, Any]],
              problem: dict[str, Any], surrogate: RadioMapSurrogate) -> float:
    opt, objects = problem["optimization_problem"], problem["objects"]
    if not constraint_valid(layout, opt["deployable_bounds"], objects, opt["constraints"]["minimum_wap_distance_m"]):
        raise ValueError("Candidate violates hard constraints")
    return evaluate_fitness(receivers, layout, surrogate, opt)


def _additional_feasible_seeds(gain: np.ndarray, sites: np.ndarray,
                               opt: dict[str, Any], selected: list[int],
                               *, time_budget_s: float = 45.0,
                               max_attempts: int = 48,
                               max_seeds: int = 12) -> tuple[list[list[int]], int]:
    """Diversify the GA with independently found, RF-feasible exact-N covers.

    The original population had just one feasible member out of 80 in each
    stage. Re-solving the fixed-N RSRP cover with different objectives and
    repairing full RF quality explores other feasible basins without changing
    the WAP count, the 600-point requirements, or the reported objective F.
    """
    profile, requirements = opt["sensor_profile"], opt["quality_requirements"]
    distance = float(opt["constraints"]["minimum_wap_distance_m"])
    _, p_sss, _, _, _ = _powers(profile)
    min_gain = 10.0 ** (float(requirements["min_rsrp_dbm"]) / 10.0) / p_sss
    coverage = gain >= min_gain
    site_distances = np.linalg.norm(sites[:, None, :] - sites[None, :, :], axis=2)
    rng = np.random.default_rng(int(opt["ga"]["random_seed"]) + 8103)
    seen = {tuple(sorted(selected))}
    feasible: list[list[int]] = []
    attempts = 0
    deadline = time.monotonic() + time_budget_s
    while attempts < max_attempts and len(feasible) < max_seeds:
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            break
        attempts += 1
        try:
            indices, _ = _cover_solution(coverage, site_distances, distance,
                                          count=len(selected), rng=rng,
                                          time_limit_s=min(5.0, remaining))
        except RuntimeError:
            continue
        if indices is None:
            break
        indices, failed = _improve_layout(gain, coverage, site_distances, indices,
                                          distance, profile, requirements)
        key = tuple(sorted(indices))
        if failed == 0 and key not in seen:
            seen.add(key)
            feasible.append(list(key))
    return feasible, attempts


def generate_and_evaluate_candidate_layouts(state_file: str) -> dict[str, Any]:
    """Build GA candidates from the previously generated Sionna surrogate."""
    state_path = Path(state_file).expanduser().resolve()
    state = read_json(state_path)
    root, paths = Path(state["project_root"]), state["paths"]["wap_optimization"]
    if state["optimization_status"].get("surrogate") != "completed":
        raise RuntimeError("Build the Sionna surrogate before generating GA candidates")
    problem = read_json(paths["problem"])
    verify_region_hash(problem)
    opt = problem["optimization_problem"]
    rng = random.Random(opt["ga"]["random_seed"])
    fitness_receivers = read_json(paths["receivers"])["receivers"]
    if len(fitness_receivers) != 600:
        raise RuntimeError("Exactly 600 scenario receivers are required")
    metadata_path = Path(paths["radio_map"]).with_suffix(".meta.json")
    surrogate = RadioMapSurrogate.from_files(paths["radio_map"], metadata_path)

    gain = surrogate.gain_matrix([r["position"] for r in fitness_receivers], surrogate.tx_positions)
    try:
        selected, count_search = minimum_quality_layout(
            gain, surrogate.tx_positions, opt["sensor_profile"], opt["quality_requirements"],
            opt["constraints"]["minimum_wap_distance_m"])
    except RuntimeError as exc:
        write_json(paths["count_search"], {"status": "failed", "reason": str(exc),
            "receiver_count": len(fitness_receivers),
            "candidate_site_count": len(surrogate.tx_positions),
            "quality_requirements": opt["quality_requirements"],
            "region_sha256": problem["scene_regions_sha256"]})
        raise
    seed = [list(map(float, surrogate.tx_positions[j])) for j in selected]
    opt["selected_wap_count"] = len(seed)
    write_json(paths["problem"], problem)
    write_json(paths["count_search"], {**count_search,
        "region_sha256": problem["scene_regions_sha256"],
        "proof_scope": "the selected 600 receiver positions and 150 candidate WAP sites",
        "selected_candidate_indices": selected,
        "selected_positions": seed})

    if not constraint_valid(seed, opt["deployable_bounds"], problem["objects"],
                            opt["constraints"]["minimum_wap_distance_m"]):
        raise RuntimeError("RF-quality seed violates deployment constraints")
    population = [{"wap_positions": seed,
                   "fitness": _evaluate(seed, fitness_receivers, problem, surrogate),
                   "source": "count_search_seed"}]
    additional, feasible_seed_attempts = _additional_feasible_seeds(
        gain, surrogate.tx_positions, opt, selected,
        max_seeds=min(12, opt["ga"]["population_size"] - 1))
    for indices in additional:
        layout = [list(map(float, surrogate.tx_positions[j])) for j in indices]
        population.append({"wap_positions": layout,
                           "fitness": _evaluate(layout, fitness_receivers, problem, surrogate),
                           "source": "independent_feasible_cover"})
    feasible_seed_count = len(population)
    attempts = 0
    while len(population) < opt["ga"]["population_size"]:
        attempts += 1
        if attempts > opt["ga"]["population_size"] * 200:
            raise RuntimeError("Could not generate enough feasible candidate layouts")
        raw = [
            random_wap_point(rng, opt["deployable_bounds"], problem["objects"], opt["constraints"]["wap_z_margin_m"])
            for _ in range(opt["selected_wap_count"])
        ]
        layout = repair_layout(
            raw, rng, opt["deployable_bounds"], problem["objects"],
            opt["constraints"]["minimum_wap_distance_m"], opt["constraints"]["wap_z_margin_m"]
        )
        # The GA must evaluate the exact site it would deploy (see snap_to_dictionary);
        # a snap that breaks the min-distance constraint makes the layout infeasible.
        layout = snap_to_dictionary(layout, surrogate.tx_positions)
        if not constraint_valid(layout, opt["deployable_bounds"], problem["objects"],
                                opt["constraints"]["minimum_wap_distance_m"]):
            continue
        population.append({"wap_positions": layout, "fitness": _evaluate(layout, fitness_receivers, problem, surrogate),
                           "source": "random_layout"})

    population.sort(key=lambda x: x["fitness"], reverse=True)
    output = write_json(paths["candidates"], {
        "receivers": fitness_receivers,
        "population": population,
        "feasible_seed_count": feasible_seed_count,
        "feasible_seed_attempts": feasible_seed_attempts,
        "radio_map_surrogate": {
            "file": paths["radio_map"], "metadata_file": str(metadata_path),
            "solver": surrogate.metadata["solver"],
            "shape": [int(v) for v in surrogate.path_gain.shape],
        },
    })
    state["optimization_status"].update({"candidates": "completed", "optimization": "pending"})
    write_json(state_path, state)
    return success_result(root, TOOL_NAME, output, {
        "state_file": str(state_path),
        "sensor_profile": opt["sensor_profile"]["key"],
        "priority_scenario": opt["priority_scenario"],
        "selected_wap_count": len(seed),
        "feasible_seed_count": feasible_seed_count,
        "count_search_file": paths["count_search"],
        "radio_map_shape": [int(v) for v in surrogate.path_gain.shape],
    })
