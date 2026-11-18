from __future__ import annotations

import json, math, os, tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence
import hashlib
from functools import lru_cache
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union

# IFC stage regions and receiver domains.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGION_FILE = PROJECT_ROOT / "data/regions/Stage_SC_Regions.json"
PREFIX = {"frame": "F", "enclosure": "E"}


@lru_cache(maxsize=2)
def _region_data_for_version(mtime_ns: int, size: int) -> dict:
    data = json.loads(REGION_FILE.read_text(encoding="utf-8"))
    for stage, digest in data["source_hashes"].items():
        source = PROJECT_ROOT / "data/ifc" / f"{stage}.ifc"
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Stage region source IFC changed: {source}")
    for stage, prefix in PREFIX.items():
        keys = {key for key in data["scenarios"] if key.startswith(prefix + "-SC")}
        expected = {f"{prefix}-SC{i}" for i in range(1, len(keys) + 1)}
        if keys != expected:
            raise RuntimeError(f"{stage} scenario numbers must be contiguous from SC1: {sorted(keys)}")
        if keys != {key for key in data["declared_targets_and_layouts"] if key.startswith(prefix + "-SC")}:
            raise RuntimeError(f"{stage} scenario metadata keys differ from region keys")
        for key in keys:
            if not (PROJECT_ROOT / "data/regions/volumes" / f"{key}.npz").is_file():
                raise FileNotFoundError(f"Missing priority volume for {key}")
    return data


def region_data() -> dict:
    stat = REGION_FILE.stat()
    return _region_data_for_version(stat.st_mtime_ns, stat.st_size)


def scenario_key(stage: str, scenario: str) -> str:
    key = f"{PREFIX[stage]}-{scenario}"
    if key not in region_data()["scenarios"]:
        raise ValueError(f"{key} has no region in the current stage dataset")
    return key


def verify_region_hash(problem: dict) -> None:
    expected = problem.get("scene_regions_sha256")
    actual = hashlib.sha256(REGION_FILE.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError("Stage region file changed during the run; redefine the optimization problem")


def available_scenarios(stage: str) -> tuple[str, ...]:
    prefix = PREFIX[stage] + "-"
    keys = [k[len(prefix):] for k in region_data()["scenarios"] if k.startswith(prefix)]
    return tuple(sorted(keys, key=lambda key: int(key[2:])))


def heuristic_zone_bounds(stage: str, scenario: str) -> list[dict]:
    """Polygon bounds for baseline layout heuristics; never used for RF scoring."""
    key = scenario_key(stage, scenario)
    output = []
    for floor in region_data()["scenarios"][key]["floors"].values():
        shape = _shape(floor["polygons"])
        xmin, ymin, xmax, ymax = shape.bounds
        output.append({"scenario": scenario, "name": region_data()["scenarios"][key]["name"],
                       "x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax,
                       "z_min": floor["z_min_m"], "z_max": floor["z_max_m"],
                       "heuristic_only": True})
    return output


def _shape(polygons: list[dict]):
    return unary_union([Polygon(p["outer"], p.get("holes", [])) for p in polygons])


def generate_receivers(stage: str, scenario: str, count: int = 600,
                       problem: dict | None = None) -> list[dict]:
    """Exactly 200 deterministic points per declared 1F/2F/3F valid domain."""
    if count != 600:
        raise ValueError("The experiment requires exactly 600 receivers")
    key = scenario_key(stage, scenario)
    floors = region_data()["scenarios"][key]["floors"]
    import trimesh
    volume_file = PROJECT_ROOT / "data/regions/volumes" / f"{key}.npz"
    mesh_data = np.load(volume_file)
    volume = trimesh.Trimesh(mesh_data["vertices"], mesh_data["faces"], process=False)
    if not volume.is_watertight:
        raise RuntimeError(f"Priority volume is not watertight: {volume_file}")
    rows = []
    for floor_name in ("1F", "2F", "3F"):
        floor = floors[floor_name]
        domain = _shape(floor["valid_floor_domain"])
        xmin, ymin, xmax, ymax = domain.bounds
        # A regular dense lattice keeps receiver placement independent of RF results.
        nx = ny = 80
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, nx + 2)[1:-1],
                             np.linspace(ymin, ymax, ny + 2)[1:-1])
        x, y = xx.ravel(), yy.ravel()
        mask = shapely.contains_xy(domain, x, y)
        z = (float(floor["z_min_m"]) + float(floor["z_max_m"])) / 2
        if problem is not None:
            opt = problem["optimization_problem"]
            bounds = opt["deployable_bounds"]
            mask &= (x >= bounds["min"][0]) & (x <= bounds["max"][0])
            mask &= (y >= bounds["min"][1]) & (y <= bounds["max"][1])
            mask &= z >= bounds["min"][2]
            mask &= z <= bounds["max"][2]
            for obj in problem["objects"]:
                box = obj["bounds"]
                if box["min"][2] - .05 <= z <= box["max"][2] + .05:
                    mask &= ~((x >= box["min"][0] - .05) & (x <= box["max"][0] + .05)
                              & (y >= box["min"][1] - .05) & (y <= box["max"][1] + .05))
        valid = np.flatnonzero(mask)
        if len(valid) < 200:
            raise RuntimeError(f"Only {len(valid)} valid receiver sites in {key}/{floor_name}")
        if key == "E-SC6":
            # Stair priority is a sloped 3D volume. A plain XY lattice would
            # sample almost none of it at a fixed floor receiver height.
            probes = np.column_stack((x[valid], y[valid], np.full(len(valid), z)))
            in_stair = volume.contains(probes)
            priority_sites, regular_sites = valid[in_stair], valid[~in_stair]
            priority_count = min(50, len(priority_sites))
            if len(regular_sites) < 200 - priority_count:
                raise RuntimeError(f"Insufficient regular receiver sites in {key}/{floor_name}")
            chosen = np.sort(np.r_[
                priority_sites[np.linspace(0, len(priority_sites) - 1, priority_count, dtype=int)],
                regular_sites[np.linspace(0, len(regular_sites) - 1, 200 - priority_count, dtype=int)]])
        else:
            chosen = valid[np.linspace(0, len(valid) - 1, 200, dtype=int)]
        for i, index in enumerate(chosen):
            rows.append({"receiver_id": len(rows) + 1,
                         "position": [float(x[index]), float(y[index]), z],
                         "weight": 1.0,
                         "floor": floor_name})
    # The dataset stores exact 3D clipped volumes. Their membership is the
    # authoritative weight rule, including E-SC6 stair surfaces.
    inside = volume.contains(np.asarray([r["position"] for r in rows]))
    for row, priority in zip(rows, inside):
        row["weight"] = 2.0 if bool(priority) else 1.0
    return rows


def dense_validation_positions(problem: dict, grid_size: int = 80) -> list[dict]:
    """Independent spatial probes; these are not the 600 experiment receivers."""
    opt = problem["optimization_problem"]
    key = scenario_key(opt["construction_stage"], opt["priority_scenario"])
    bounds = opt["deployable_bounds"]
    rows: list[dict] = []
    for floor_name in ("1F", "2F", "3F"):
        floor = region_data()["scenarios"][key]["floors"][floor_name]
        domain = _shape(floor["valid_floor_domain"])
        xmin, ymin, xmax, ymax = domain.bounds
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, grid_size + 2)[1:-1],
                             np.linspace(ymin, ymax, grid_size + 2)[1:-1])
        x, y = xx.ravel(), yy.ravel()
        z = (float(floor["z_min_m"]) + float(floor["z_max_m"])) / 2
        mask = shapely.contains_xy(domain, x, y)
        mask &= (x >= bounds["min"][0]) & (x <= bounds["max"][0])
        mask &= (y >= bounds["min"][1]) & (y <= bounds["max"][1])
        mask &= bounds["min"][2] <= z <= bounds["max"][2]
        for obj in problem["objects"]:
            box = obj["bounds"]
            if box["min"][2] - .05 <= z <= box["max"][2] + .05:
                mask &= ~((x >= box["min"][0] - .05) & (x <= box["max"][0] + .05)
                          & (y >= box["min"][1] - .05) & (y <= box["max"][1] + .05))
        rows.extend({"position": [float(a), float(b), z], "floor": floor_name}
                    for a, b in zip(x[mask], y[mask]))
    return rows


SERVER_NAME = "WAP Placement Optimization MCP Server"
TOOL_NAME = "define_optimization_problem"

# Basic/reliable normalization anchors (数据.txt 五、eq 13; research thresholds from the
# flood-WSN application requirements, NOT 3GPP mandatory limits). Every metric is
# normalized as clip((x - basic) / (reliable - basic), 0, 1). These bounds must not be
# mixed with any other normalization limits.
NORMALIZATION = {"rsrp": [-140.0, -110.0], "rsrq": [-19.5, -15.0], "sinr": [0.0, 15.0]}

# 3GPP TS 38.215 SS-based measurement configuration (数据.txt Table 2).
SSS_RE_COUNT = 127            # SSS subcarrier indexes 56..182
RSSI_RBS = 106                # RSSI is measured over the 106 carrier RBs
RSSI_SYMBOLS = 84             # measurementSlots 1111111000 -> 84 observed symbols
RSSI_SUBCARRIERS = RSSI_RBS * 12  # 1272 15 kHz subcarriers in the RSSI bandwidth
SSB_SCS_KHZ = 30.0            # SSB numerology (SSS REs live on the 30 kHz grid)

@dataclass(frozen=True)
class SensorProfile:
    """Five WAP transmit-power profiles on the same 3.5 GHz carrier."""
    key: str = "SP3"
    frequency_ghz: float = 3.5
    bandwidth_mhz: float = 20.0
    subcarrier_spacing_khz: float = 15.0
    num_resource_blocks: int = 106
    subcarriers_per_rb: int = 12
    tx_power_dbm: float = 23.0
    tx_gain_dbi: float = 0.0
    rx_gain_dbi: float = 0.0
    noise_figure_db: float = 9.0
    noise_floor_dbm: float = -92.0
    channel_count: int = 1
    cochannel_load_factor: float = 1.0


# SP3 is the 23 dBm MOO reference profile. Channel reuse distance has been
# removed: one carrier only, so every other WAP is co-channel interference.
SP3 = SensorProfile()

SENSOR_PROFILES: dict[str, SensorProfile] = {
    "SP1": replace(SP3, key="SP1", tx_power_dbm=13.0),
    "SP2": replace(SP3, key="SP2", tx_power_dbm=18.0),
    "SP3": SP3,
    "SP4": replace(SP3, key="SP4", tx_power_dbm=28.0),
    "SP5": replace(SP3, key="SP5", tx_power_dbm=33.0),
}


def get_sensor_profile(key: str = "SP3") -> SensorProfile:
    if key not in SENSOR_PROFILES:
        raise ValueError(f"Unknown sensor profile {key!r}; expected one of {list(SENSOR_PROFILES)}")
    return SENSOR_PROFILES[key]


def read_json(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        temp = f.name
    os.replace(temp, path)
    return path


def relative(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve())


def success_result(root: Path, tool: str, process_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {"status": "success", "server": SERVER_NAME, "tool": tool,
            "process_file": relative(process_path, root), "summary": summary}


def noise_power_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Thermal noise floor P_N(B) = -174 + 10log10(B) + NF (数据.txt eq 12), B in Hz."""
    return -174.0 + 10.0 * math.log10(float(bandwidth_hz)) + float(noise_figure_db)


# Vectorized RF quality metrics for the 600 receiver constraints.
def _powers(profile: dict[str, Any]) -> tuple[float, float, float, float, int]:
    n_sc = int(profile["num_resource_blocks"]) * int(profile["subcarriers_per_rb"])
    p_tx = 10.0 ** (float(profile["tx_power_dbm"]) / 10.0)
    p_sss = p_tx / n_sc * (30.0 / float(profile["subcarrier_spacing_khz"]))
    nf = float(profile["noise_figure_db"])
    p_n15 = 10.0 ** (noise_power_dbm(15000.0, nf) / 10.0)
    p_n30 = 10.0 ** (noise_power_dbm(30000.0, nf) / 10.0)
    return p_tx, p_sss, p_n15, p_n30, n_sc


def metric_arrays(gains: np.ndarray, profile: dict[str, Any]) -> dict[str, np.ndarray]:
    """Vector form of define_problem.ss_metrics_from_gains for one layout."""
    gains = np.asarray(gains, dtype=np.float64)
    if gains.ndim != 2 or gains.shape[1] == 0:
        raise ValueError("Expected a receiver-by-WAP gain matrix")
    p_tx, p_sss, p_n15, p_n30, n_sc = _powers(profile)
    gmax, gsum = gains.max(axis=1), gains.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rsrp = 10.0 * np.log10(p_sss * gmax)
        rsrq = 10.0 * np.log10(106.0 * p_sss * gmax / (p_tx * gsum + n_sc * p_n15))
        sinr = 10.0 * np.log10(p_sss * gmax / (
            float(profile["cochannel_load_factor"]) * p_sss * (gsum - gmax) + p_n30))
    return {"rsrp_dbm": rsrp, "rsrq_db": rsrq, "sinr_db": sinr}


def quality_mask(gains: np.ndarray, profile: dict[str, Any], requirements: dict[str, float]) -> np.ndarray:
    metrics = metric_arrays(gains, profile)
    return (np.isfinite(metrics["rsrp_dbm"])
            & (metrics["rsrp_dbm"] >= requirements["min_rsrp_dbm"])
            & (metrics["rsrq_db"] >= requirements["min_rsrq_db"])
            & (metrics["sinr_db"] >= requirements["min_sinr_db"]))



def ss_metrics_from_gains(gains: Sequence[float], profile: dict[str, Any] | SensorProfile) -> dict[str, Any]:
    """Research approximation of SS metrics from N wideband WAP path gains.

    Metric names and reference-signal concepts follow TS 38.215; the code does
    not simulate an NR resource grid or perform a standards-equivalent UE
    measurement. Full-buffer overlap and flat-channel assumptions are explicit.

    gains: linear path gains |H_ij|^2 of all WAPs (Sionna RadioMapSolver path_gain matrix
    column/row slice, isotropic 1x1 antennas so no gain is double-applied).

    Flat-channel equivalence: the ray tracer reports the wideband average |H_ij|^2, so
    P^SSS_ij(k) = P_SSS,EPRE,mW * |H_ij|^2 is constant over the 127 SSS REs (k=56..182) and
    eq (4)'s average equals that constant value. Under the full-buffer assumption (co-channel
    load 1.0, uniform PSD) every one of the 1272 x 84 RSSI resources carries the same
    per-RE power, so eq (8) reduces to RSSI = sum_j P_tx*|H_ij|^2 + 1272*P_N,15 after the
    1/84 symbol average and 1/1272 subcarrier sum cancel; the 30 kHz SSB and 15 kHz carrier
    grids are aligned through the SSS EPRE ratio, not through direct RB index mapping.
    All powers are computed in mW before any logarithm, per 数据.txt eq (3).
    """
    if isinstance(profile, SensorProfile):
        profile = asdict(profile)
    tx_power_dbm = float(profile["tx_power_dbm"])
    n_rb = int(profile["num_resource_blocks"])
    n_sc_per_rb = int(profile["subcarriers_per_rb"])
    scs_khz = float(profile["subcarrier_spacing_khz"])
    nf_db = float(profile["noise_figure_db"])
    load = float(profile["cochannel_load_factor"])

    n_sc_total = n_rb * n_sc_per_rb                     # 1272 carrier subcarriers
    p_tx_mw = 10.0 ** (tx_power_dbm / 10.0)
    # SSS EPRE (数据.txt eq 2): P_SSS,EPRE = P_tx - 10log10(106x12) + 10log10(30/15) dBm/RE
    # (=-5.03 dBm/RE for the 23 dBm SP3 reference; adjacent profiles differ by 5 dB).
    p_sss_epre_dbm = tx_power_dbm - 10.0 * math.log10(n_sc_total) + 10.0 * math.log10(SSB_SCS_KHZ / scs_khz)
    p_sss_epre_mw = 10.0 ** (p_sss_epre_dbm / 10.0)
    # Noise powers (数据.txt eq 12): per 15 kHz carrier subcarrier and per 30 kHz SSS RE.
    p_n_15_mw = 10.0 ** (noise_power_dbm(15e3, nf_db) / 10.0)   # -123.24 dBm at NF=9
    p_n_30_mw = 10.0 ** (noise_power_dbm(30e3, nf_db) / 10.0)   # -120.23 dBm at NF=9

    g = [max(float(v), 0.0) for v in gains]
    serving = int(max(range(len(g)), key=lambda j: g[j])) if g else -1
    if serving < 0 or g[serving] <= 0.0:
        # No valid link: uniform zero-score rule (数据.txt 五、), NOT excluded from the mean.
        return {"serving_wap": -1, "rsrp_dbm": None, "rsrq_db": None, "sinr_db": None,
                "no_link": True, "sss_epre_dbm": p_sss_epre_dbm}

    # eq (3)+(4): SS-RSRP is the linear average over the 127 SSS REs of the serving WAP
    # s(i) = argmax_j SS-RSRP_ij (eq 5); constant per RE under the flat-channel assumption.
    ss_rsrp_mw = p_sss_epre_mw * g[serving]
    # eq (8)+(9): RSSI over the 106 RBs x 84 symbols window includes all WAPs (serving
    # included) plus 15 kHz-subcarrier noise on every observed resource.
    rssi_mw = p_tx_mw * sum(g) + n_sc_total * p_n_15_mw
    rsrq_db = 10.0 * math.log10(RSSI_RBS * ss_rsrp_mw / max(rssi_mw, 1e-30))
    # eq (10)+(11): SSS-RE signal vs. the non-coherent sum of the other synchronized
    # WAPs (single carrier, full co-channel overlap) plus one 30 kHz RE of noise.
    p_s = p_sss_epre_mw * g[serving]
    p_i = load * p_sss_epre_mw * sum(g[j] for j in range(len(g)) if j != serving)
    sinr_db = 10.0 * math.log10(p_s / max(p_i + p_n_30_mw, 1e-30))

    return {"serving_wap": serving, "rsrp_dbm": 10.0 * math.log10(ss_rsrp_mw),
            "rsrq_db": rsrq_db, "sinr_db": sinr_db, "no_link": False,
            "ss_rsrp_mw": ss_rsrp_mw, "rssi_mw": rssi_mw,
            "interference_mw": p_i, "noise_30khz_mw": p_n_30_mw,
            "sss_epre_dbm": p_sss_epre_dbm}


def composite_score(metrics: dict[str, Any], normalization: dict[str, Sequence[float]]) -> float:
    """Equal-weight composite F_i (数据.txt eq 13-14): RSRP/RSRQ/SINR each contribute 1/3
    after clip((x - basic)/(reliable - basic), 0, 1) normalization. A receiver without a
    valid link receives the uniform score 0 and stays in the F_total denominator."""
    for key, bounds in (("rsrp_dbm", normalization["rsrp"]),
                        ("rsrq_db", normalization["rsrq"]),
                        ("sinr_db", normalization["sinr"])):
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            return 0.0
    total = 0.0
    for key, bounds in (("rsrp_dbm", normalization["rsrp"]),
                        ("rsrq_db", normalization["rsrq"]),
                        ("sinr_db", normalization["sinr"])):
        lo, hi = map(float, bounds)
        total += max(0.0, min(1.0, (float(metrics[key]) - lo) / (hi - lo)))
    return total / 3.0


def measurement_plane_geometry(receivers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Measurement planes spanning the 600 scenario receivers."""
    lo = [min(r["position"][a] for r in receivers) for a in range(3)]
    hi = [max(r["position"][a] for r in receivers) for a in range(3)]
    center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2]
    size = [hi[0] - lo[0], hi[1] - lo[1]]
    z_levels = sorted({float(r["position"][2]) for r in receivers})
    return {"center": center, "size": size, "z_levels": z_levels}


def _ply_bounds(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        count = 0
        for line in f:
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            if line.strip() == "end_header":
                break
        vertices = [[float(x) for x in f.readline().split()[:3]] for _ in range(count)]
    if not vertices:
        raise ValueError(f"No PLY vertices: {path}")
    return {"min": [min(v[i] for v in vertices) for i in range(3)],
            "max": [max(v[i] for v in vertices) for i in range(3)]}


def _scene_geometry(state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    objects, lo, hi = [], [float("inf")] * 3, [float("-inf")] * 3
    for item in state["objects"]:
        if item.get("mesh_status") != "completed":
            continue
        bounds = _ply_bounds(Path(item["mesh_file"]))
        objects.append({"bounds": bounds})
        lo = [min(lo[i], bounds["min"][i]) for i in range(3)]
        hi = [max(hi[i], bounds["max"][i]) for i in range(3)]
    if not objects:
        raise ValueError("No completed mesh objects in pipeline_state.json")
    return objects, {"min": lo, "max": hi}


def _deployable(bounds: dict[str, Sequence[float]]) -> dict[str, list[float]]:
    lo, hi = list(bounds["min"]), list(bounds["max"])
    if hi[0] - lo[0] <= 1.6 or hi[1] - lo[1] <= 1.6:
        raise ValueError("Scene is too small for deployment margins")
    z = min(.25, max((hi[2] - lo[2]) * .05, 0.0))
    return {"min": [lo[0] + .8, lo[1] + .8, lo[2] + z],
            "max": [hi[0] - .8, hi[1] - .8, hi[2] - z]}


def define_optimization_problem(
    state_file: str,
    receiver_count: int = 600,
    population_size: int = 80,
    generations: int = 35,
    crossover_probability: float = .75,
    mutation_probability: float = .30,
    minimum_wap_distance_m: float = 4.0,
    random_seed: int = 20260717,
    sensor_profile: str = "SP3",
    smart_scenario: str = "SC1",
    min_rsrp_dbm: float = -140.0,
    min_rsrq_db: float = -19.5,
    min_sinr_db: float = 0.0,
) -> dict[str, Any]:
    """Define the stage problem using the current IFC-derived scenario regions."""
    state_path = Path(state_file).expanduser().resolve()
    state = read_json(state_path)
    root = Path(state["project_root"])
    if state.get("scene_status") != "completed" or not Path(state["paths"]["scene_xml"]).is_file():
        raise RuntimeError("Mitsuba scene is not ready")
    if receiver_count != 600:
        raise ValueError("receiver_count must be 600")
    if min(population_size, generations) <= 0:
        raise ValueError("Optimization counts must be positive")
    if not all(math.isfinite(value) for value in (min_rsrp_dbm, min_rsrq_db, min_sinr_db)):
        raise ValueError("RF quality thresholds must be finite")

    profile = get_sensor_profile(sensor_profile)
    construction_stage = state["construction_stage"]
    if construction_stage not in PREFIX:
        raise ValueError(f"Unknown construction_stage {construction_stage!r}")
    current_key = scenario_key(construction_stage, smart_scenario)

    process = root / "process" / "wap_layout_optimization"
    paths = {
        "problem": str(process / "01_problem_definition.json"),
        "candidates": str(process / "02_candidate_layouts.json"),
        "receivers": str(process / "02_receivers.json"),
        "count_search": str(process / "02_wap_count_search.json"),
        "domain_validation": str(process / "04_dense_domain_validation.json"),
        "optimization": str(process / "03_genetic_optimization.json"),
        "sensor_positions": str(root / "final_results" / "sensor_positions.json"),
        "radio_map": str(process / "05_radio_map_surrogate.npz"),
    }
    state["paths"]["wap_optimization"] = paths
    state["optimization_status"] = {"problem": "completed", "surrogate": "pending", "candidates": "pending",
                                    "optimization": "pending"}

    objects, scene_bounds = _scene_geometry(state)
    deploy = _deployable(scene_bounds)
    payload = {
        "objects": objects,
        "propagation": {
            "model": "Ray-Tracing-Assisted Surrogate Optimization",
            "solver": "Sionna RT RadioMapSolver",
            "data_source": "direct numeric path_gain / rss / sinr matrices over the building",
            "note": "GA fitness reads the RadioMapSolver surrogate matrices; no log-distance proxy.",
        },
        "scene_regions_source": str(REGION_FILE),
        "scene_regions_sha256": __import__("hashlib").sha256(REGION_FILE.read_bytes()).hexdigest(),
        "optimization_problem": {
            "construction_stage": construction_stage,
            "sensor_profile": asdict(profile),
            "priority_scenario": smart_scenario,
            "scenario_name": region_data()["scenarios"][current_key]["name"],
            "receiver_count": 600,
            "priority_weight": 2.0,
            "regular_weight": 1.0,
            "wap_count_search": {"start": 1, "method": "minimum feasible count under all-receiver RF thresholds"},
            "quality_requirements": {"min_rsrp_dbm": min_rsrp_dbm,
                                     "min_rsrq_db": min_rsrq_db,
                                     "min_sinr_db": min_sinr_db,
                                     "required_receiver_count": 600},
            "deployable_bounds": deploy,
            "region_key": current_key,
            "spatial_zones": heuristic_zone_bounds(construction_stage, smart_scenario),
            "normalization": NORMALIZATION,
            "measurement": {
                "model": "wideband path-gain approximation of SS metrics",
                "standard_reference": "3GPP TS 38.215 metric definitions; not a waveform-equivalent UE measurement",
                "sss_re_count": SSS_RE_COUNT,
                "sss_subcarrier_indexes": [56, 182],
                "rssi_rbs": RSSI_RBS,
                "rssi_symbols": RSSI_SYMBOLS,
                "rssi_subcarriers": RSSI_SUBCARRIERS,
                "ssb_scs_khz": SSB_SCS_KHZ,
                "normalization_rule": "clip((x-basic)/(reliable-basic),0,1); no-link receivers score 0 and stay in the denominator",
            },
            "constraints": {"wap_z_margin_m": .35, "minimum_wap_distance_m": minimum_wap_distance_m},
            "ga": {"population_size": population_size, "generations": generations,
                   "crossover_probability": crossover_probability, "mutation_probability": mutation_probability,
                   "tournament_size": 4, "elite_count": min(4, population_size - 1), "random_seed": random_seed},
        },
    }
    output = write_json(paths["problem"], payload)
    write_json(state_path, state)
    return success_result(root, TOOL_NAME, output, {
        "state_file": str(state_path),
        "construction_stage": construction_stage,
        "sensor_profile": profile.key,
        "smart_scenario": smart_scenario,
    })
