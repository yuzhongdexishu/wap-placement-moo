"""Build the Sionna surrogate and the current scenario's 600 receivers."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import numpy as np

from .define_problem import PROJECT_ROOT, generate_receivers, read_json, success_result, verify_region_hash, write_json
from .generate_candidates import RadioMapSurrogate, build_radio_map_surrogate, is_valid_wap

TOOL_NAME = "build_sionna_surrogate"
SHARED_ROOT = PROJECT_ROOT / "result/six_llm/shared_surrogates"


def _shared_cache(state: dict, problem: dict, receivers: list[dict],
                  dictionary: list[dict], solver_options: dict) -> tuple[Path, str] | None:
    """Path gain depends on scene/frequency/sites, not the five TX power levels."""
    task_root = Path(state["project_root"]).resolve()
    if not task_root.is_relative_to((PROJECT_ROOT / "result/six_llm/work").resolve()):
        return None
    opt = problem["optimization_problem"]
    evidence = {
        "stage": opt["construction_stage"], "scenario": opt["priority_scenario"],
        "ifc": hashlib.sha256(Path(state["source_ifc"]).read_bytes()).hexdigest(),
        "materials": {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for key, path in state["paths"]["radio_tables"].items()},
        "regions": problem["scene_regions_sha256"],
        "frequency_ghz": opt["sensor_profile"]["frequency_ghz"],
        "receivers": [row["position"] for row in receivers],
        "transmitters": [row["position"] for row in dictionary],
        "solver": solver_options,
    }
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    name = f"{opt['construction_stage']}_{opt['priority_scenario']}_{digest[:16]}"
    return SHARED_ROOT / f"{name}.npz", digest


def _reuse_shared_map(state: dict, problem: dict, receivers: list[dict],
                      dictionary: list[dict], paths: dict,
                      solver_options: dict) -> RadioMapSurrogate | None:
    cache_info = _shared_cache(state, problem, receivers, dictionary, solver_options)
    if cache_info is None:
        return None
    cache, digest = cache_info
    meta = cache.with_suffix(".meta.json")
    if not cache.is_file() or not meta.is_file():
        return None
    metadata = read_json(meta)
    if metadata.get("cache_key_sha256") != digest:
        raise RuntimeError(f"Shared path-gain cache metadata mismatch: {cache}")
    with np.load(cache) as stored:
        expected = np.asarray([row["position"] for row in dictionary], dtype=float)
        if (stored["tx_positions"].shape != expected.shape or
                not np.allclose(stored["tx_positions"], expected, rtol=0, atol=1e-6)):
            raise RuntimeError(f"Shared path-gain cache sites mismatch: {cache}")
    target = Path(paths["radio_map"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache, target)
    write_json(target.with_suffix(".meta.json"), {**metadata,
        "cache": {"shared_path_gain": str(cache), "source_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}})
    return RadioMapSurrogate.from_files(target, target.with_suffix(".meta.json"))


def _store_shared_map(state: dict, problem: dict, receivers: list[dict],
                      dictionary: list[dict], solver_options: dict,
                      surrogate: RadioMapSurrogate) -> None:
    cache_info = _shared_cache(state, problem, receivers, dictionary, solver_options)
    if cache_info is None:
        return
    cache, digest = cache_info
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, path_gain=surrogate.path_gain,
                            cell_centers=surrogate.cell_centers,
                            z_levels=surrogate.z_levels,
                            tx_positions=surrogate.tx_positions)
        os.replace(temporary, cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_json(cache.with_suffix(".meta.json"), {**surrogate.metadata,
               "cache_key_sha256": digest,
               "stored_fields": ["path_gain", "cell_centers", "z_levels", "tx_positions"]})


def _candidate_dictionary(legal: list[dict], opt: dict, objects: list[dict],
                          count: int = 150) -> list[dict]:
    """Mix legal receiver sites with scene-centre and regular installation anchors.

    Receiver-only sampling omitted the legal centre sites used by the expert
    baselines by several metres. The anchors are derived from the same problem
    geometry before any baseline result is read, and the dictionary stays at
    exactly 150 candidate sites for all methods and stages.
    """
    bounds = opt["deployable_bounds"]
    lo, hi = bounds["min"], bounds["max"]
    margin = min(0.5, max((float(hi[2]) - float(lo[2])) * 0.08, 0.1))
    z_lo, z_hi = float(lo[2]) + margin, float(hi[2]) - margin
    levels = [z_lo + (z_hi - z_lo) * (i + 0.5) / 3.0 for i in range(3)]
    anchors: list[list[float]] = []
    for zone in opt["spatial_zones"]:
        if zone.get("scenario") == opt["priority_scenario"]:
            x = (float(zone["x_min"]) + float(zone["x_max"])) / 2.0
            y = (float(zone["y_min"]) + float(zone["y_max"])) / 2.0
            anchors.extend([[x, y, z] for z in levels])
    for z in levels:
        for xf in (0.25, 0.5, 0.75):
            for yf in (0.25, 0.5, 0.75):
                anchors.append([float(lo[0]) + xf * (float(hi[0]) - float(lo[0])),
                                float(lo[1]) + yf * (float(hi[1]) - float(lo[1])), z])
    selected: list[dict] = []
    seen: set[tuple[float, float, float]] = set()
    def add(point: list[float]) -> None:
        rounded = tuple(round(float(v), 6) for v in point)
        if rounded not in seen and is_valid_wap(rounded, bounds, objects):
            selected.append({"position": list(rounded)})
            seen.add(rounded)
    for point in anchors:
        if len(selected) >= min(30, count):
            break
        add(point)
    remaining = count - len(selected)
    for index in np.linspace(0, len(legal) - 1, remaining, dtype=int):
        add(legal[int(index)]["position"])
    if len(selected) < count:
        for row in legal:
            add(row["position"])
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"Only {len(selected)} distinct legal candidate WAP sites")
    return selected


def _reuse_pm_map(state: dict, problem: dict, receivers: list[dict], dictionary: list[dict],
                  paths: dict, solver_options: dict) -> RadioMapSurrogate | None:
    """Reuse a verified PM map for a same-condition six-model benchmark task."""
    task_root = Path(state["project_root"]).resolve()
    benchmark_root = (PROJECT_ROOT / "result/six_llm/work").resolve()
    if not task_root.is_relative_to(benchmark_root):
        return None
    stage = problem["optimization_problem"]["construction_stage"]
    pm_root = PROJECT_ROOT / "result" / stage
    pm_state_path = pm_root / "workspace/bim/pipeline_state.json"
    if not pm_state_path.is_file():
        return None
    pm_state = read_json(pm_state_path)
    pm_paths = pm_state.get("paths", {}).get("wap_optimization", {})
    required = ("problem", "receivers", "radio_map")
    if any(not Path(pm_paths.get(key, "")).is_file() for key in required):
        return None
    pm_problem = read_json(pm_paths["problem"])
    for key in ("scene_regions_sha256", "objects"):
        if pm_problem.get(key) != problem.get(key):
            return None
    for key in ("sensor_profile", "priority_scenario", "deployable_bounds"):
        if pm_problem["optimization_problem"].get(key) != problem["optimization_problem"].get(key):
            return None
    if read_json(pm_paths["receivers"])["receivers"] != receivers:
        return None
    pm_map = Path(pm_paths["radio_map"])
    pm_meta = read_json(pm_map.with_suffix(".meta.json"))
    if any(pm_meta["solver"].get(key) != value for key, value in solver_options.items()):
        return None
    with np.load(pm_map) as stored:
        expected_tx = np.asarray([r["position"] for r in dictionary], dtype=float)
        if stored["tx_positions"].shape != expected_tx.shape or not np.allclose(
                stored["tx_positions"], expected_tx, rtol=0.0, atol=1e-6):
            return None
    digest = hashlib.sha256(pm_map.read_bytes()).hexdigest()
    manifest_path = PROJECT_ROOT / "result/six_llm/shared_surrogate_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    previous = manifest.get(stage)
    if previous is not None and previous != digest:
        raise RuntimeError(f"PM {stage} surrogate changed during the six-model benchmark")
    manifest[stage] = digest
    write_json(manifest_path, manifest)
    target = Path(paths["radio_map"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pm_map, target)
    write_json(target.with_suffix(".meta.json"), {**pm_meta,
        "cache": {"reused_from_pm": True, "source_sha256": digest}})
    return RadioMapSurrogate.from_files(target, target.with_suffix(".meta.json"))


def build_sionna_surrogate(state_file: str, max_depth: int = 3,
                          los: bool = True, specular_reflection: bool = True,
                          diffraction: bool = False, refraction: bool = True,
                          samples_per_tx: int = 1000000) -> dict:
    state_path = Path(state_file).expanduser().resolve()
    state = read_json(state_path)
    root, paths = Path(state["project_root"]), state["paths"]["wap_optimization"]
    problem = read_json(paths["problem"])
    verify_region_hash(problem)
    opt = problem["optimization_problem"]
    receivers = generate_receivers(opt["construction_stage"], opt["priority_scenario"],
                                   count=600, problem=problem)
    legal = [r for r in receivers if is_valid_wap(r["position"], opt["deployable_bounds"], problem["objects"])]
    if len(legal) < 150:
        raise RuntimeError(f"Only {len(legal)} legal WAP dictionary sites")
    dictionary = _candidate_dictionary(legal, opt, problem["objects"], count=150)
    solver_options = {"max_depth": max_depth, "los": los,
                      "specular_reflection": specular_reflection,
                      "diffraction": diffraction, "refraction": refraction,
                      "samples_per_tx": samples_per_tx, "seed": 41,
                      "cell_size_m": 0.75}
    surrogate = _reuse_shared_map(state, problem, receivers, dictionary, paths, solver_options)
    if surrogate is None:
        surrogate = _reuse_pm_map(state, problem, receivers, dictionary, paths, solver_options)
    cache_hit = surrogate is not None
    if surrogate is None:
        surrogate = build_radio_map_surrogate(
            state, problem, dictionary, plane_receivers=receivers,
            max_depth=max_depth, los=los, specular_reflection=specular_reflection,
            diffraction=diffraction, refraction=refraction,
            samples_per_tx=samples_per_tx)
        Path(paths["radio_map"]).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(paths["radio_map"], path_gain=surrogate.path_gain,
                            rss=surrogate.rss, sinr=surrogate.sinr,
                            cell_centers=surrogate.cell_centers,
                            z_levels=surrogate.z_levels,
                            tx_positions=surrogate.tx_positions)
        _store_shared_map(state, problem, receivers, dictionary, solver_options, surrogate)
    metadata_path = Path(paths["radio_map"]).with_suffix(".meta.json")
    write_json(metadata_path, surrogate.metadata)
    output = write_json(paths["receivers"], {"receivers": receivers,
                                            "region_sha256": problem["scene_regions_sha256"]})
    state["optimization_status"].update({"surrogate": "completed", "candidates": "pending",
                                         "optimization": "pending"})
    write_json(state_path, state)
    return success_result(root, TOOL_NAME, output, {"state_file": str(state_path),
                  "receiver_count": 600, "tx_dictionary_count": 150,
                  "cache_hit": cache_hit,
                  "radio_map_shape": [int(v) for v in surrogate.path_gain.shape]})
