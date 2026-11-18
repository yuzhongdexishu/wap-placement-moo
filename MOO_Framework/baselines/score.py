"""Score baseline layouts on exact-site Sionna data and the PM 600 points."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tools.wap_layout_optimization.define_problem import (
    composite_score, dense_validation_positions, quality_mask, read_json,
    ss_metrics_from_gains, verify_region_hash, write_json)
from tools.wap_layout_optimization.generate_candidates import RadioMapSurrogate, constraint_valid


def _nearest_legal_sites(original: list[list[float]], sites: np.ndarray,
                         min_distance: float) -> list[list[float]]:
    """Project the heuristic layout onto distinct, spaced dictionary sites."""
    selected: list[int] = []
    for position in original:
        order = np.argsort(np.linalg.norm(sites - np.asarray(position), axis=1))
        chosen = next((int(j) for j in order if j not in selected and
                       all(float(np.linalg.norm(sites[j] - sites[k])) >= min_distance
                           for k in selected)), None)
        if chosen is None:
            raise RuntimeError("Baseline cannot be projected onto spaced surrogate sites")
        selected.append(chosen)
    return [list(map(float, sites[j])) for j in selected]


def score_baseline(stage: str, method: str, original: list[list[float]],
                   metadata: dict, *, exact_surrogate_file: Path | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    pm = root / "result" / stage
    state = read_json(pm / "workspace/bim/pipeline_state.json")
    paths = state["paths"]["wap_optimization"]
    problem = read_json(paths["problem"])
    verify_region_hash(problem)
    opt = problem["optimization_problem"]
    if len(original) != opt["selected_wap_count"]:
        raise ValueError(f"{stage}/{method}: baseline and PM WAP counts differ")
    surrogate_file = Path(exact_surrogate_file or paths["radio_map"])
    surrogate = RadioMapSurrogate.from_files(
        surrogate_file, surrogate_file.with_suffix(".meta.json"))
    receivers = read_json(paths["candidates"])["receivers"]
    if len(receivers) != 600:
        raise RuntimeError("Baseline comparison requires exactly 600 PM receivers")
    if exact_surrogate_file is None:
        positions = _nearest_legal_sites(original, surrogate.tx_positions,
                                        opt["constraints"]["minimum_wap_distance_m"])
    else:
        positions = [[float(v) for v in point] for point in original]
        if any(float(np.min(np.linalg.norm(surrogate.tx_positions - np.asarray(p), axis=1))) > 1e-6
               for p in positions):
            raise RuntimeError("Baseline site missing from its exact Sionna surrogate")
    if not constraint_valid(positions, opt["deployable_bounds"], problem["objects"],
                            opt["constraints"]["minimum_wap_distance_m"]):
        raise RuntimeError("Projected baseline violates installation constraints")
    displacement = np.linalg.norm(np.asarray(positions) - np.asarray(original), axis=1)
    gains = surrogate.gain_matrix([r["position"] for r in receivers], positions)
    passed = quality_mask(gains, opt["sensor_profile"], opt["quality_requirements"])
    rows = []
    for receiver, gain in zip(receivers, gains):
        metrics = ss_metrics_from_gains(gain, opt["sensor_profile"])
        rows.append({"receiver_id": receiver["receiver_id"],
                     "position": receiver["position"], "weight": receiver["weight"],
                     "rsrp_dbm": metrics["rsrp_dbm"], "rsrq_db": metrics["rsrq_db"],
                     "sinr_db": metrics["sinr_db"],
                     "composite_score": composite_score(metrics, opt["normalization"])})
    case = pm / method
    final = case / "final_results"
    write_json(final / "sensor_positions.json", {"sensor_positions": positions,
               "original_heuristic_positions": original,
               "snapped_to_shared_dictionary": exact_surrogate_file is None})
    total_weight = sum(r["weight"] for r in rows)
    means = {}
    coverage = {"receivers": 600,
                "no_link": int(np.count_nonzero(~np.any(gains > 0, axis=1))),
                "failed_quality": int(np.count_nonzero(~passed))}
    for key, name in (("rsrp_dbm", "rsrp"), ("rsrq_db", "rsrq"),
                      ("sinr_db", "sinr"), ("composite_score", "composite")):
        valid = [r for r in rows if r[key] is not None and math.isfinite(float(r[key]))]
        denom = sum(r["weight"] for r in valid)
        mean = sum(r["weight"] * r[key] for r in valid) / denom if denom else None
        means[key] = mean
        mean_key = "mean_composite_score" if key == "composite_score" else f"mean_{key}"
        write_json(final / f"receiver_{name}.json", {mean_key: mean,
            "source": "Sionna RadioMapSolver surrogate",
            "surrogate_file": str(surrogate_file),
            "region_sha256": problem["scene_regions_sha256"],
            "coverage": coverage, "quality_requirements": opt["quality_requirements"],
            "receivers": [{"receiver_id": r["receiver_id"], "position": r["position"],
                           "weight": r["weight"], key: r[key]} for r in rows]})
    probes = dense_validation_positions(problem)
    dense = surrogate.gain_matrix([r["position"] for r in probes], positions)
    dense_pass = quality_mask(dense, opt["sensor_profile"], opt["quality_requirements"])
    manifest = {"stage": stage, "method": method, "wap_count": len(positions),
                "sensor_profile": opt["sensor_profile"]["key"],
                "tx_power_dbm": opt["sensor_profile"]["tx_power_dbm"],
                "region_sha256": problem["scene_regions_sha256"],
                "same_receiver_set_as_pm": True,
                "surrogate_site_projection": exact_surrogate_file is None,
                "surrogate_file": str(surrogate_file),
                "site_projection_mean_m": float(displacement.mean()),
                "site_projection_max_m": float(displacement.max()),
                "source_layout": metadata,
                "mean_composite_score": means["composite_score"],
                "coverage": coverage,
                "feasible_on_600": bool(np.all(passed)),
                "dense_probe_count": len(probes),
                "dense_failed_count": int(np.count_nonzero(~dense_pass))}
    write_json(case / "case_manifest.json", manifest)
    return manifest
