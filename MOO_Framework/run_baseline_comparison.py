"""Re-score PM and EbM/GcM original sites on one exact-site Sionna map."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from baselines.score import score_baseline
from tools.wap_layout_optimization.define_problem import (
    composite_score, get_sensor_profile, quality_mask, read_json,
    ss_metrics_from_gains, write_json)
from tools.wap_layout_optimization.generate_candidates import build_radio_map_surrogate, constraint_valid

ROOT = Path(__file__).resolve().parent
METHODS = ("EbM1", "EbM2", "EbM3", "GcM1", "GcM2")


def export_comparison_csv() -> Path:
    """Export the paired 600-point objective and feasibility for every method."""
    output = ROOT / "result/comparison_table.csv"
    columns = ("stage", "method", "wap_count", "mean_composite_score",
               "failed_600", "no_link_600", "feasible_on_600",
               "sensor_profile", "samples_per_tx", "surrogate_file")
    rows = []
    for stage in ("frame", "enclosure"):
        base = ROOT / "result" / stage
        pm = read_json(base / "process/baselines/pm_joint_receiver_composite.json")
        reports = [pm] + [read_json(base / method / "case_manifest.json") for method in METHODS]
        samples = read_json(base / "process/baselines/baseline_surrogate.meta.json")["solver"]["samples_per_tx"]
        for report in reports:
            coverage = report["coverage"]
            rows.append({"stage": stage, "method": report["method"],
                         "wap_count": report["wap_count"],
                         "mean_composite_score": report["mean_composite_score"],
                         "failed_600": coverage["failed_quality"],
                         "no_link_600": coverage["no_link"],
                         "feasible_on_600": coverage["failed_quality"] == 0 and coverage["no_link"] == 0,
                         "sensor_profile": report["sensor_profile"],
                         "samples_per_tx": samples,
                         "surrogate_file": report["surrogate_file"]})
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_pm_joint_metric_files(base: Path, report: dict, positions: list[list[float]]) -> None:
    """Write the paired-map PM rows in the same schema as baseline metric files."""
    write_json(base / "sensor_positions.json", {"sensor_positions": positions})
    for name, field in (("rsrp", "rsrp_dbm"), ("rsrq", "rsrq_db"),
                        ("sinr", "sinr_db"), ("composite", "composite_score")):
        valid = [row for row in report["receivers"] if row[field] is not None]
        denominator = sum(row["weight"] for row in valid)
        mean = sum(row["weight"] * row[field] for row in valid) / denominator if denominator else None
        mean_key = "mean_composite_score" if name == "composite" else f"mean_{field}"
        write_json(base / f"receiver_{name}.json", {
            mean_key: mean, "source": report["source"],
            "surrogate_file": report["surrogate_file"],
            "region_sha256": report["region_sha256"],
            "coverage": report["coverage"],
            "receivers": [{"receiver_id": row["receiver_id"],
                           "position": row["position"], "weight": row["weight"],
                           field: row[field]} for row in report["receivers"]]})


def _score_pm_on_map(stage: str, problem: dict, receivers: list[dict],
                     surrogate, pm_positions: list[list[float]], map_file: Path) -> dict:
    opt = problem["optimization_problem"]
    profile = opt["sensor_profile"]
    gains = surrogate.gain_matrix([r["position"] for r in receivers], pm_positions)
    passed = quality_mask(gains, profile, opt["quality_requirements"])
    rows = []
    for receiver, gain in zip(receivers, gains):
        metrics = ss_metrics_from_gains(gain, profile)
        rows.append({"receiver_id": receiver["receiver_id"],
                     "position": receiver["position"], "weight": receiver["weight"],
                     "rsrp_dbm": metrics["rsrp_dbm"],
                     "rsrq_db": metrics["rsrq_db"],
                     "sinr_db": metrics["sinr_db"],
                     "composite_score": composite_score(metrics, opt["normalization"])})
    mean = sum(r["weight"] * r["composite_score"] for r in rows) / sum(r["weight"] for r in rows)
    return {"stage": stage, "method": "PM", "wap_count": len(pm_positions),
            "mean_composite_score": mean,
            "source": "Sionna RadioMapSolver surrogate",
            "surrogate_file": str(map_file),
            "sensor_profile": profile["key"],
            "tx_power_dbm": profile["tx_power_dbm"],
            "region_sha256": problem["scene_regions_sha256"],
            "coverage": {"receivers": len(receivers),
                         "no_link": int(np.count_nonzero(~np.any(gains > 0, axis=1))),
                         "failed_quality": int(np.count_nonzero(~passed))},
            "receivers": rows}


def run_stage(stage: str) -> list[dict]:
    pm = ROOT / "result" / stage
    state = read_json(pm / "workspace/bim/pipeline_state.json")
    paths = state["paths"]["wap_optimization"]
    problem = read_json(paths["problem"])
    pm_profile = problem["optimization_problem"]["sensor_profile"]
    if pm_profile["key"] != "SP3" or pm_profile["tx_power_dbm"] != get_sensor_profile("SP3").tx_power_dbm:
        raise RuntimeError(f"{stage} PM results are not the current SP3/23 dBm experiment; rerun MOO first")
    receivers = read_json(paths["candidates"])["receivers"]
    settings = read_json(Path(paths["radio_map"]).with_suffix(".meta.json"))["solver"]
    layouts = {}
    metadata = {}
    sites = []
    seen = set()
    pm_positions = read_json(paths["sensor_positions"])["sensor_positions"]
    if len(pm_positions) != problem["optimization_problem"]["selected_wap_count"]:
        raise RuntimeError(f"{stage} PM WAP count differs from the defined problem")
    for point in pm_positions:
        key = tuple(round(float(v), 6) for v in point)
        if key not in seen:
            sites.append({"position": [float(v) for v in point]})
            seen.add(key)
    for method in METHODS:
        case = pm / method
        source = read_json(case / "final_results/sensor_positions.json")
        original = source["original_heuristic_positions"]
        if not constraint_valid(original, problem["optimization_problem"]["deployable_bounds"],
                                problem["objects"],
                                problem["optimization_problem"]["constraints"]["minimum_wap_distance_m"]):
            raise RuntimeError(f"Original {stage}/{method} layout violates deployment constraints")
        layouts[method] = original
        case_manifest = read_json(case / "case_manifest.json")
        if (case_manifest.get("sensor_profile") != problem["optimization_problem"]["sensor_profile"]["key"]
                or case_manifest.get("tx_power_dbm") != problem["optimization_problem"]["sensor_profile"]["tx_power_dbm"]):
            raise RuntimeError(f"{stage}/{method} baseline belongs to an old SP configuration; regenerate it")
        metadata[method] = case_manifest["source_layout"]
        for point in original:
            key = tuple(round(float(v), 6) for v in point)
            if key not in seen:
                sites.append({"position": [float(v) for v in point]})
                seen.add(key)
    if not sites:
        raise RuntimeError("Generate expert and geometric baseline layouts first")
    surrogate = build_radio_map_surrogate(
        state, problem, sites, plane_receivers=receivers,
        max_depth=int(settings["max_depth"]), los=bool(settings["los"]),
        specular_reflection=bool(settings["specular_reflection"]),
        diffraction=bool(settings["diffraction"]), refraction=bool(settings["refraction"]),
        samples_per_tx=int(settings["samples_per_tx"]))
    out = pm / "process/baselines/baseline_surrogate.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, path_gain=surrogate.path_gain, rss=surrogate.rss,
                        sinr=surrogate.sinr, cell_centers=surrogate.cell_centers,
                        z_levels=surrogate.z_levels, tx_positions=surrogate.tx_positions)
    write_json(out.with_suffix(".meta.json"), surrogate.metadata)
    pm_report = _score_pm_on_map(stage, problem, receivers, surrogate, pm_positions, out)
    write_json(out.parent / "pm_joint_receiver_composite.json", pm_report)
    write_pm_joint_metric_files(out.parent, pm_report, pm_positions)
    print(f"{stage}/PM: paired-map F={pm_report['mean_composite_score']:.4f}, "
          f"failed600={pm_report['coverage']['failed_quality']}", flush=True)
    results = []
    for method in METHODS:
        result = score_baseline(stage, method, layouts[method], metadata[method],
                                exact_surrogate_file=out)
        results.append(result)
        print(f"{stage}/{method}: exact-site raw_mean={result['mean_composite_score']:.4f}, "
              f"feasible_on_600={result['feasible_on_600']}, "
              f"failed600={result['coverage']['failed_quality']}, "
              f"dense_failed={result['dense_failed_count']}", flush=True)
    return results


def main() -> None:
    rows = [row for stage in ("frame", "enclosure") for row in run_stage(stage)]
    write_json(ROOT / "result/baseline_comparison_summary.json", {"cases": rows,
        "method": "PM and original baseline sites evaluated by one Sionna RadioMapSolver per stage",
        "pm_joint": {stage: {
            key: value for key, value in read_json(
                ROOT / "result" / stage / "process/baselines/pm_joint_receiver_composite.json").items()
            if key != "receivers"}
            for stage in ("frame", "enclosure")}})
    export_comparison_csv()


if __name__ == "__main__":
    main()
