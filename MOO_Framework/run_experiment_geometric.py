"""Run GcM1/GcM2 geometric layouts on the same PM surrogate and receivers."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from baselines.geometric_layouts import (
    METHOD_DESCRIPTIONS, GeometryGASettings, build_height_layers,
    build_weighted_area_grid, equivalent_service_radius_m,
    optimize_geometry_layout)
from baselines.score import score_baseline
from tools.wap_layout_optimization.define_problem import get_sensor_profile, read_json
from tools.wap_layout_optimization.define_problem import heuristic_zone_bounds

ROOT = Path(__file__).resolve().parent
METHODS = ("GcM1", "GcM2")


def run_case(stage: str, method: str, skip_existing: bool = False) -> dict:
    case = ROOT / "result" / stage / method / "case_manifest.json"
    state = read_json(ROOT / "result" / stage / "workspace/bim/pipeline_state.json")
    problem = read_json(state["paths"]["wap_optimization"]["problem"])
    opt = problem["optimization_problem"]
    if (opt["sensor_profile"]["key"] != "SP3"
            or opt["sensor_profile"]["tx_power_dbm"] != get_sensor_profile("SP3").tx_power_dbm):
        raise RuntimeError(f"{stage} PM results are not the current SP3/23 dBm experiment; rerun MOO first")
    if skip_existing and case.is_file():
        old = read_json(case)
        if (old.get("sensor_profile") == opt["sensor_profile"]["key"]
                and old.get("tx_power_dbm") == opt["sensor_profile"]["tx_power_dbm"]
                and old.get("region_sha256") == problem["scene_regions_sha256"]
                and old.get("wap_count") == opt["selected_wap_count"]):
            return old
    opt["spatial_zones"] = heuristic_zone_bounds(stage, opt["priority_scenario"])
    count = int(opt["selected_wap_count"])
    seed = 20260718 + (0 if stage == "frame" else 100) + (METHODS.index(method) + 1) * 10
    settings = GeometryGASettings(random_seed=seed)
    radius = equivalent_service_radius_m(SimpleNamespace(**opt["sensor_profile"]), 0.0)
    layers = build_height_layers(opt["deployable_bounds"], settings.height_layer_count)
    area = build_weighted_area_grid(opt["deployable_bounds"], opt["spatial_zones"],
        float(opt["priority_weight"]), float(opt["regular_weight"]),
        settings.grid_shape_xy, priority_scenario=opt["priority_scenario"], stage=stage)
    generated = optimize_geometry_layout(method, count, opt["deployable_bounds"],
        problem["objects"], layers, area, radius, settings)
    if not generated["constraint_audit"]["valid"]:
        raise RuntimeError(f"{stage}/{method} invalid geometric layout")
    result = score_baseline(stage, method, generated["best_positions"],
        {"generator": "geometric_layouts", "description": METHOD_DESCRIPTIONS[method],
         "random_seed": seed, "geometric_fitness": generated["best_fitness"],
         "constraint_audit": generated["constraint_audit"]})
    print(f"{stage}/{method}: N={count}, raw_mean={result['mean_composite_score']:.4f}, "
          f"feasible_on_600={result['feasible_on_600']}, "
          f"600-point failures={result['coverage']['failed_quality']}, "
          f"dense failures={result['dense_failed_count']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="frame,enclosure")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    for stage in args.stages.split(","):
        if stage not in ("frame", "enclosure"):
            raise ValueError(stage)
        for method in args.methods.split(","):
            if method not in METHODS:
                raise ValueError(method)
            run_case(stage, method, args.skip_existing)


if __name__ == "__main__":
    main()
