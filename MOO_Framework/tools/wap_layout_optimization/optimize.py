from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .define_problem import (composite_score, dense_validation_positions, metric_arrays,
                             quality_mask, read_json, ss_metrics_from_gains, success_result,
                             verify_region_hash, write_json)
from .generate_candidates import RadioMapSurrogate, constraint_valid, evaluate_fitness

TOOL_NAME = "optimize_wap_layout"


def _tournament(population: list[dict[str, Any]], rng: random.Random, size: int) -> dict[str, Any]:
    return max(rng.sample(population, min(size, len(population))), key=lambda x: x["fitness"])


def _breed(a: list[list[float]], b: list[list[float]], surrogate: RadioMapSurrogate,
           distances: np.ndarray, minimum_distance: float, mutation: float,
           crossover: float, rng: random.Random) -> list[list[float]]:
    """Breed directly on the Sionna transmitter dictionary, keeping WAP spacing."""
    chosen: list[int] = []
    for pa, pb in zip(a, b):
        inherited = pb if rng.random() < crossover * .5 else pa
        preferred = rng.randrange(len(surrogate.tx_positions)) if rng.random() < mutation else surrogate.tx_index(inherited)
        options = list(range(len(surrogate.tx_positions)))
        rng.shuffle(options)
        options.insert(0, preferred)
        pick = next((j for j in options if j not in chosen and
                     (not chosen or np.all(distances[j, chosen] >= minimum_distance))), None)
        if pick is None:
            raise RuntimeError("Could not breed a spaced WAP layout on the candidate dictionary")
        chosen.append(pick)
    return [list(map(float, surrogate.tx_positions[j])) for j in chosen]


def optimize_wap_layout(state_file: str) -> dict[str, Any]:
    """Tool 8: optimize dictionary WAP sites and export all 600 metrics."""
    state_path = Path(state_file).expanduser().resolve(); state = read_json(state_path)
    root, paths = Path(state["project_root"]), state["paths"]["wap_optimization"]
    problem, candidates = read_json(paths["problem"]), read_json(paths["candidates"])
    verify_region_hash(problem)
    opt, ga = problem["optimization_problem"], problem["optimization_problem"]["ga"]
    receivers, objects = candidates["receivers"], problem["objects"]
    surrogate = RadioMapSurrogate.from_files(paths["radio_map"], Path(paths["radio_map"]).with_suffix(".meta.json"))
    rng = random.Random(ga["random_seed"] + 1)
    population = sorted(candidates["population"], key=lambda x: x["fitness"], reverse=True)

    sites = surrogate.tx_positions
    distances = np.linalg.norm(sites[:, None, :] - sites[None, :, :], axis=2)
    minimum_distance = opt["constraints"]["minimum_wap_distance_m"]
    for _ in range(ga["generations"]):
        next_population = [deepcopy(x) for x in population[:ga["elite_count"]]]
        while len(next_population) < ga["population_size"]:
            a = _tournament(population, rng, ga["tournament_size"])
            b = _tournament(population, rng, ga["tournament_size"])
            child = _breed(a["wap_positions"], b["wap_positions"], surrogate,
                           distances, minimum_distance, ga["mutation_probability"],
                           ga["crossover_probability"], rng)
            next_population.append({"wap_positions": child,
                                    "fitness": evaluate_fitness(receivers, child, surrogate, opt)})
        population = sorted(next_population, key=lambda x: x["fitness"], reverse=True)

    # A discrete one-site search refines the best feasible GA individual. It
    # checks coverage before the expensive RF score so every accepted layout
    # keeps the 600-link invariant.
    best_positions = population[0]["wap_positions"]
    best_fitness = float(population[0]["fitness"])
    all_coverage = surrogate.gain_matrix([r["position"] for r in receivers], sites) > 0
    chosen = [surrogate.tx_index(p) for p in best_positions]
    for _ in range(5):
        improved = False
        for slot in range(len(chosen)):
            others = chosen[:slot] + chosen[slot + 1:]
            already_covered = np.any(all_coverage[:, others], axis=1)
            for candidate in range(len(sites)):
                if candidate in others or not np.all(distances[candidate, others] >= minimum_distance):
                    continue
                if not np.all(already_covered | all_coverage[:, candidate]):
                    continue
                trial = chosen.copy(); trial[slot] = candidate
                positions = [list(map(float, sites[j])) for j in trial]
                fitness = evaluate_fitness(receivers, positions, surrogate, opt)
                if fitness > best_fitness + 1e-10:
                    chosen, best_positions, best_fitness = trial, positions, fitness
                    improved = True
        if not improved:
            break
    if not constraint_valid(best_positions, opt["deployable_bounds"], objects,
                            minimum_distance):
        raise RuntimeError("Final WAP layout violates deployment constraints")
    gains = surrogate.gain_matrix([r["position"] for r in receivers], best_positions)
    meets_quality = quality_mask(gains, opt["sensor_profile"], opt["quality_requirements"])
    failed = np.flatnonzero(~meets_quality)
    if len(failed):
        raise RuntimeError(
            f"Final layout has {len(failed)}/600 receivers below the RF quality thresholds")
    probes = dense_validation_positions(problem)
    probe_gains = surrogate.gain_matrix([r["position"] for r in probes], best_positions)
    probe_good = quality_mask(probe_gains, opt["sensor_profile"], opt["quality_requirements"])
    probe_metrics = metric_arrays(probe_gains, opt["sensor_profile"])
    probe_report = {
        "status": "passed" if bool(np.all(probe_good)) else "failed",
        "scope": "independent dense probes of the valid floor domains, not experiment receivers",
        "receiver_count": 600,
        "probe_count": len(probes),
        "failed_probe_count": int(np.count_nonzero(~probe_good)),
        "no_link_probe_count": int(np.count_nonzero(~np.any(probe_gains > 0, axis=1))),
        "failed_by_metric": {
            field: int(np.count_nonzero(~np.isfinite(probe_metrics[field])
                                       | (probe_metrics[field] < opt["quality_requirements"][limit])))
            for field, limit in (("rsrp_dbm", "min_rsrp_dbm"),
                                 ("rsrq_db", "min_rsrq_db"), ("sinr_db", "min_sinr_db"))},
        "failed_by_floor": {floor: sum(not bool(ok) for row, ok in zip(probes, probe_good)
                                       if row["floor"] == floor)
                            for floor in ("1F", "2F", "3F")},
        "first_failed_positions": [probes[i]["position"]
                                   for i in np.flatnonzero(~probe_good)[:20]],
        "quality_requirements": opt["quality_requirements"],
        "region_sha256": problem["scene_regions_sha256"],
    }
    domain_path = Path(paths.get("domain_validation") or
                       root / "process/wap_layout_optimization/04_dense_domain_validation.json")
    write_json(domain_path, probe_report)
    metric_rows = []
    for receiver, gain in zip(receivers, gains):
        metrics = ss_metrics_from_gains(gain, opt["sensor_profile"])
        if metrics["no_link"]:
            raise RuntimeError("Full-coverage invariant failed while exporting receiver metrics")
        metric_rows.append({"receiver_id": receiver["receiver_id"],
                            "position": receiver["position"], "weight": receiver["weight"],
                            "rsrp_dbm": metrics["rsrp_dbm"], "rsrq_db": metrics["rsrq_db"],
                            "sinr_db": metrics["sinr_db"],
                            "composite_score": composite_score(metrics, opt["normalization"])})
    total_weight = sum(r["weight"] for r in metric_rows)
    final = root / "final_results"
    write_json(paths["sensor_positions"], {"sensor_positions": best_positions})
    for key, file_name in (("rsrp_dbm", "receiver_rsrp.json"),
                           ("rsrq_db", "receiver_rsrq.json"),
                           ("sinr_db", "receiver_sinr.json"),
                           ("composite_score", "receiver_composite.json")):
        mean_key = "mean_composite_score" if key == "composite_score" else f"mean_{key}"
        value = sum(r["weight"] * r[key] for r in metric_rows) / total_weight
        write_json(final / file_name, {mean_key: value,
                                      "source": "Sionna RadioMapSolver surrogate",
                                      "experiment_schema": "surrogate-v2-600-domain-aabb",
                                      "region_sha256": problem["scene_regions_sha256"],
                                      "coverage": {"receivers": 600, "no_link": 0,
                                                   "failed_quality": 0},
                                      "quality_requirements": opt["quality_requirements"],
                                      "measurement_model": opt["measurement"]["model"],
                                      "proof_scope": "the selected 600 receiver positions and 150 candidate WAP sites",
                                      "domain_validation": str(domain_path),
                                      "domain_validation_status": probe_report["status"],
                                      "receivers": [{"receiver_id": r["receiver_id"],
                                                     "position": r["position"],
                                                     "weight": r["weight"], key: r[key]}
                                                    for r in metric_rows]})
    output = write_json(paths["optimization"], {"best_positions": best_positions,
                                                "surrogate_fitness": best_fitness,
                                                "receiver_count": 600, "no_link_count": 0,
                                                "failed_quality_count": 0,
                                                "actual_wap_count": len(best_positions),
                                                "selected_wap_count": opt["selected_wap_count"],
                                                "minimum_count_proven": read_json(paths["count_search"])["minimum_count_proven"],
                                                "proof_scope": "the selected 600 receiver positions and 150 candidate WAP sites",
                                                "count_search_file": paths["count_search"],
                                                "domain_validation_file": str(domain_path),
                                                "domain_validation_status": probe_report["status"],
                                                "quality_requirements": opt["quality_requirements"]})
    state["optimization_status"]["optimization"] = "completed"
    write_json(state_path, state)
    return success_result(root, TOOL_NAME, output, {"state_file": str(state_path),
                                                   "receiver_count": 600, "no_link_count": 0,
                                                   "failed_quality_count": 0,
                                                   "domain_validation_status": probe_report["status"],
                                                   "domain_failed_probe_count": probe_report["failed_probe_count"],
                                                   "actual_wap_count": len(best_positions)})
