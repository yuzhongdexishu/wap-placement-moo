"""Project each floor's RF map onto the exact IFC architecture render camera."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import shapely

from plot_fig_common import TARGET_WIDTH_CM, cm_to_in, read_current_result, save_both
from plot_3d_metric_heatmaps import RF_CMAP, METRICS, METHODS_IN_ORDER, case_base, floor_domain
from tools.wap_layout_optimization.define_problem import metric_arrays, read_json
from tools.wap_layout_optimization.generate_candidates import RadioMapSurrogate

ROOT = Path(__file__).resolve().parent
CAMERAS = {
    "frame": ((-59.2489, -83.5677, 48.7016), (0.7511, -1.5677, 6.2016), 17.9204),
    "enclosure": ((-59.1027, -83.3954, 48.6778), (0.8973, -1.3954, 6.1778), 18.7168),
}


def camera_frame(stage: str):
    position, target, scale = CAMERAS[stage]
    view = np.asarray(position, dtype=float) - np.asarray(target, dtype=float)
    view /= np.linalg.norm(view)
    right = np.cross([0.0, 0.0, 1.0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    return np.asarray(target, dtype=float), right, up, scale


def project(points: np.ndarray, stage: str, width: int, height: int) -> np.ndarray:
    """VTK parallel camera: IFC metres to uncropped screenshot pixels."""
    target, right, up, scale = camera_frame(stage)
    delta = np.asarray(points, dtype=float).reshape(-1, 3) - target
    pixels_per_metre = height / (2.0 * scale)
    return np.column_stack((width / 2.0 + delta @ right * pixels_per_metre,
                            height / 2.0 - delta @ up * pixels_per_metre))


def cropped_base(stage: str, floor_name: str):
    path = ROOT / "data/visualization_bases" / f"{stage}_{floor_name}_cutaway.png"
    if not path.is_file():
        raise FileNotFoundError(f"Render the matching floor base first: {path}")
    image = np.asarray(Image.open(path).convert("RGB"))
    height, width = image.shape[:2]
    if (width, height) != (3000, 2250):
        raise ValueError(f"Camera registration expects a 3000x2250 image: {path}")
    yy, xx = np.where(image.min(axis=2) < 245)
    x0, x1 = max(0, int(xx.min()) - 30), min(width, int(xx.max()) + 31)
    y0, y1 = max(0, int(yy.min()) - 30), min(height, int(yy.max()) + 31)
    return image[y0:y1, x0:x1], (x0, y0), (width, height)


@lru_cache(maxsize=2)
def shared_surrogate(stage: str) -> RadioMapSurrogate:
    path = ROOT / "result" / stage / "process/baselines/baseline_surrogate.npz"
    return RadioMapSurrogate.from_files(path, path.with_suffix(".meta.json"))


def surrogate_floor_cells(stage: str, method: str, field: str, domain,
                          receiver_z: float, floor_z: float,
                          size: tuple[int, int], crop: tuple[int, int], norm):
    """Draw exact Sionna map cells; zero-path cells use the lowest colour."""
    surrogate = shared_surrogate(stage)
    state = read_json(ROOT / "result" / stage / "workspace/bim/pipeline_state.json")
    problem = read_json(state["paths"]["wap_optimization"]["problem"])
    profile = problem["optimization_problem"]["sensor_profile"]
    waps = read_json(case_base(stage, method) / "sensor_positions.json")["sensor_positions"]
    tx = []
    for wap in waps:
        index = surrogate.tx_index(wap)
        if float(np.linalg.norm(surrogate.tx_positions[index] - np.asarray(wap))) > 1e-6:
            raise ValueError(f"{stage}/{method}: WAP is absent from shared Sionna map")
        tx.append(index)
    level = int(np.argmin(abs(surrogate.z_levels - receiver_z)))
    if abs(float(surrogate.z_levels[level]) - receiver_z) > 1e-3:
        raise ValueError(f"{stage}: receiver height is absent from shared Sionna map")
    gains = surrogate.path_gain[level, tx].transpose(1, 2, 0)
    ny, nx, n_tx = gains.shape
    metric = metric_arrays(gains.reshape(-1, n_tx), profile)[field].reshape(ny, nx)
    centres = surrogate.cell_centers[level]
    dx, dy = surrogate.plane["cell_size"]
    cells, colours = [], []
    no_path_count = 0
    for iy in range(ny):
        for ix in range(nx):
            x, y = map(float, centres[iy, ix, :2])
            if not bool(shapely.contains_xy(domain, x, y)):
                continue
            value = float(metric[iy, ix])
            if not np.isfinite(value):
                no_path_count += 1
                value = float(norm.vmin)
            x0, x1 = x - dx / 2, x + dx / 2
            y0, y1 = y - dy / 2, y + dy / 2
            corners = np.array([[x0, y0, floor_z], [x1, y0, floor_z],
                                [x1, y1, floor_z], [x0, y1, floor_z]])
            pixel = project(corners, stage, *size) - np.asarray(crop)
            colour = list(RF_CMAP(norm(value)))
            colour[3] = 0.60
            cells.append(pixel)
            colours.append(colour)
    return cells, colours, no_path_count


def render(stage: str, method: str, metric: tuple, floor_name: str,
           output_dir: Path, dpi: int, norm: PowerNorm) -> tuple[Path, int, int]:
    key, field, filename, label, unit = metric
    base, crop, size = cropped_base(stage, floor_name)
    state = read_json(ROOT / "result" / stage / "workspace/bim/pipeline_state.json")
    problem = read_json(state["paths"]["wap_optimization"]["problem"])
    domain, receiver_z, floor_z, _faces = floor_domain(
        stage, problem["optimization_problem"]["priority_scenario"], floor_name)
    report = read_current_result(case_base(stage, method) / filename)
    rows = report["receivers"]
    points = np.asarray([r["position"] for r in rows], dtype=float)
    values = np.asarray([np.nan if r[field] is None else r[field] for r in rows], dtype=float)
    mask = np.isclose(points[:, 2], receiver_z, atol=1e-3) & np.isfinite(values)
    if mask.sum() < 8:
        raise ValueError(f"{stage}/{method}/{floor_name}/{key}: fewer than eight RF points")
    cells, colours, no_path_count = surrogate_floor_cells(
        stage, method, field, domain, receiver_z, floor_z, size, crop, norm)
    if not cells:
        raise ValueError("No registered RF cells")
    image_h, image_w = base.shape[:2]
    fig, ax = plt.subplots(figsize=(cm_to_in(TARGET_WIDTH_CM),
                                    cm_to_in(TARGET_WIDTH_CM) * image_h / image_w + 0.65))
    ax.imshow(base)
    ax.add_collection(PolyCollection(cells, facecolors=colours, edgecolors="none", zorder=3))
    waps = np.asarray(read_json(case_base(stage, method) / "sensor_positions.json")["sensor_positions"])
    levels = [floor_domain(stage, problem["optimization_problem"]["priority_scenario"], name)[1]
              for name in ("1F", "2F", "3F")]
    same_floor = np.argmin(abs(waps[:, 2, None] - np.asarray(levels)), axis=1) == ("1F", "2F", "3F").index(floor_name)
    if same_floor.any():
        xy = project(waps[same_floor], stage, *size) - np.asarray(crop)
        ax.scatter(xy[:, 0], xy[:, 1], s=55, c="#FF2922", edgecolors="black", linewidths=.7, zorder=6)
    ax.set(xlim=(0, image_w), ylim=(image_h, 0))
    ax.axis("off")
    ax.set_title(f"{label} of {stage} · {method} · {floor_name}",
                 fontsize=18.5, fontweight="bold", pad=8)
    colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RF_CMAP), ax=ax,
                             fraction=.027, pad=.012, shrink=.70)
    ticks = np.linspace(norm.vmin, norm.vmax, 5)
    colourbar.set_ticks(ticks)
    colourbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
    colourbar.ax.tick_params(labelsize=13)
    colourbar.ax.set_title(f"{label}\n[{unit}]", fontsize=15, pad=8)
    fig.legend(handles=[Line2D([0], [0], marker="o", color="none",
                               markerfacecolor="#FF2922", markeredgecolor="black",
                               markersize=8, label=f"WAP (n={len(waps)})")],
               loc="lower center", bbox_to_anchor=(.35, .015),
               frameon=False, fontsize=15.5)
    fig.subplots_adjust(left=.01, right=.91, top=.90, bottom=.08)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stage}_{method}_{key}_{floor_name}.png"
    save_both(fig, path, dpi)
    plt.close(fig)
    return path, no_path_count, len(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="frame,enclosure")
    parser.add_argument("--methods", default=",".join(METHODS_IN_ORDER))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/registered_heatmaps")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    coverage_rows = []
    for stage in args.stages.split(","):
        norms = {}
        for metric in METRICS:
            key, field, filename, _label, _unit = metric
            all_values = []
            for method in METHODS_IN_ORDER:
                rows = read_current_result(case_base(stage, method) / filename)["receivers"]
                all_values.extend(float(row[field]) for row in rows if row[field] is not None)
            lo, hi = np.percentile(all_values, [0.1, 99.9])
            if not hi > lo:
                raise ValueError(f"Degenerate {stage}/{key} RF colour scale")
            norms[key] = PowerNorm(gamma=1.5, vmin=float(lo), vmax=float(hi), clip=True)
        for method in args.methods.split(","):
            for metric in METRICS:
                for floor_name in ("1F", "2F", "3F"):
                    path, no_path_count, cell_count = render(
                        stage, method, metric, floor_name,
                        args.out_dir, args.dpi, norms[metric[0]])
                    coverage_rows.append({"stage": stage, "method": method,
                                          "metric": metric[0], "floor": floor_name,
                                          "grid_cells": cell_count,
                                          "zero_path_cells": no_path_count})
                    print(f"saved {path} (zero-path map cells={no_path_count})", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "grid_coverage_report.json"
    report_path.write_text(json.dumps({
        "grid_source": "shared Sionna path-gain map at 0.75 m cell spacing",
        "zero_path_display": "lowest palette colour; not a finite RF measurement",
        "plots": coverage_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
