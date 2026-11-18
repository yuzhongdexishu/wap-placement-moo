"""Stack three camera-registered floor cutaways into one 3D RF overview."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

from plot_fig_common import TARGET_WIDTH_CM, cm_to_in, read_current_result, save_both
from plot_3d_metric_heatmaps import METRICS, METHODS_IN_ORDER, RF_CMAP, case_base, floor_domain
from plot_registered_floor_heatmaps import project, surrogate_floor_cells
from tools.wap_layout_optimization.define_problem import read_json

ROOT = Path(__file__).resolve().parent
FLOORS = ("1F", "2F", "3F")


def _layers(stage: str):
    images = []
    for floor in FLOORS:
        path = ROOT / "data/visualization_bases" / f"{stage}_{floor}_cutaway.png"
        image = np.asarray(Image.open(path).convert("RGB"))
        if image.shape[:2] != (2250, 3000):
            raise ValueError(f"Cutaway camera canvas changed: {path}")
        images.append(image)
    yy, xx = np.where(np.any([im.min(axis=2) < 245 for im in images], axis=0))
    x0, x1 = max(0, int(xx.min()) - 30), min(3000, int(xx.max()) + 31)
    y0, y1 = max(0, int(yy.min()) - 30), min(2250, int(yy.max()) + 31)
    return [image[y0:y1, x0:x1] for image in images], (x0, y0)


def _norm(stage: str, filename: str, field: str) -> PowerNorm:
    values = []
    for method in METHODS_IN_ORDER:
        rows = read_current_result(case_base(stage, method) / filename)["receivers"]
        values.extend(float(row[field]) for row in rows if row[field] is not None)
    lo, hi = np.percentile(values, [0.1, 99.9])
    return PowerNorm(gamma=1.5, vmin=float(lo), vmax=float(hi), clip=True)


def plot_one(stage: str, method: str, metric: tuple, output: Path, dpi: int) -> None:
    key, field, filename, label, unit = metric
    image_layers, crop = _layers(stage)
    height, width = image_layers[0].shape[:2]
    state = read_json(ROOT / "result" / stage / "workspace/bim/pipeline_state.json")
    problem = read_json(state["paths"]["wap_optimization"]["problem"])
    scenario = problem["optimization_problem"]["priority_scenario"]
    rows = read_current_result(case_base(stage, method) / filename)["receivers"]
    positions = np.asarray([row["position"] for row in rows], dtype=float)
    values = np.asarray([np.nan if row[field] is None else row[field] for row in rows], dtype=float)
    norm = _norm(stage, filename, field)
    fig, ax = plt.subplots(figsize=(cm_to_in(TARGET_WIDTH_CM),
                                    cm_to_in(TARGET_WIDTH_CM) * height / width + .65))
    ax.set_facecolor("white")
    for image, floor in zip(image_layers, FLOORS):
        domain, receiver_z, floor_z, _ = floor_domain(stage, scenario, floor)
        # The three architecture cutaways use one identical camera and image frame.
        alpha = np.clip((255.0 - image.min(axis=2)) / 40.0, 0.0, 1.0) * .62
        rgba = np.dstack((image / 255.0, alpha))
        ax.imshow(rgba, zorder=1, interpolation="bilinear")
        mask = np.isclose(positions[:, 2], receiver_z, atol=1e-3) & np.isfinite(values)
        if mask.sum() < 8:
            raise ValueError(f"{stage}/{method}/{floor}/{key}: insufficient RF samples")
        cells, colours, _no_path_count = surrogate_floor_cells(
            stage, method, field, domain, receiver_z, floor_z,
            (3000, 2250), crop, norm)
        ax.add_collection(PolyCollection(cells, facecolors=colours,
                                         edgecolors="none", zorder=2))
        xmin, ymin, xmax, ymax = domain.bounds
        xy = project([[xmin + .6, ymax - .6, floor_z]], stage, 3000, 2250)[0] - crop
        ax.text(xy[0], xy[1], floor, fontsize=14.5, color="#26313A", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": .8})
    waps = np.asarray(read_json(case_base(stage, method) / "sensor_positions.json")["sensor_positions"])
    wap_xy = project(waps, stage, 3000, 2250) - crop
    ax.scatter(wap_xy[:, 0], wap_xy[:, 1], s=45, color="#FF2922",
               edgecolors="black", linewidths=.7, zorder=8)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    ax.set_title(f"{label} of {stage} · {method}", fontsize=18.5,
                 fontweight="bold", pad=8)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    save_both(fig, output, dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="frame,enclosure")
    parser.add_argument("--methods", default=",".join(METHODS_IN_ORDER))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/stacked_registered_heatmaps")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for stage in args.stages.split(","):
        for method in args.methods.split(","):
            for metric in METRICS:
                output = args.out_dir / f"{stage}_{method}_{metric[0]}_stacked.png"
                plot_one(stage, method, metric, output, args.dpi)
                print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
