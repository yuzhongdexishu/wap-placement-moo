"""Plot PM and baseline scores together with 600-point RF feasibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import numpy as np

from plot_fig_common import TARGET_WIDTH_CM, cm_to_in, read_current_result, save_both

ROOT = Path(__file__).resolve().parent
GROUPS = {"expert-driven methods": ("EbM1", "EbM2", "EbM3", "PM"),
          "geometric coverage methods": ("GcM1", "GcM2", "PM")}
OUTPUT_NAMES = {"expert-driven methods": "expert", "geometric coverage methods": "geometric"}
METHODS = tuple(dict.fromkeys(method for group in GROUPS.values() for method in group))
COLORS = {"EbM1": "#4A7FB8", "EbM2": "#C79A3B", "EbM3": "#4FA37C",
          "GcM1": "#4A7FB8", "GcM2": "#C79A3B", "PM": "#A93B4F"}


def _base(stage: str, method: str) -> Path:
    return ROOT / "result" / stage / ("" if method == "PM" else method) / "final_results"


def _case(stage: str, method: str) -> dict:
    base = _base(stage, method)
    result_path = (ROOT / "result" / stage / "process/baselines/pm_joint_receiver_composite.json"
                   if method == "PM" else base / "receiver_composite.json")
    result = read_current_result(result_path)
    rows = result["receivers"]
    weights = np.asarray([float(row["weight"]) for row in rows])
    scores = np.asarray([float(row["composite_score"]) for row in rows])
    expected = float(np.dot(weights, scores) / weights.sum())
    if not np.isclose(expected, result["mean_composite_score"], rtol=1e-6):
        raise ValueError(f"Composite mean disagrees with weighted receiver rows: {base}")
    positions = json.loads((base / "sensor_positions.json").read_text(encoding="utf-8"))["sensor_positions"]
    failed = int(result["coverage"]["failed_quality"])
    no_link = int(result["coverage"]["no_link"])
    valid = failed == 0 and no_link == 0
    return {"score": expected, "feasible": valid,
            "failed": failed,
            "n_waps": len(positions),
            "surrogate_file": result.get("surrogate_file"),
            "receiver_positions": [row["position"] for row in rows]}


def plot_group(group: str, output: Path, dpi: int) -> None:
    methods = GROUPS[group]
    fig, axes = plt.subplots(1, 2, figsize=(cm_to_in(TARGET_WIDTH_CM), 3.1))
    for ax, stage in zip(axes, ("frame", "enclosure")):
        cases = {method: _case(stage, method) for method in methods}
        reference = cases["PM"]
        for method, value in cases.items():
            if value["n_waps"] != reference["n_waps"]:
                raise ValueError(f"WAP-count mismatch in {stage}/{method}")
            if value["receiver_positions"] != reference["receiver_positions"]:
                raise ValueError(f"Receiver-set mismatch in {stage}/{method}")
            if value["surrogate_file"] != reference["surrogate_file"]:
                raise ValueError(f"Propagation-map mismatch in {stage}/{method}")
        x = np.arange(len(methods))
        scores = [cases[method]["score"] for method in methods]
        bars = ax.bar(x, scores, color=[COLORS[method] for method in methods], width=0.55)
        for bar, method in zip(bars, methods):
            if not cases[method]["feasible"]:
                bar.set_hatch("///")
                bar.set_edgecolor("#28313D")
                bar.set_linewidth(0.7)
        for i, method in enumerate(methods[:-1]):
            baseline, pm = cases[method]["score"], reference["score"]
            bracket_y = (1.30 - 0.10 * i) if group == "expert-driven methods" else (1.19 - 0.10 * i)
            end_x = x[-1] - 0.18 + 0.18 * i
            path = MplPath([(i, baseline + 0.005), (i, bracket_y),
                            (end_x, bracket_y), (end_x, pm + 0.005)])
            ax.add_patch(FancyArrowPatch(path=path, arrowstyle="->", mutation_scale=8,
                                         color="#28313D", linewidth=1.0,
                                         shrinkA=0, shrinkB=0, zorder=5))
            uplift = (pm / baseline - 1.0) * 100.0
            ax.text((i + end_x) / 2, bracket_y + 0.014, f"+{uplift:.1f}%",
                    ha="center", va="bottom", fontsize=7, color="#28313D")
        top_y = 1.43 if group == "expert-driven methods" else 1.32
        ax.set_ylim(0, top_y)
        ax.set_xlim(-0.55, len(methods) - 0.45)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(f"{stage} | {reference['n_waps']} WAPs", fontsize=10)
        ax.set_xticks(x, methods, fontsize=8)
        for i, value in enumerate(scores):
            if methods[i] == "PM":
                ax.text(i, value - 0.045, f"{value:.3f}", ha="center", va="top",
                        fontsize=7, color="white")
            else:
                label_shift = 0.27 if group == "expert-driven methods" else 0.14
                ax.text(i - label_shift, value + 0.012, f"{value:.3f}",
                        ha="center", va="bottom", fontsize=7)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis="both", direction="out", length=3, width=0.8)
        # The zero tick and zero gridline would protrude past the shared origin.
        zero_tick = ax.yaxis.get_major_ticks()[0]
        zero_tick.tick1line.set_visible(False)
        zero_tick.tick2line.set_visible(False)
        for grid_y in (0.5, 1.0):
            ax.axhline(grid_y, color="#B8BDC4", alpha=.2, linewidth=.8, zorder=0)
        # Draw both axes from the same origin; no spine or grid stroke extends past it.
        origin_x, right_x = -0.55, len(methods) - 0.45
        for start, end in (((origin_x, 0), (right_x, 0)),
                           ((origin_x, 0), (origin_x, top_y))):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                         mutation_scale=9, color="black",
                                         linewidth=0.85, shrinkA=0, shrinkB=0,
                                         clip_on=False, zorder=6))
    axes[0].set_ylabel("Values after normalization", fontsize=8)
    fig.suptitle(f"PM versus {group}", fontsize=11)
    fig.tight_layout(rect=(0, .04, 1, .90), w_pad=1.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    save_both(fig, output, dpi)
    plt.close(fig)
    print(f"saved {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "plots/uplift")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    for group in GROUPS:
        plot_group(group, args.out_dir / f"uplift_{OUTPUT_NAMES[group]}.png", args.dpi)


if __name__ == "__main__":
    main()
