"""Six-model formal task completion chart for the active benchmark scope."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_fig_common import TARGET_WIDTH_CM, cm_to_in, save_both
from send_task2 import (RESULTS_ROOT, all_task_results, build_task_matrix,
                        load_models, result_matches_current_profile)

ROOT = Path(__file__).resolve().parent
COLORS = ("#327DCD", "#E76736", "#18A884", "#F0A500", "#DD6D9A", "#138A19")


def completion_rows() -> tuple[list[dict], int]:
    from send_task2 import _active_scope_ids
    active = _active_scope_ids()
    matrix = build_task_matrix(repetitions=1)
    if active is not None:
        matrix = [task for task in matrix if task["task_id"] in active]
    by_model: dict[str, set[str]] = {}
    for task in matrix:
        by_model.setdefault(task["model_key"], set()).add(task["task_id"])
    records = {str(row.get("task_id")): row for row in all_task_results()
               if result_matches_current_profile(row)}
    rows = []
    for model in load_models():
        assigned = by_model.get(model["key"], set())
        attempted = sum(task_id in records for task_id in assigned)
        successful = sum(records.get(task_id, {}).get("success") is True for task_id in assigned)
        rows.append({"key": model["key"],
                     "name": model.get("display_name") or model["key"],
                     "assigned": len(assigned), "attempted": attempted,
                     "successful": successful,
                     "rate": 100.0 * successful / len(assigned) if assigned else 0.0})
    return rows, len(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "plots/six_llm/completion_rate.png")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    rows, expected = completion_rows()
    successful = sum(row["successful"] for row in rows)
    fig, ax = plt.subplots(figsize=(cm_to_in(TARGET_WIDTH_CM), 3.7))
    y = np.arange(len(rows))
    ax.barh(y, [100.0] * len(rows), color="#E9EDF1", height=.62, zorder=1)
    ax.barh(y, [row["rate"] for row in rows], color=COLORS,
            height=.62, zorder=2)
    ax.set_yticks(y, [row["name"] for row in rows], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#DCE2E8", linewidth=.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#B5BEC6")
    for i, row in enumerate(rows):
        ax.text(102, i, f"{row['rate']:.1f}%", va="center", fontsize=16,
                fontweight="bold", color="#202A33")
    fig.suptitle("Six-LLM Completion Rate", x=.5, y=.97, ha="center",
                 fontsize=21, fontweight="bold")
    fig.subplots_adjust(left=.34, right=.98, top=.84, bottom=.05)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    save_both(fig, args.output, args.dpi)
    plt.close(fig)
    print(f"saved {args.output} ({successful}/{expected} successes)")


if __name__ == "__main__":
    main()
