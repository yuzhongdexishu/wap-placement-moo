"""Grouped violin plots of finite RF values on the shared 600-point grid.

Quality failure counts are reported separately so an invalid baseline is not
made to appear feasible by filtering missing RF values from a violin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from plot_fig_common import TARGET_WIDTH_CM, cm_to_in, save_both, read_current_result

V17_ROOT = Path(__file__).resolve().parent

STAGES = [
    ("frame", "Frame construction"),
    ("enclosure", "Enclosure construction"),
]

# one figure per comparison group
GROUPS = {
    "expert": {
        "title": "Expert group: PM vs EbM1/EbM2/EbM3",
        "methods": ["EbM1", "EbM2", "EbM3", "PM"],
    },
    "geometric": {
        "title": "Geometric group: PM vs GcM1/GcM2",
        "methods": ["GcM1", "GcM2", "PM"],
    },
}

METRICS = [
    ("rsrp", "rsrp_dbm", "receiver_rsrp.json", "RSRP", "dBm"),
    ("rsrq", "rsrq_db", "receiver_rsrq.json", "RSRQ", "dB"),
    ("sinr", "sinr_db", "receiver_sinr.json", "SINR", "dB"),
]

# validated categorical palette: one colour per METRIC (methods sit on the x-axis)
METRIC_COLORS = {"RSRP": "#4A7FB8", "RSRQ": "#C79A3B", "SINR": "#4FA37C"}

# one axis per metric: left = RSRP, right inner = RSRQ, right outer = SINR
AXIS_METRICS = {"left": ("rsrp",), "right_inner": ("rsrq",), "right_outer": ("sinr",)}
AXIS_LABELS = {"left": "RSRP (dBm)", "right_inner": "RSRQ (dB)", "right_outer": "SINR (dB)"}

INK = "#2B3440"
MED_COLOR = "#7A1E2E"

# triplet layout: method centres spaced METHOD_STEP apart, metric violins at +-OFFSET;
# the three violins of one method sit tight together (compact block), and the
# inter-method gap is large so each method's block reads as one group
METHOD_STEP = 2.6
OFFSET = 0.45
VIOLIN_WIDTH = 0.44
BOX_HALF = 0.09


def read_json(path: Path) -> dict:
    if path.name.startswith("receiver_"):
        return read_current_result(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_case(base: Path) -> dict:
    """V17-format results of one case -> {metric_key: {vals, ids}} (None -> NaN)."""
    out: dict[str, dict] = {}
    for key, field, fname, _label, _unit in METRICS:
        rows = read_json(base / fname)["receivers"]
        out[key] = {
            "vals": np.asarray(
                [r[field] if r[field] is not None else np.nan for r in rows], dtype=float
            ),
            "ids": [str(r["receiver_id"]) for r in rows],
        }
    return out


def case_base(stage: str, method: str) -> Path:
    if method == "PM":
        return V17_ROOT / "result" / stage / "process/baselines"
    return V17_ROOT / "result" / stage / method / "final_results"


def run_checks(stages: list[tuple[str, str]], methods: list[str]) -> tuple[list[str], set[tuple[str, str]]]:
    """Verify file presence / 600 receivers / same receiver grid as PM.

    Returns (report lines, missing cases) - missing cases are skipped when
    plotting and never fabricated.
    """
    report: list[str] = []
    missing: set[tuple[str, str]] = set()
    for stage, _label in stages:
        pm_base = case_base(stage, "PM")
        for _key, _field, fname, _lbl, _unit in METRICS:
            path = pm_base / fname
            if not path.is_file():
                raise FileNotFoundError(f"Missing PM file: {path}")
            if len(read_json(path)["receivers"]) != 600:
                raise ValueError(f"{path}: expected 600 receivers")
        pm_data = load_case(pm_base)
        pm_ids = pm_data["rsrp"]["ids"]
        report.append(f"[PM {stage}] 600 receivers ok")
        for method in methods:
            if method == "PM":
                continue
            base = case_base(stage, method)
            files_missing = [f for _, _, f, _, _ in METRICS if not (base / f).is_file()]
            if files_missing:
                report.append(f"[missing {stage}/{method}] {', '.join(files_missing)} - "
                              f"method skipped, no data invented")
                missing.add((stage, method))
                continue
            data = load_case(base)
            if len(data["rsrp"]["vals"]) != 600:
                raise ValueError(f"{stage}/{method}: expected 600 receivers")
            if data["rsrp"]["ids"] != pm_ids:
                raise ValueError(f"{stage}/{method}: receiver grid differs from PM")
            report.append(f"[case {stage}/{method}] 600 receivers on the same grid as PM (pointwise paired)")
    return report, missing


def violin_block(ax, datasets: dict[str, np.ndarray], metrics: list[tuple[str, str, str, str, str]],
                 center: float, rng: np.random.Generator) -> None:
    """Draw the three metric violins (body + box + med + scatter) around one method tick."""
    positions = {"rsrp": center - OFFSET, "rsrq": center, "sinr": center + OFFSET}
    for key, _field, _fname, label, _unit in metrics:
        pos = positions[key]
        arr = datasets[key]
        color = METRIC_COLORS[label]

        body = ax.violinplot(
            [arr], positions=[pos], widths=VIOLIN_WIDTH, showextrema=False,
        )["bodies"][0]
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor(color)
        body.set_linewidth(0.9)
        body.set_zorder(2)

        q1, med, q3 = np.percentile(arr, [25.0, 50.0, 75.0])
        iqr = max(float(q3 - q1), 1e-9)
        lo = float(max(arr.min(), q1 - 1.5 * iqr))
        hi = float(min(arr.max(), q3 + 1.5 * iqr))

        # box + whiskers + caps (no median line: removed per user request)
        ax.add_patch(Rectangle((pos - BOX_HALF, q1), 2 * BOX_HALF, float(q3 - q1),
                               facecolor="white", edgecolor=INK, lw=1.0, zorder=4))
        ax.plot([pos, pos], [lo, q1], color=INK, lw=1.0, zorder=4)
        ax.plot([pos, pos], [q3, hi], color=INK, lw=1.0, zorder=4)
        ax.plot([pos - 0.06, pos + 0.06], [lo, lo], color=INK, lw=1.0, zorder=4)
        ax.plot([pos - 0.06, pos + 0.06], [hi, hi], color=INK, lw=1.0, zorder=4)

        # small transparent jittered receiver sample
        sample = rng.choice(arr, size=min(100, len(arr)), replace=False)
        jitter = rng.uniform(-0.10, 0.10, size=len(sample))
        ax.scatter(pos + jitter, sample, s=9, alpha=0.14, color=color, linewidths=0, zorder=3)

        # Each median is labelled beside its own violin. The three labels in a
        # method group use different directions so their text boxes stay apart.
        dx, dy, align = {
            "rsrp": (-9, 3, "right"),
            "rsrq": (0, -15, "center"),
            "sinr": (9, 3, "left"),
        }[key]
        ax.annotate(
            f"{med:.1f}", xy=(pos, med), xytext=(dx, dy),
            textcoords="offset points", ha=align, va="center",
            fontsize=6.5, color=INK, zorder=9,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white",
                  "edgecolor": "none", "alpha": 0.94},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped violin distributions (PM vs baselines).")
    parser.add_argument("--out-dir", type=Path, default=V17_ROOT / "plots" / "violin")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--groups", type=str, default="expert,geometric")
    parser.add_argument("--stages", type=str, default="frame,enclosure")
    args = parser.parse_args()

    stages = [s for s in STAGES if s[0] in args.stages.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    for group_key in args.groups.split(","):
        group = GROUPS[group_key]
        methods = group["methods"]
        for stage, stage_label in stages:
            report, missing = run_checks([(stage, stage_label)], methods)
            print("\n".join(report))
            present = [m for m in methods if (stage, m) not in missing]
            if len(present) < 2:
                print(f"[skip] {group_key}/{stage}: fewer than 2 cases present - figure skipped")
                continue

            fig, ax = plt.subplots(figsize=(cm_to_in(TARGET_WIDTH_CM), 3.1))
            ax2 = ax.twinx()   # RSRQ
            ax3 = ax.twinx()   # SINR, spine offset well outside RSRQ so the two axes never overlap
            ax3.spines["right"].set_position(("outward", 60))

            datasets_all: dict[str, dict[str, np.ndarray]] = {}
            valid_counts: dict[str, int] = {}
            failed_counts: dict[str, int] = {}
            for method in present:
                data = load_case(case_base(stage, method))
                datasets_all[method] = {}
                for key, _f, _fn, _lbl, _unit in METRICS:
                    vals = data[key]["vals"]
                    datasets_all[method][key] = vals[np.isfinite(vals)]
                valid_counts[method] = int(np.isfinite(data["rsrp"]["vals"]).sum())
                failed_counts[method] = int(read_json(case_base(stage, method) /
                    "receiver_composite.json")["coverage"]["failed_quality"])

            rng = np.random.default_rng(7)
            # one axis per metric, each with its own scale (all three stretched)
            axes_of = {"left": ax, "right_inner": ax2, "right_outer": ax3}
            for i, method in enumerate(present):
                center = i * METHOD_STEP
                for axis_key, keys in AXIS_METRICS.items():
                    violin_block(
                        axes_of[axis_key],
                        {m: datasets_all[method][m] for m in keys},
                        [m for m in METRICS if m[0] in keys], center, rng,
                    )

            # y ranges from the real data of each metric, padded for labels/whiskers
            def _limits(vals: np.ndarray) -> tuple[float, float]:
                lo, hi = float(vals.min()), float(vals.max())
                pad = max((hi - lo) * 0.14, 0.5)
                return lo - pad * 0.3, hi + pad
            for axis_key, keys in AXIS_METRICS.items():
                vals = np.concatenate([datasets_all[m][k] for m in present for k in keys])
                axes_of[axis_key].set_ylim(*_limits(vals))

            centers = np.arange(len(present)) * METHOD_STEP
            ax.set_xticks(centers)
            ax.set_xticklabels([m for m in present],
                               fontsize=10.5)
            for tick, m in zip(ax.get_xticklabels(), present):
                if m == "PM":
                    tick.set_fontweight("bold")
                    tick.set_color(MED_COLOR)
            ax.tick_params(axis="x", length=0)

            # axis colours follow the metric colours (scale-to-metric mapping)
            ax.set_ylabel(AXIS_LABELS["left"], fontsize=9.5, color=METRIC_COLORS["RSRP"])
            ax2.set_ylabel(AXIS_LABELS["right_inner"], fontsize=9.5, color=METRIC_COLORS["RSRQ"])
            ax3.set_ylabel(AXIS_LABELS["right_outer"], fontsize=9.5, color=METRIC_COLORS["SINR"])
            ax.tick_params(axis="y", labelsize=9, colors=METRIC_COLORS["RSRP"])
            ax2.tick_params(axis="y", labelsize=9, pad=2, colors=METRIC_COLORS["RSRQ"])
            ax3.tick_params(axis="y", labelsize=9, pad=2, colors=METRIC_COLORS["SINR"])
            ax.grid(axis="y", color="#D8DDE3", linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#364152")
            ax.spines["bottom"].set_color("#364152")
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax2.spines["bottom"].set_visible(False)
            ax2.spines["right"].set_color(METRIC_COLORS["RSRQ"])
            ax2.grid(False)
            ax3.spines["top"].set_visible(False)
            ax3.spines["left"].set_visible(False)
            ax3.spines["bottom"].set_visible(False)
            ax3.spines["right"].set_color(METRIC_COLORS["SINR"])
            ax3.grid(False)

            # metric colour legend
            handles = [Patch(facecolor=METRIC_COLORS[label], edgecolor=METRIC_COLORS[label],
                             alpha=0.45, label=label) for _k, _f, _fn, label, _u in METRICS]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03),
                      ncol=3, frameon=False, fontsize=9.5, columnspacing=1.4)

            counts_line = " | ".join(f"{m}: {n}" for m, n in failed_counts.items())
            note = "RF quality failures / 600: " + counts_line
            fig.text(0.5, 0.012, note, ha="center", va="bottom", fontsize=7.5,
                     color="#5A6572", style="italic")

            # no title text: the figures are unlabelled by design
            fig.tight_layout(rect=(0, 0.03, 1, 0.97))
            # tight_layout ignores the twinx label stacks on the right; reserve
            # the space explicitly (incl. the 60 pt outer SINR spine) so the
            # figure stays inside the 16 cm width
            fig.subplots_adjust(left=0.16, right=0.64)
            path = out_dir / f"violin_{group_key}_{stage}.png"
            save_both(fig, path, args.dpi)
            plt.close(fig)
            produced.extend([path, path.with_name(f"{path.stem}_notext.png")])
            print(f"saved {path} + notext  valid receivers: {valid_counts}")

    print(f"\nDone: {len(produced)} violin figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
