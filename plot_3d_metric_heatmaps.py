"""3D heatmaps and WAP placement from current 600-point surrogate results.

PM and expert/geometry baselines must use the current region hash and identical
receiver positions. Background IFC mesh is sampled for drawing speed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import triangulate, unary_union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from plot_fig_common import TARGET_WIDTH_CM, cm_to_in, save_both, read_current_result
from tools.wap_layout_optimization.define_problem import read_json as read_problem_json, region_data, scenario_key

V17_ROOT = Path(__file__).resolve().parent

# (stage dir, human label)
STAGES = [
    ("frame", "Frame construction"),
    ("enclosure", "Enclosure construction"),
]

METHODS_IN_ORDER = ["PM", "EbM1", "EbM2", "EbM3", "GcM1", "GcM2"]

# (metric key, json field, file name, display label, unit)
METRICS = [
    ("rsrp", "rsrp_dbm", "receiver_rsrp.json", "RSRP", "dBm"),
    ("rsrq", "rsrq_db", "receiver_rsrq.json", "RSRQ", "dB"),
    ("sinr", "sinr_db", "receiver_sinr.json", "SINR", "dB"),
]

# RF colormap used by the paper figures.
RF_CMAP = LinearSegmentedColormap.from_list(
    "rf_reference",
    ["#2856F6", "#00BDF2", "#00E49A", "#A8EC00", "#FFE600", "#FF9500", "#FF2B16", "#FF00C8"],
    N=256,
)

# Gamma-composed twin of RF_CMAP for the colourbar: CBAR_CMAP(f) = RF_CMAP(f**1.5).
# The surfaces colour value v as RF_CMAP(norm(v)) with PowerNorm(gamma=1.5), i.e.
# a value at data fraction f gets colour RF_CMAP(f**1.5). Rendering the colourbar
# from this composed cmap with a plain Normalize reproduces that mapping exactly
# (matplotlib's own colourbar gradient mesh does not reflect PowerNorm's gamma).
CBAR_CMAP = LinearSegmentedColormap.from_list(
    "rf_gamma15", RF_CMAP(np.linspace(0.0, 1.0, 256) ** 1.5), N=256,
)

WAP_COLOR = "#FF2922"
WAP_EDGE = "#101010"
MESH_COLOR = "#B7BEC7"
BOUNDS_COLOR = (0.30, 0.36, 0.44, 0.85)


def read_json(path: Path) -> dict:
    if path.name.startswith("receiver_"):
        return read_current_result(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse one ASCII PLY file -> (vertices (n,3) float, faces (m,3) int)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    n_vertex = n_face = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith("element vertex"):
            n_vertex = int(line.split()[2])
        elif line.startswith("element face"):
            n_face = int(line.split()[2])
        elif line.strip() == "end_header":
            header_end = i
            break
    if n_vertex == 0 or n_face == 0 or header_end == 0:
        raise ValueError(f"Unsupported PLY layout: {path}")
    body = lines[header_end + 1:]
    vertices = np.asarray(
        [[float(t) for t in ln.split()] for ln in body[:n_vertex]], dtype=float
    )
    faces = np.asarray(
        [[int(t) for t in ln.split()[1:]] for ln in body[n_vertex : n_vertex + n_face]],
        dtype=int,
    )
    return vertices, faces


_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def load_mesh(stage: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all PLY meshes of a stage -> (triangles (T,3,3), lo (3,), hi (3,))."""
    if stage in _MESH_CACHE:
        return _MESH_CACHE[stage]
    mesh_dir = V17_ROOT / "result" / stage / "workspace" / "bim" / "meshes"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory missing: {mesh_dir}")
    triangles: list[np.ndarray] = []
    for ply in sorted(mesh_dir.glob("*.ply")):
        verts, faces = parse_ply(ply)
        if len(faces):
            triangles.append(verts[faces])
    tris = np.concatenate(triangles, axis=0) if triangles else np.zeros((0, 3, 3), dtype=float)
    if len(tris) == 0:
        raise ValueError(f"No triangles found in {mesh_dir}")
    lo = tris.min(axis=(0, 1))
    hi = tris.max(axis=(0, 1))
    if len(tris) > 4000:
        tris = tris[np.linspace(0, len(tris)-1, 4000, dtype=int)]
    _MESH_CACHE[stage] = (tris, lo, hi)
    return _MESH_CACHE[stage]


def load_case(base: Path) -> tuple[dict, np.ndarray, list[str]]:
    """V17-format results of one case -> ({metric: {pos, vals}}, waps, receiver_ids).

    Works for PM (base = result/<stage>/final_results) and for every baseline
    method (base = result/<stage>/<METHOD>/final_results) - same JSON schema.
    """
    out: dict[str, dict] = {}
    ids: list[str] = []
    for key, field, fname, _label, _unit in METRICS:
        data = read_json(base / fname)
        rows = data["receivers"]
        pos = np.asarray([r["position"] for r in rows], dtype=float)
        vals = np.asarray([r[field] if r[field] is not None else np.nan for r in rows], dtype=float)
        out[key] = {"pos": pos, "vals": vals}
        if not ids:
            ids = [str(r["receiver_id"]) for r in rows]
    waps = np.asarray(read_json(base / "sensor_positions.json")["sensor_positions"], dtype=float)
    return out, waps, ids


def case_base(stage: str, method: str) -> Path:
    if method == "PM":
        return V17_ROOT / "result" / stage / "process/baselines"
    return V17_ROOT / "result" / stage / method / "final_results"


def _box_edges(lo: np.ndarray, hi: np.ndarray) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    ids = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [(corners[a], corners[b]) for a, b in ids]


@lru_cache(maxsize=24)
def floor_domain(stage: str, scenario: str, floor_name: str):
    floor = region_data()["scenarios"][scenario_key(stage, scenario)]["floors"][floor_name]
    domain = unary_union([Polygon(p["outer"], p.get("holes", []))
                          for p in floor["valid_floor_domain"]])
    receiver_z = (float(floor["z_min_m"]) + float(floor["z_max_m"])) / 2.0
    heights = region_data()["parameters"]["receiver_height_above_surface_m"]
    floor_z = receiver_z - (float(heights[0]) + float(heights[1])) / 2.0
    triangles = [triangle for triangle in triangulate(domain)
                 if domain.covers(triangle.representative_point())]
    faces = [[(x, y, floor_z - 0.03) for x, y in list(triangle.exterior.coords)[:3]]
             for triangle in triangles]
    return domain, receiver_z, floor_z, faces


def draw_floor_base(ax, domain, floor_z: float, floor_name: str, faces: list) -> None:
    """Draw the exact valid floor footprint below its RF surface."""
    if faces:
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors="#CED4D9", edgecolors="none",
            alpha=0.62, shade=False, zorder=2.5))
    polygons = domain.geoms if hasattr(domain, "geoms") else [domain]
    for polygon in polygons:
        xy = np.asarray(polygon.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], np.full(len(xy), floor_z + 0.06),
                color=(0.22, 0.28, 0.34, 0.85), linewidth=1.0, zorder=4)
    xmin, ymin, xmax, ymax = domain.bounds
    floor_label_y = {"1F": 0.31, "2F": 0.48, "3F": 0.65}[floor_name]
    ax.text2D(0.02, floor_label_y, floor_name, transform=ax.transAxes,
            fontsize=14.0, color="#26313A", zorder=30,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8})


def layer_surface(ax, pts: np.ndarray, vals: np.ndarray,
                  norm: PowerNorm, domain, z: float) -> None:
    """Interpolate receivers only inside their matching valid floor footprint."""
    xs, ys = pts[:, 0], pts[:, 1]
    n = len(vals)
    xmin, ymin, xmax, ymax = domain.bounds
    nx = max(30, min(70, int(math.ceil((xmax - xmin) / 0.75))))
    ny = max(25, min(60, int(math.ceil((ymax - ymin) / 0.75))))
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(gx, gy)
    dist2 = (X[..., None] - xs[None, None, :]) ** 2 + (Y[..., None] - ys[None, None, :]) ** 2
    flat = dist2.reshape(-1, n)
    k = 6
    neighbours = np.argpartition(flat, k - 1, axis=1)[:, :k]
    rows = np.arange(flat.shape[0])
    near = flat[rows[:, None], neighbours]
    weights = 1.0 / np.maximum(near, 0.35**2)
    V = (np.sum(weights * vals[neighbours], axis=1) / np.sum(weights, axis=1)).reshape(X.shape)
    min_d2 = flat.min(axis=1).reshape(X.shape)
    valid_area = shapely.contains_xy(domain, X, Y) & (min_d2 <= 3.0**2)
    face = RF_CMAP(norm(V))
    face[..., 3] = 0.76
    ax.plot_surface(
        X, Y, np.where(valid_area, z, np.nan), rstride=1, cstride=1, facecolors=face,
        linewidth=0.0, antialiased=False, shade=False, alpha=0.76, zorder=3,
    )


def plot_layout_map(
    stage: str,
    method: str,
    waps: np.ndarray,
    tris: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    """Building base map + sensor (WAP) placement ONLY — no metric surfaces.

    Draws the real PLY mesh as a slightly more opaque base map, the outer
    bounds outline, the WAP markers with small per-WAP index labels, and a
    legend. No receivers, no interpolation surfaces, no colourbar. Exports
    the labelled PNG plus its ``*_notext`` twin (same convention as the
    heatmaps). Returns the two output paths.
    """
    fig = plt.figure(figsize=(cm_to_in(TARGET_WIDTH_CM), 4.05))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    # real building base map (a touch more opaque than in the heatmaps so the
    # shell reads clearly without any coloured surface on top)
    ax.add_collection3d(Poly3DCollection(
        tris, facecolors=MESH_COLOR, edgecolors=(0.55, 0.59, 0.64, 0.35),
        linewidths=0.25, alpha=0.5, shade=False, zorder=1,
    ))
    ax.add_collection3d(Line3DCollection(
        _box_edges(lo, hi), colors=[BOUNDS_COLOR], linewidths=1.1, zorder=2,
    ))

    # WAP markers
    ax.scatter(
        waps[:, 0], waps[:, 1], waps[:, 2],
        s=92, marker="o", color=WAP_COLOR, edgecolors=WAP_EDGE,
        linewidths=0.9, depthshade=False, zorder=20,
    )
    # small per-WAP index labels so the placement is unambiguous
    for i, w in enumerate(waps, start=1):
        ax.text(
            float(w[0]), float(w[1]), float(w[2]), f" {i}",
            color="#20242A", fontsize=13.0, zorder=21,
        )

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    size = np.maximum(hi - lo, 1e-6)
    ax.set_box_aspect((size[0], size[1], size[2] * 1.15))
    ax.view_init(elev=38, azim=-55)
    ax.set_axis_off()
    ax.set_title(f"Sensor placement · {stage} · {method}",
                 fontsize=18.5, fontweight="bold", pad=8)

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=WAP_COLOR,
                      markeredgecolor=WAP_EDGE, markersize=9,
                      label=f"WAP (n={len(waps)})")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, -0.02),
              frameon=False, fontsize=15.5)
    fig.subplots_adjust(left=0.01, right=0.95, bottom=0.02, top=0.92)

    path = out_dir / f"layout_{stage}_{method}.png"
    save_both(fig, path, dpi)
    plt.close(fig)
    return [path, path.with_name(f"{path.stem}_notext.png")]


def run_checks(stages: list[tuple[str, str]], methods: list[str]) -> tuple[list[str], set[tuple[str, str]]]:
    """Pre-plot integrity checks. Returns (report lines, missing cases).

    Every case must ship the 600-receiver / 12-WAP V17 JSONs; baselines must
    sit on the *same* receiver grid as PM (pointwise-paired comparison).
    Missing cases are reported and skipped - never fabricated.
    """
    report: list[str] = []
    missing: set[tuple[str, str]] = set()

    for stage, _label in stages:
        # ---- V17 PM reference files ----
        pm_base = V17_ROOT / "result" / stage / "final_results"
        for _key, _field, fname, _label, _unit in METRICS + [("composite", "composite_score", "receiver_composite.json", "", "")]:
            path = pm_base / fname
            if not path.is_file():
                raise FileNotFoundError(f"Missing V17 PM file: {path}")
            rows = read_json(path)["receivers"]
            if len(rows) != 600:
                raise ValueError(f"{path}: expected 600 receivers, found {len(rows)}")
        sp = read_json(pm_base / "sensor_positions.json")["sensor_positions"]
        if not sp:
            raise ValueError(f"{pm_base / 'sensor_positions.json'}: no WAPs")
        scene_xml = V17_ROOT / "result" / stage / "workspace" / "bim" / f"{stage}.xml"
        mesh_dir = V17_ROOT / "result" / stage / "workspace" / "bim" / "meshes"
        if not scene_xml.is_file():
            raise FileNotFoundError(f"Missing scene XML: {scene_xml}")
        n_ply = len(list(mesh_dir.glob("*.ply"))) if mesh_dir.is_dir() else 0
        if n_ply == 0:
            raise FileNotFoundError(f"No PLY meshes in {mesh_dir}")
        state = read_problem_json(V17_ROOT / "result" / stage / "workspace/bim/pipeline_state.json")
        problem = read_problem_json(state["paths"]["wap_optimization"]["problem"])
        scenario = problem["optimization_problem"]["priority_scenario"]
        slab_tops = [float(parse_ply(path)[0][:, 2].max()) for path in mesh_dir.glob("slab_*.ply")]
        for floor_name in ("1F", "2F", "3F"):
            _domain, _receiver_z, floor_z, _faces = floor_domain(stage, scenario, floor_name)
            if not slab_tops or min(abs(top - floor_z) for top in slab_tops) > 0.05:
                raise ValueError(f"{stage}/{floor_name}: no IFC slab top at z={floor_z:.3f} m")
        report.append(f"[PM {stage}] 600 receivers, {len(sp)} WAPs, scene xml ok, {n_ply} PLY meshes ok")

        pm_metrics, _waps, pm_ids = load_case(pm_base)
        pm_pos = pm_metrics["rsrp"]["pos"]

        # ---- baseline cases (same V17 audit chain -> same receiver grid) ----
        for method in methods:
            if method == "PM":
                continue
            base = case_base(stage, method)
            files_missing = [f for _, _, f, _, _ in METRICS if not (base / f).is_file()]
            if not (base / "sensor_positions.json").is_file():
                files_missing.append("sensor_positions.json")
            if files_missing:
                report.append(f"[missing {stage}/{method}] {', '.join(files_missing)} - "
                              f"case skipped, no data invented")
                missing.add((stage, method))
                continue
            metrics, waps, ids = load_case(base)
            n_rows = len(metrics["rsrp"]["vals"])
            if n_rows != 600:
                raise ValueError(f"{stage}/{method}: expected 600 receivers, found {n_rows}")
            if len(waps) != len(sp):
                raise ValueError(f"{stage}/{method}: expected {len(sp)} WAPs, found {len(waps)}")
            same_ids = ids == pm_ids
            same_pos = np.allclose(metrics["rsrp"]["pos"], pm_pos, atol=1e-6)
            if not (same_ids and same_pos):
                raise ValueError(f"{stage}/{method}: receiver grid differs from PM")
            report.append(f"[case {stage}/{method}] 600 receivers on the same grid as PM "
                          f"(pointwise paired), {len(waps)} WAPs ok")
    report.append("CHECKS PASSED" if not missing else "CHECKS PASSED (missing cases reported above)")
    return report, missing


def main() -> int:
    parser = argparse.ArgumentParser(description="3D RF metric heatmaps for every experiment.")
    parser.add_argument("--out-dir", type=Path, default=V17_ROOT / "plots" / "3d_heatmaps")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--stages", type=str, default="frame,enclosure",
                        help="comma separated: frame,enclosure")
    parser.add_argument("--methods", type=str, default="PM,EbM1,EbM2,EbM3,GcM1,GcM2",
                        help="comma separated methods to plot")
    args = parser.parse_args()

    stages = [s for s in STAGES if s[0] in args.stages.split(",")]
    methods = [m for m in METHODS_IN_ORDER if m in args.methods.split(",")]

    # ---- pre-plot checks ----
    report, missing = run_checks(stages, methods)
    print("\n".join(report))
    (V17_ROOT / "plots").mkdir(exist_ok=True)
    (V17_ROOT / "plots" / "consistency_report.md").write_text(
        "# Data consistency report\n\n```\n" + "\n".join(report) + "\n```\n", encoding="utf-8"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # colour bounds per experiment (0.1/99.9 percentile of its own receivers,
    # same recipe as the paper per-case heatmaps); pooled fallback if degenerate
    pooled: dict[tuple[str, str], tuple[float, float]] = {}
    for stage, _label in stages:
        for key, _field, _fname, _lbl, _unit in METRICS:
            chunks = []
            for method in methods:
                if (stage, method) in missing:
                    continue
                chunks.append(load_case(case_base(stage, method))[0][key]["vals"])
            all_vals = np.concatenate(chunks)
            all_vals = all_vals[np.isfinite(all_vals)]
            pooled[(stage, key)] = (
                float(np.nanpercentile(all_vals, 0.5)),
                float(np.nanpercentile(all_vals, 99.5)),
            )

    produced: list[Path] = []
    for stage, _stage_label in stages:
        tris, lo, hi = load_mesh(stage)
        state = read_problem_json(V17_ROOT / "result" / stage / "workspace/bim/pipeline_state.json")
        problem = read_problem_json(state["paths"]["wap_optimization"]["problem"])
        scenario = problem["optimization_problem"]["priority_scenario"]
        for method in methods:
            if (stage, method) in missing:
                continue
            metrics, waps, _ids = load_case(case_base(stage, method))
            for key, _field, _fname, label, unit in METRICS:
                vals = metrics[key]["vals"]
                vals_f = vals[np.isfinite(vals)]
                c_lo = float(np.nanpercentile(vals_f, 0.1))
                c_hi = float(np.nanpercentile(vals_f, 99.9))
                if not (np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo):
                    c_lo, c_hi = pooled[(stage, key)]
                norm = PowerNorm(gamma=1.5, vmin=c_lo, vmax=c_hi, clip=True)
                fig = plt.figure(figsize=(cm_to_in(TARGET_WIDTH_CM), 4.05))
                ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

                # IFC model supplies the structural context; floor-aligned RF
                # planes remain visible through the semi-transparent geometry.
                ax.add_collection3d(Poly3DCollection(
                    tris, facecolors="#B8C0C7", edgecolors="none",
                    alpha=0.32, shade=False, zorder=1,
                ))

                pos = metrics[key]["pos"]
                valid = np.isfinite(vals)

                # one surface per horizontal z-layer (layers with too few points -> scatter)
                for floor_name in ("1F", "2F", "3F"):
                    domain, receiver_z, floor_z, faces = floor_domain(stage, scenario, floor_name)
                    if not np.any(np.isclose(pos[:, 2], receiver_z, atol=1e-3)):
                        raise ValueError(f"{stage}/{floor_name}: RF receiver height differs from floor base")
                    draw_floor_base(ax, domain, floor_z, floor_name, faces)
                    idx = np.where(np.isclose(pos[:, 2], receiver_z, atol=1e-3) & valid)[0]
                    if idx.size == 0:
                        continue
                    if idx.size >= 8:
                        layer_surface(ax, pos[idx], vals[idx], norm, domain, floor_z + 0.04)
                    else:
                        ax.scatter(
                            pos[idx, 0], pos[idx, 1], np.full(idx.size, floor_z + 0.04), c=vals[idx],
                            cmap=RF_CMAP, norm=norm, s=22, linewidths=0.3,
                            edgecolors="#222222", depthshade=False, zorder=3,
                        )

                # WAP markers
                ax.scatter(
                    waps[:, 0], waps[:, 1], waps[:, 2],
                    s=82, marker="o", color=WAP_COLOR, edgecolors=WAP_EDGE,
                    linewidths=0.85, depthshade=False, zorder=20,
                )

                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                size = np.maximum(hi - lo, 1e-6)
                ax.set_box_aspect((size[0], size[1], size[2] * 1.15))
                ax.view_init(elev=38, azim=-55)
                ax.set_axis_off()
                ax.set_title(f"{label} of {stage} · {method}", fontsize=18.5, fontweight="bold", pad=8)

                # colourbar: gamma-composed cmap + Normalize so the bar shows
                # exactly the surface colours; ticks pinned inside the bounds
                sm = plt.cm.ScalarMappable(norm=Normalize(vmin=c_lo, vmax=c_hi), cmap=CBAR_CMAP)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.055, shrink=0.73)
                ticks = np.linspace(c_lo, c_hi, 5)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
                cbar.ax.set_title(f"{label}\n[{unit}]", fontsize=15.0, pad=8)
                cbar.ax.tick_params(labelsize=14.5)

                handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=WAP_COLOR,
                                  markeredgecolor=WAP_EDGE, markersize=9, label=f"WAP (n={len(waps)})")]
                ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, -0.02),
                          frameon=False, fontsize=15.5)
                fig.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.92)

                path = out_dir / f"3d_{stage}_{method}_{key}.png"
                save_both(fig, path, args.dpi)
                plt.close(fig)
                produced.extend([path, path.with_name(f"{path.stem}_notext.png")])
                print(f"saved {path} + notext")

    # ---- base-map + sensor placement maps (one per present case) ----
    for stage, _stage_label in stages:
        tris, lo, hi = load_mesh(stage)
        for method in methods:
            if (stage, method) in missing:
                continue
            sp_path = case_base(stage, method) / "sensor_positions.json"
            if not sp_path.is_file():
                print(f"[layout skip] {stage}/{method}: sensor_positions.json missing")
                continue
            waps = np.asarray(read_json(sp_path)["sensor_positions"], dtype=float)
            layout_paths = plot_layout_map(stage, method, waps, tris, lo, hi,
                                          out_dir, args.dpi)
            produced.extend(layout_paths)
            print(f"saved {layout_paths[0]} + notext")

    if missing:
        print("\nMissing cases (skipped, no data invented):")
        for stage, method in sorted(missing):
            print(f"  - {stage}/{method}")
    print(f"\nDone: {len(produced)} figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
