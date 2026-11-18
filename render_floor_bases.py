"""Render floor cutaways with the exact architecture-image parallel cameras."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv

from plot_registered_floor_heatmaps import CAMERAS

ROOT = Path(__file__).resolve().parent
FLOORS = {"1F": 0.0, "2F": 4.5, "3F": 8.4}
COLOURS = {
    "slab": "#DEE1DF", "wall": "#E2E2DA", "window": "#AFCCD4",
    "plate": "#AFCCD4", "member": "#6D7D83", "railing": "#6D7D83",
    "door": "#B0B4AC", "stair_flight": "#C1C8C9",
}


def render(stage: str, floor_name: str) -> Path:
    floor_z = FLOORS[floor_name]
    lower, upper = floor_z - 0.7, floor_z + 1.25
    plotter = pv.Plotter(off_screen=True, window_size=(3000, 2250), lighting="none")
    plotter.set_background("white")
    plotter.enable_parallel_projection()
    mesh_dir = ROOT / "result" / stage / "workspace/bim/meshes"
    count = 0
    for path in sorted(mesh_dir.glob("*.ply")):
        mesh = pv.read(path)
        if mesh.n_cells == 0 or mesh.bounds[5] < lower or mesh.bounds[4] > upper:
            continue
        mesh = mesh.clip(normal=(0, 0, 1), origin=(0, 0, lower), invert=False)
        mesh = mesh.clip(normal=(0, 0, -1), origin=(0, 0, upper), invert=False)
        if mesh.n_cells == 0:
            continue
        kind = path.stem.split("_")[0]
        plotter.add_mesh(mesh, color=COLOURS.get(kind, "#C6CBCB"),
                         opacity=0.55 if kind in {"window", "plate"} else 1.0,
                         smooth_shading=False, ambient=.42, diffuse=.65,
                         specular=.08, show_edges=False)
        count += 1
    if count == 0:
        raise RuntimeError(f"No {stage}/{floor_name} IFC meshes intersect the cutaway")
    plotter.add_light(pv.Light(position=(-25, -40, 75), focal_point=(0, 0, 4),
                               color="white", intensity=.65))
    plotter.add_light(pv.Light(position=(40, 25, 40), focal_point=(0, 0, 4),
                               color="#E7EDF3", intensity=.35))
    position, target, scale = CAMERAS[stage]
    plotter.camera_position = [position, target, (0, 0, 1)]
    plotter.camera.parallel_scale = scale
    plotter.reset_camera_clipping_range()
    plotter.enable_anti_aliasing("ssaa")
    output = ROOT / "data/visualization_bases" / f"{stage}_{floor_name}_cutaway.png"
    plotter.screenshot(str(output))
    plotter.close()
    print(f"saved {output} ({count} meshes)", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", default="frame,enclosure")
    parser.add_argument("--floors", default="1F,2F,3F")
    args = parser.parse_args()
    for stage in args.stages.split(","):
        for floor_name in args.floors.split(","):
            render(stage, floor_name)


if __name__ == "__main__":
    main()
