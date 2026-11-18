from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

from .parse_ifc_building import SERVER_NAME, load_state, success, write_json

TOOL_NAME = "export_object_meshes"

# G/P/W integration (adapted from the former IFC_XML_PLY source). This tool embeds the PLY export side of the
# G/P/W functionality: meshes are written in the G1 PLYMesh.to_ply_ascii conventions
# (welded shared-index triangles, world coordinates in metres, ascii header) under
# Sionna-native names {label}_{ifc_id}.ply, so the scene XML can reference them with
# face_normals for later wedge/diffraction extraction (W7).
PLY_HEADER = "comment IFC world coordinates; units metre"


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """G1 PLYMesh.to_ply_ascii equivalent over the welded cache arrays."""
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(PLY_HEADER + "\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property double x\nproperty double y\nproperty double z\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for x, y, z in vertices:
            f.write(f"{x:.12g} {y:.12g} {z:.12g}\n")
        for a, b, c in faces:
            f.write(f"3 {int(a)} {int(b)} {int(c)}\n")


def export_object_meshes(state_file: str) -> dict[str, Any]:
    """Input: pipeline_state.json + geometry_cache/*.npz. Output: meshes/*.ply + updated state."""
    state_path = Path(state_file).expanduser().resolve()
    state = load_state(state_path)
    root = Path(state["project_root"])
    meshes_dir = Path(state["paths"]["meshes_dir"])
    expected_meshes = root / "workspace/bim/meshes"
    if meshes_dir.is_symlink() or meshes_dir.resolve() != expected_meshes.resolve():
        raise RuntimeError(f"Unsafe mesh cleanup target: {meshes_dir}")
    meshes_dir.mkdir(parents=True, exist_ok=True)
    for old in meshes_dir.glob("*.ply"):
        old.unlink()

    exported = failed = 0
    state["scene_status"] = "pending"
    for item in state["objects"]:
        if item.get("geometry_status") != "completed":
            continue
        try:
            data = np.load(item["cache_file"])
            mesh_file = meshes_dir / f"{item['mesh_label']}_{item['ifc_id']}.ply"
            _write_ply(mesh_file, data["vertices"], data["faces"])
            item.update({"mesh_status": "completed", "mesh_file": str(mesh_file)})
            item.pop("mesh_error", None)
            exported += 1
        except Exception as exc:
            item.update({"mesh_status": "failed", "mesh_error": f"{type(exc).__name__}: {exc}"})
            item.pop("mesh_file", None)
            failed += 1

    write_json(state_path, state)
    output = Path(state["paths"]["process"]["export_meshes"])
    write_json(output, {
        "schema_version": "1.0", "server": SERVER_NAME, "tool": TOOL_NAME,
        "construction_stage": state["construction_stage"], "state_file": str(state_path),
        "meshes_dir": str(meshes_dir), "ply_files": exported, "failed": failed,
    })
    return success(root, TOOL_NAME, output, {
        "construction_stage": state["construction_stage"],
        "ply_files": exported, "failed": failed,
        "state_file": str(state_path), "meshes_dir": str(meshes_dir),
    })
