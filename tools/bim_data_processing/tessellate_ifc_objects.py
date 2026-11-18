from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any
import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.geom
except Exception:
    ifcopenshell = None

from .parse_ifc_building import SERVER_NAME, load_state, success, write_json

TOOL_NAME = "tessellate_ifc_objects"

# ---------------------------------------------------------------------------
# G/P/W integration (adapted from the former IFC_XML_PLY source). This tool embeds the geometry side of the
# G/P/W functionality:
#   G1-G7 geometry transforms -> PLYMesh + mesh_from_product: one shared welded
#       triangulation kernel (weld tolerance 1e-7 m, degenerate/duplicate face removal,
#       vertex compaction) dispatched per IFC type; G2-G7 are exact IFC-type wrappers
#       around the same kernel.
#   W7 DiffractionLossTransform -> welded-mesh edge audit (shared/nonmanifold edge
#       counts and the face_normals flag) so the ray tracer can extract wedges later.
# ---------------------------------------------------------------------------
WELD_TOLERANCE_M = 1e-7


@dataclass
class PLYMesh:
    """G1 PLYMesh: welded metre-scale triangular mesh with shared vertex indices."""
    vertices: list[tuple[float, float, float]]
    faces: list[tuple[int, int, int]]

    def to_ply_ascii(self) -> str:
        out = ["ply", "format ascii 1.0", "comment IFC world coordinates; units metre",
               f"element vertex {len(self.vertices)}", "property double x",
               "property double y", "property double z",
               f"element face {len(self.faces)}", "property list uchar int vertex_indices",
               "end_header"]
        out.extend(" ".join(f"{p:.12g}" for p in vertex) for vertex in self.vertices)
        out.extend(f"3 {a} {b} {c}" for a, b, c in self.faces)
        return "\n".join(out) + "\n"


def _mesh_from_product(entity: Any, weld_tolerance_m: float = WELD_TOLERANCE_M) -> PLYMesh:
    """G1 mesh_from_product: welded shared-index triangulation of one IFC product.

    Reusing coincident tessellation vertices is required so neighboring faces share
    topological edges; without shared indices Sionna can miss a wedge edge even when
    its two triangles look adjacent."""
    if entity is None or getattr(entity, "Representation", None) is None:
        raise ValueError("IFC element has no geometric Representation")
    if not isfinite(weld_tolerance_m) or weld_tolerance_m <= 0:
        raise ValueError("Welding tolerance must be positive and finite")
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    settings.set("convert-back-units", False)
    shape = ifcopenshell.geom.create_shape(settings, entity)
    coords = list(shape.geometry.verts)
    faces = list(shape.geometry.faces)
    if not coords or not faces or len(coords) % 3 or len(faces) % 3:
        raise ValueError("IfcOpenShell returned empty or invalid triangulation")
    original = [tuple(float(x) for x in coords[i:i+3]) for i in range(0, len(coords), 3)]
    if any(not all(isfinite(x) for x in point) for point in original):
        raise ValueError("Non-finite geometry coordinates")

    vertex_ids: dict[tuple[int, int, int], int] = {}
    vertices: list[tuple[float, float, float]] = []
    inverse: list[int] = []
    for point in original:
        key = tuple(round(x / weld_tolerance_m) for x in point)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(point)
        inverse.append(vertex_ids[key])

    valid_faces: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for i in range(0, len(faces), 3):
        try:
            a, b, c = (inverse[int(index)] for index in faces[i:i+3])
        except (ValueError, IndexError) as exc:
            raise ValueError("Invalid tessellation indices") from exc
        if len({a, b, c}) < 3:
            continue
        v0, v1, v2 = (vertices[k] for k in (a, b, c))
        u = [v1[j]-v0[j] for j in range(3)]
        v = [v2[j]-v0[j] for j in range(3)]
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if sqrt(sum(x*x for x in cross)) < 1e-12:
            continue
        if (a, b, c) not in seen:
            seen.add((a, b, c))
            valid_faces.append((a, b, c))
    if not valid_faces:
        raise ValueError("All tessellation faces are degenerate")
    used = sorted({index for face in valid_faces for index in face})
    compact = {old: new for new, old in enumerate(used)}
    return PLYMesh([vertices[index] for index in used],
                   [tuple(compact[index] for index in face) for face in valid_faces])


def _geometry_transform(entity: Any, ifc_type: str) -> PLYMesh:
    """G1-G7 dispatch: every scene type shares the welded geometry kernel; the wrapper
    enforces the exact IFC type before conversion."""
    if entity is None or not entity.is_a(ifc_type):
        raise TypeError(f"Geometry transform requires {ifc_type}")
    return _mesh_from_product(entity)


def _w7_edge_audit(mesh: PLYMesh) -> dict[str, Any]:
    """W7: welded-mesh edge counts used by the ray tracer for wedge/edge extraction.
    Boundary and manifold edges alike are retained; no false promise of manifoldness."""
    if not mesh.vertices or not mesh.faces:
        raise ValueError("W7 requires a nonempty triangular mesh")
    edge_count: Counter = Counter()
    for triangle in mesh.faces:
        if len(triangle) != 3 or len(set(triangle)) < 3:
            raise ValueError("Invalid triangle found by W7")
        a, b, c = triangle
        for u, v in ((a, b), (b, c), (c, a)):
            edge_count[tuple(sorted((u, v)))] += 1
    return {"face_normals": True,
            "shared_edges": sum(count >= 2 for count in edge_count.values()),
            "nonmanifold_edges": sum(count > 2 for count in edge_count.values())}


def tessellate_ifc_objects(state_file: str) -> dict[str, Any]:
    """Input: pipeline_state.json + source IFC. Output: geometry_cache/*.npz + updated state.
    G1-G7 geometry transforms and the W7 edge audit run inside this tool."""
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell is required for tessellate_ifc_objects")

    state_path = Path(state_file).expanduser().resolve()
    state = load_state(state_path)
    root = Path(state["project_root"])
    cache_dir = Path(state["paths"]["geometry_cache_dir"])
    expected_cache = root / "workspace/bim/geometry_cache"
    if cache_dir.is_symlink() or cache_dir.resolve() != expected_cache.resolve():
        raise RuntimeError(f"Unsafe geometry-cache cleanup target: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("*.npz"):
        old.unlink()

    model = ifcopenshell.open(state["source_ifc"])

    completed = failed = 0
    state["scene_status"] = "pending"
    for item in state["objects"]:
        item["mesh_status"] = "pending"
        item.pop("mesh_file", None)
        entity = model.by_id(item["ifc_id"])
        try:
            mesh = _geometry_transform(entity, item["ifc_type"])
            wedge = _w7_edge_audit(mesh)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            cache_file = cache_dir / f"{item['object_key']}.npz"
            np.savez_compressed(cache_file, vertices=vertices, faces=faces)
            item.update({"geometry_status": "completed", "cache_file": str(cache_file), "w7": wedge})
            item.pop("geometry_error", None)
            completed += 1
        except Exception as exc:
            item.update({"geometry_status": "failed", "geometry_error": f"{type(exc).__name__}: {exc}"})
            item.pop("cache_file", None)
            failed += 1

    write_json(state_path, state)
    output = Path(state["paths"]["process"]["tessellate"])
    write_json(output, {
        "schema_version": "1.0", "server": SERVER_NAME, "tool": TOOL_NAME,
        "construction_stage": state["construction_stage"], "state_file": str(state_path),
        "geometry_cache_dir": str(cache_dir), "tessellated": completed, "failed": failed,
        "containers_skipped": state.get("containers_skipped", 0),
        "weld_tolerance_m": WELD_TOLERANCE_M, "w7_edge_audit": True,
    })
    return success(root, TOOL_NAME, output, {
        "construction_stage": state["construction_stage"],
        "tessellated": completed, "failed": failed,
        "state_file": str(state_path), "geometry_cache_dir": str(cache_dir),
    })
