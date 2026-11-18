"""G1: column geometry and shared IFC -> metre-scale triangular PLY utility.

G2-G7 and the main dispatcher reuse this geometry kernel rather than creating
additional numbered transformation functions. Vertices are welded (within an
explicit tolerance) so neighboring faces can share topological edges for W7.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt


@dataclass
class PLYMesh:
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


def mesh_from_product(entity, weld_tolerance_m: float = 1e-7) -> PLYMesh:
    import ifcopenshell.geom

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
    if any(not all(isfinite(x) for x in p) for p in original):
        raise ValueError("Non-finite geometry coordinates")

    # Reuse coincident IFC tessellation vertices. Without shared indices,
    # Sionna can miss a wedge edge even when its two triangles look adjacent.
    vertex_ids = {}
    vertices = []
    inverse = []
    for p in original:
        key = tuple(round(x/weld_tolerance_m) for x in p)
        if key not in vertex_ids:
            vertex_ids[key] = len(vertices)
            vertices.append(p)
        inverse.append(vertex_ids[key])

    valid_faces = []
    seen = set()
    for i in range(0, len(faces), 3):
        try:
            a, b, c = (inverse[int(index)] for index in faces[i:i+3])
        except (ValueError, IndexError) as exc:
            raise ValueError("Invalid tessellation indices") from exc
        if len({a,b,c}) < 3:
            continue
        v0, v1, v2 = (vertices[k] for k in (a,b,c))
        u = [v1[j]-v0[j] for j in range(3)]
        v = [v2[j]-v0[j] for j in range(3)]
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        if sqrt(sum(x*x for x in cross)) < 1e-12:
            continue
        if (a,b,c) not in seen:
            seen.add((a,b,c))
            valid_faces.append((a,b,c))
    if not valid_faces:
        raise ValueError("All tessellation faces are degenerate")
    used = sorted({i for f in valid_faces for i in f})
    compact = {old:new for new,old in enumerate(used)}
    return PLYMesh([vertices[i] for i in used],
                   [tuple(compact[i] for i in f) for f in valid_faces])


def G1_ColumnGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcColumn"):
        raise TypeError("G1 requires IfcColumn")
    return mesh_from_product(ifc_entity)
