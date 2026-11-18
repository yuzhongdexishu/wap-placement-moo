"""W7: use welded triangular PLY geometry for Sionna wedge/edge extraction.

The function does not compute wedge angles or diffraction losses. The later
ray tracer decides which mesh edges interact with individual paths.
"""
from collections import Counter


def W7_DiffractionLossTransform(mesh):
    if not mesh.vertices or not mesh.faces:
        raise ValueError("W7 requires a nonempty triangular mesh")
    edge_count = Counter()
    for triangle in mesh.faces:
        if len(triangle) != 3 or len(set(triangle)) < 3:
            raise ValueError("Invalid triangle found by W7")
        a, b, c = triangle
        for u, v in ((a,b), (b,c), (c,a)):
            edge_count[tuple(sorted((u,v)))] += 1
    # Boundary and manifold edges alike are retained. Inter-object edges may
    # still need reconciliation downstream; no false promise of full manifoldness.
    return {"face_normals": True, "shared_edges": sum(v >= 2 for v in edge_count.values()),
            "nonmanifold_edges": sum(v > 2 for v in edge_count.values())}
