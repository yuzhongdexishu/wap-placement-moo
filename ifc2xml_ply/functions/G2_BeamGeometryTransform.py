"""G2: exact IFC IfcBeam product geometry -> PLY; no default dimensions."""
from .G1_ColumnGeometryTransform import mesh_from_product, PLYMesh


def G2_BeamGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcBeam"):
        raise TypeError("G2 requires IfcBeam")
    return mesh_from_product(ifc_entity)
