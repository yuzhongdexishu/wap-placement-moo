"""G5: exact IFC IfcWindow product geometry -> PLY; no default dimensions."""
from .G1_ColumnGeometryTransform import mesh_from_product, PLYMesh


def G5_WindowGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcWindow"):
        raise TypeError("G5 requires IfcWindow")
    return mesh_from_product(ifc_entity)
