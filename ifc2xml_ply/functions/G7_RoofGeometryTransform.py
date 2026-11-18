"""G7: exact IFC IfcRoof product geometry -> PLY; no default dimensions."""
from .G1_ColumnGeometryTransform import mesh_from_product, PLYMesh


def G7_RoofGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcRoof"):
        raise TypeError("G7 requires IfcRoof")
    return mesh_from_product(ifc_entity)
