"""G3: exact IFC IfcWall product geometry -> PLY; no default dimensions."""
from .G1_ColumnGeometryTransform import mesh_from_product, PLYMesh


def G3_WallGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcWall"):
        raise TypeError("G3 requires IfcWall")
    return mesh_from_product(ifc_entity)
