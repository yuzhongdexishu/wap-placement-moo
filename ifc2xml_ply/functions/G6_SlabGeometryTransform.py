"""G6: exact IFC IfcSlab product geometry -> PLY; no default dimensions."""
from .G1_ColumnGeometryTransform import mesh_from_product, PLYMesh


def G6_SlabGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcSlab"):
        raise TypeError("G6 requires IfcSlab")
    return mesh_from_product(ifc_entity)
