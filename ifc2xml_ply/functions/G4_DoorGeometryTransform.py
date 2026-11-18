"""G4: exact IFC IfcDoor product geometry -> PLY; no default dimensions."""
from .G1_ColumnGeometryTransform import mesh_from_product, PLYMesh


def G4_DoorGeometryTransform(ifc_entity) -> PLYMesh:
    if ifc_entity is None or not ifc_entity.is_a("IfcDoor"):
        raise TypeError("G4 requires IfcDoor")
    return mesh_from_product(ifc_entity)
