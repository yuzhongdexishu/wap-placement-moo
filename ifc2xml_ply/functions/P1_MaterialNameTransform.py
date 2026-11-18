"""P1: IFC material name to XML. No invented 'Unknown' value."""
from xml.etree.ElementTree import Element, SubElement


def P1_MaterialNameTransform(ifc_entity):
    if ifc_entity is None or not ifc_entity.is_a("IfcMaterial"):
        raise TypeError("P1 requires IfcMaterial")
    name = getattr(ifc_entity, "Name", None)
    if not name or name.strip().casefold() in ("<unnamed>", "unknown"):
        raise ValueError("IfcMaterial has no meaningful Name")
    result = Element("Material")
    SubElement(result, "Name").text = name
    return result
