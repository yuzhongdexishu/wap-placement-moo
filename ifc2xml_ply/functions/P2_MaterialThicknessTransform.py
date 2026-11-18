"""P2: IFC material-layer thickness to metres. Returns one <Material> fragment."""
from xml.etree.ElementTree import Element, SubElement


def P2_MaterialThicknessTransform(ifc_entity, unit_scale=0.001):
    if ifc_entity is None or not ifc_entity.is_a("IfcMaterialLayer"):
        raise TypeError("P2 requires IfcMaterialLayer")
    thickness = getattr(ifc_entity, "LayerThickness", None)
    if thickness is None:
        raise ValueError("No LayerThickness on IFC material layer")
    result = Element("Material")
    SubElement(result, "Thickness", {"unit": "m"}).text = f"{float(thickness)*unit_scale:.9g}"
    return result
