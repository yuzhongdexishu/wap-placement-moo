"""P3: IfcSurfaceStyle -> Color, only if an actual RGB value exists."""
from xml.etree.ElementTree import Element, SubElement


def P3_MaterialColorTransform(ifc_entity):
    if ifc_entity is None or not ifc_entity.is_a("IfcSurfaceStyle"):
        raise TypeError("P3 requires IfcSurfaceStyle")
    for style in (getattr(ifc_entity, "Styles", None) or []):
        if not (style.is_a("IfcSurfaceStyleRendering") or style.is_a("IfcSurfaceStyleShading")):
            continue
        color = getattr(style, "SurfaceColour", None)
        if color is not None and color.is_a("IfcColourRGB"):
            result = Element("Material")
            el = SubElement(result, "Color")
            for key, component in zip(("R", "G", "B"), (color.Red, color.Green, color.Blue)):
                SubElement(el, key).text = f"{float(component):.9g}"
            # Alpha is an export convention, not evidence of IFC material opacity.
            SubElement(el, "A", {"source": "export_convention"}).text = "1"
            return result
    raise ValueError("IfcSurfaceStyle contains no RGB surface color")
