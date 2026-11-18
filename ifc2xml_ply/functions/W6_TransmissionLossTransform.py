"""W6: derive Sionna slab thickness from IFC layers or explicit expert input.

This prepares transmission *inputs* only; it never calculates path losses.
"""
from math import isfinite
from .W2_RelativePermittivityTransform import table_value


def W6_TransmissionLossTransform(layers, ifc_length_unit_scale,
                                 default_thickness_m, expert_row=None):
    if layers:
        from .P2_MaterialThicknessTransform import P2_MaterialThicknessTransform
        values = []
        for layer in layers:
            fragment = P2_MaterialThicknessTransform(layer, ifc_length_unit_scale)
            values.append(float(fragment.find("Thickness").text))
        thickness = sum(values)
        source = "IFC P2 material-layer thickness (single-slab approximation)"
    else:
        value = table_value(expert_row, "thickness_m")
        if value is not None:
            thickness, source = value, "expert_radio_parameters.csv thickness_m"
        else:
            if default_thickness_m is None:
                raise ValueError("No IFC/expert thickness; specify --default-thickness-m")
            thickness = float(default_thickness_m)
            source = "explicit --default-thickness-m assumption"
    if not isfinite(thickness) or thickness <= 0:
        raise ValueError("Sionna effective slab thickness must be finite and > 0")
    return thickness, source
