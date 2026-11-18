"""W3: enforce Sionna built-in RadioMaterial's non-magnetic mu_r=1 assumption."""
from .W2_RelativePermittivityTransform import from_properties, table_value


def W3_RelativePermeabilityTransform(properties, expert_row):
    x, _ = from_properties(properties, ["RelativePermeability", "Permeability", "mu_r", "Mr"])
    source = "IFC Pset_MaterialWirelessPropagation" if x is not None else ""
    if x is None:
        x = table_value(expert_row, "relative_permeability")
        if x is not None:
            source = "expert_radio_parameters.csv"
    if x is None:
        x, source = 1.0, "explicit Sionna non-magnetic assumption"
    if abs(x-1.0) > 1e-6:
        raise ValueError(f"Sionna built-in RadioMaterial requires mu_r=1; received {x}")
    return x, source
