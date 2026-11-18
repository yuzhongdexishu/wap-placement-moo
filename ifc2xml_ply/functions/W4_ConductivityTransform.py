"""W4: conductivity (S/m) read from IFC property, expert CSV, or ITU coefficients."""
from .W2_RelativePermittivityTransform import from_properties, table_value, itu_value


def W4_ConductivityTransform(properties, expert_row, itu_row, frequency_ghz,
                             ifc_missing_unit=None):
    x, prop = from_properties(properties, ["Conductivity", "ElectricalConductivity"])
    source = "IFC Pset_MaterialWirelessPropagation" if x is not None else ""
    if x is not None:
        unit = getattr(prop, "Unit", None)
        if unit is None:
            if ifc_missing_unit not in ("S/m", "mS/m"):
                raise ValueError("IFC conductivity has no unit; specify --ifc-conductivity-unit")
            x *= 0.001 if ifc_missing_unit == "mS/m" else 1.0
        else:
            # IFC compound conductivity units cannot be safely inferred from
            # arbitrary derived-unit names or prefixes. Reject unrecognized.
            name = (getattr(unit, "Name", None) or "").upper()
            prefix = (getattr(unit, "Prefix", None) or "").upper()
            if name not in ("SIEMENS_PER_METRE", "SIEMENS_PER_METER") or prefix not in ("", "MILLI"):
                raise ValueError("Unsupported IFC conductivity unit; convert to S/m in the expert CSV")
            x *= 0.001 if prefix == "MILLI" else 1.0
    if x is None:
        x = table_value(expert_row, "conductivity_s_m")
        if x is not None: source = "expert_radio_parameters.csv"
    if x is None and itu_row is not None:
        x = itu_value(itu_row, frequency_ghz, "sigma")
        source = "ITU-R P.2040-4 class coefficient table"
    if x is None:
        raise ValueError("No conductivity from IFC/expert/ITU")
    if x < 0:
        raise ValueError("Conductivity must be >= 0")
    return x, source
