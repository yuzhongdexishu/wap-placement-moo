"""W1: operating frequency and wavelength (conversion metadata only).

If an IFC wireless Pset carries a Frequency, read and check the unit; it must
match --frequency-ghz for frequency-specific static material properties.
The downstream Sionna caller must still configure scene.frequency.
"""
from math import isfinite
C0 = 299792458.0


def W1_FrequencyTransform(frequency_ghz: float, ifc_property=None, assumed_ifc_unit=None):
    f = float(frequency_ghz)
    if not isfinite(f) or f <= 0:
        raise ValueError("Frequency must be finite and positive (GHz)")
    if ifc_property is not None:
        v = getattr(ifc_property, "NominalValue", None)
        if v is None:
            raise ValueError("IFC Frequency has no nominal value")
        value = float(getattr(v, "wrappedValue", v))
        if not isfinite(value) or value <= 0:
            raise ValueError("IFC Frequency must be finite and positive")
        unit = getattr(ifc_property, "Unit", None)
        if unit is None:
            name = (assumed_ifc_unit or "").upper()
            factors = {"HZ":1e-9, "KHZ":1e-6, "MHZ":1e-3, "GHZ":1}
            if name not in factors:
                raise ValueError("IFC Frequency without unit requires --ifc-frequency-unit")
            frequency_in_ghz = value * factors[name]
        else:
            name = (getattr(unit, "Name", None) or "").upper()
            prefix = (getattr(unit, "Prefix", None) or "").upper()
            factors = {"":1e-9, "KILO":1e-6, "MEGA":1e-3, "GIGA":1}
            if name != "HERTZ" or prefix not in factors:
                raise ValueError("Unsupported IFC Frequency unit")
            frequency_in_ghz = value * factors[prefix]
        if abs(frequency_in_ghz-f) > max(1e-8, f*1e-6):
            raise ValueError(f"IFC Pset Frequency {frequency_in_ghz:g} GHz differs from configured {f:g} GHz")
    return {"frequency_ghz": f, "wavelength_m": C0/(f*1e9)}
