"""W5: configure Sionna radio material scattering input; no reflection-loss calculation."""
from .W2_RelativePermittivityTransform import table_value


def W5_ReflectionLossTransform(expert_row):
    # Zero means only specular reflection (not zero reflection loss).
    scattering = table_value(expert_row, "scattering_coefficient")
    xpd = table_value(expert_row, "xpd_coefficient")
    scattering = 0.0 if scattering is None else scattering
    xpd = 0.0 if xpd is None else xpd
    if not 0 <= scattering <= 1 or not 0 <= xpd <= 1:
        raise ValueError("Scattering and XPD coefficients must be in [0, 1]")
    return {"scattering_coefficient": scattering, "xpd_coefficient": xpd}
