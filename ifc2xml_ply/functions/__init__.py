"""functions — IFC -> XML / PLY transformation functions.

Geometry and wireless adapters reuse focused shared helpers. Groups:

* ``G1``-``G7``: geometry mesh transforms (column, beam, wall, door, window, slab, roof).
* ``P1``-``P3``: material attribute transforms (name, thickness, colour).
* ``W1``-``W7``: wireless/material-frequency attribute transforms
  (frequency, relative permittivity, relative permeability, conductivity,
  reflection/transmission/edge-geometry input preparation).
"""
from .G1_ColumnGeometryTransform import G1_ColumnGeometryTransform
from .G2_BeamGeometryTransform import G2_BeamGeometryTransform
from .G3_WallGeometryTransform import G3_WallGeometryTransform
from .G4_DoorGeometryTransform import G4_DoorGeometryTransform
from .G5_WindowGeometryTransform import G5_WindowGeometryTransform
from .G6_SlabGeometryTransform import G6_SlabGeometryTransform
from .G7_RoofGeometryTransform import G7_RoofGeometryTransform
from .P1_MaterialNameTransform import P1_MaterialNameTransform
from .P2_MaterialThicknessTransform import P2_MaterialThicknessTransform
from .P3_MaterialColorTransform import P3_MaterialColorTransform
from .W1_FrequencyTransform import W1_FrequencyTransform
from .W2_RelativePermittivityTransform import W2_RelativePermittivityTransform
from .W3_RelativePermeabilityTransform import W3_RelativePermeabilityTransform
from .W4_ConductivityTransform import W4_ConductivityTransform
from .W5_ReflectionLossTransform import W5_ReflectionLossTransform
from .W6_TransmissionLossTransform import W6_TransmissionLossTransform
from .W7_DiffractionLossTransform import W7_DiffractionLossTransform

__version__ = "0.1.0"

__all__ = [
    "G1_ColumnGeometryTransform",
    "G2_BeamGeometryTransform",
    "G3_WallGeometryTransform",
    "G4_DoorGeometryTransform",
    "G5_WindowGeometryTransform",
    "G6_SlabGeometryTransform",
    "G7_RoofGeometryTransform",
    "P1_MaterialNameTransform",
    "P2_MaterialThicknessTransform",
    "P3_MaterialColorTransform",
    "W1_FrequencyTransform",
    "W2_RelativePermittivityTransform",
    "W3_RelativePermeabilityTransform",
    "W4_ConductivityTransform",
    "W5_ReflectionLossTransform",
    "W6_TransmissionLossTransform",
    "W7_DiffractionLossTransform",
]
