"""W2: IFC/expert/ITU material permittivity with provenance.

Also contains common W2/W4 CSV and property-source utilities. Nothing in this
module calculates propagation loss.
"""
from __future__ import annotations
import csv
import re
from math import isfinite


def normalize(name):
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def property_value(prop):
    if not prop.is_a("IfcPropertySingleValue"):
        raise ValueError("Wireless properties must use IfcPropertySingleValue")
    value = getattr(prop, "NominalValue", None)
    if value is None:
        raise ValueError(f"Empty IFC wireless property: {prop.Name}")
    v = float(getattr(value, "wrappedValue", value))
    if not isfinite(v):
        raise ValueError(f"Non-finite IFC wireless property: {prop.Name}")
    return v


def wireless_property_sets(model):
    """Find material-level and object-level Pset_MaterialWirelessPropagation.

    IFC2X3 IfcExtendedMaterialProperties is supported in addition to IFC4's
    IfcMaterialProperties and object IfcRelDefinesByProperties assignments.
    """
    from collections import defaultdict
    target = normalize("Pset_MaterialWirelessPropagation")
    mats, objects = defaultdict(dict), defaultdict(dict)

    for typ in ("IfcExtendedMaterialProperties", "IfcMaterialProperties"):
        try:
            sources = model.by_type(typ)
        except RuntimeError:  # absent entity in this IFC schema
            continue
        for definition in sources:
            if normalize(getattr(definition, "Name", "")) != target:
                continue
            material = getattr(definition, "Material", None)
            if material is None:
                continue
            props = (getattr(definition, "ExtendedProperties", None) or
                     getattr(definition, "Properties", None) or [])
            for prop in props:
                mats[material.id()][normalize(prop.Name)] = prop
    for relation in model.by_type("IfcRelDefinesByProperties"):
        pset = getattr(relation, "RelatingPropertyDefinition", None)
        if pset is None or normalize(getattr(pset, "Name", "")) != target:
            continue
        for product in (relation.RelatedObjects or []):
            for prop in (getattr(pset, "HasProperties", None) or []):
                objects[product.id()][normalize(prop.Name)] = prop
    return mats, objects


def from_properties(properties, names):
    for name in names:
        prop = properties.get(normalize(name))
        if prop is not None:
            return property_value(prop), prop
    return None, None


def table_value(record, column):
    if record and (record.get(column) or "").strip():
        x = float(record[column])
        if not isfinite(x):
            raise ValueError(f"Non-finite {column} in expert CSV")
        return x
    return None


def itu_value(coefficient, frequency_ghz, value_type):
    f = float(frequency_ghz)
    if not float(coefficient["min_GHz"]) <= f <= float(coefficient["max_GHz"]):
        raise ValueError(f"{coefficient['material_class']} ITU coefficients not valid at {f:g} GHz")
    a, b = (("a", "b") if value_type == "eps" else ("c", "d"))
    return float(coefficient[a]) * f**float(coefficient[b])


def W2_RelativePermittivityTransform(properties, expert_row, itu_row, frequency_ghz):
    x, _ = from_properties(properties, ["RelativePermittivity", "DielectricConstant", "Permittivity", "epsilon_r", "Er"])
    source = "IFC Pset_MaterialWirelessPropagation" if x is not None else ""
    if x is None:
        x = table_value(expert_row, "relative_permittivity")
        if x is not None: source = "expert_radio_parameters.csv"
    if x is None and itu_row is not None:
        x = itu_value(itu_row, frequency_ghz, "eps")
        source = "ITU-R P.2040-4 class coefficient table"
    if x is None:
        raise ValueError("No relative permittivity from IFC/expert/ITU")
    if x < 1:
        raise ValueError(f"Sionna RadioMaterial requires relative permittivity >= 1, got {x}")
    return x, source
