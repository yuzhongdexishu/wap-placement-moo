from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import ifcopenshell
    import ifcopenshell.util.unit
except Exception:  # Allows the module to load before the runtime dependency is installed.
    ifcopenshell = None

SERVER_NAME = "BIM-based Scene Modelling MCP Server"
TOOL_NAME = "parse_ifc_building"

# G/P/W integration (adapted from the former IFC_XML_PLY source). This tool embeds the G/P/W functionality that
# runs at parsing time:
#   W1 FrequencyTransform  -> carrier frequency / wavelength record + IFC Pset check
#   P1 MaterialNameTransform -> material name extraction (validated again in tool 4)
#   P3 MaterialColorTransform -> first RGB surface color per material (emitted in tool 4)
#   W2/W4 input collection -> Pset_MaterialWirelessPropagation numeric values per
#                             material/product (consumed by tools 2/4)
#   material association collection (IfcRelAssociatesMaterial) + layer thicknesses,
#                             consumed by the scene generator (tool 4, W6/P2)
RADIO_FREQUENCY_GHZ = 3.5          # W1: n78 carrier frequency of the SP profiles
C0 = 299792458.0                   # speed of light, m/s
WIRELESS_PSET_NAME = "Pset_MaterialWirelessPropagation"
# Static project-level reference data lives in the repository data/ directory. The agents
# pass a stage-local run root (result/<stage>) as project_root_path, so these tables must
# be resolved from this file's location, never from the run root.
PROJECT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# G1-G7 dispatch table (former IFC_XML_PLY geometry kernel): the geometry label per IFC type used for
# PLY file names and scene shape ids. Subtypes match through is_a().
GEOMETRY_LABELS: tuple[tuple[str, str], ...] = (
    ("column", "IfcColumn"), ("beam", "IfcBeam"), ("wall", "IfcWall"),
    ("door", "IfcDoor"), ("window", "IfcWindow"), ("slab", "IfcSlab"),
    ("roof", "IfcRoof"), ("member", "IfcMember"), ("plate", "IfcPlate"),
    ("stair_flight", "IfcStairFlight"), ("curtain_wall", "IfcCurtainWall"),
    ("railing", "IfcRailing"), ("stair", "IfcStair"), ("proxy", "IfcBuildingElementProxy"),
)
# Containers decompose into separately exported constituents (former IFC_XML_PLY behavior).
CONTAINER_LABELS = ("roof", "curtain_wall", "stair", "railing")


def load_state(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline state not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        temp = Path(f.name)
    os.replace(temp, path)
    return path


def relative(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve())


def success(root: str | Path, tool: str, process_file: str | Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "success", "server": SERVER_NAME, "tool": tool,
        "process_file": relative(process_file, root), "summary": summary,
    }


def _safe_token(value: str, fallback: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or fallback


def _normalize_name(name: str | None) -> str:
    """W2 normalize(): case-fold and strip non-alphanumerics for property matching."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def _property_value(prop: Any) -> float:
    """W2 property_value(): wireless properties must be single-valued and finite."""
    if not prop.is_a("IfcPropertySingleValue"):
        raise ValueError("Wireless properties must use IfcPropertySingleValue")
    value = getattr(prop, "NominalValue", None)
    if value is None:
        raise ValueError(f"Empty IFC wireless property: {prop.Name}")
    number = float(getattr(value, "wrappedValue", value))
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"Non-finite IFC wireless property: {prop.Name}")
    return number


def _prop_unit(prop: Any) -> tuple[str, str]:
    """Unit (name, prefix) of an IFC property, if declared."""
    unit = getattr(prop, "Unit", None)
    if unit is None:
        return "", ""
    return str(getattr(unit, "Name", None) or "").upper(), str(getattr(unit, "Prefix", None) or "").upper()


def _frequency_ghz(value: float, unit_name: str, unit_prefix: str) -> float:
    """W1: resolve an IFC Frequency property to GHz, rejecting unknown units."""
    if unit_name:
        factors = {"": 1e-9, "KILO": 1e-6, "MEGA": 1e-3, "GIGA": 1.0}
        if unit_name != "HERTZ" or unit_prefix not in factors:
            raise ValueError(f"Unsupported IFC Frequency unit: {unit_prefix}{unit_name}")
        return value * factors[unit_prefix]
    raise ValueError("IFC Pset Frequency has no unit; declare HERTZ with a prefix")


def _wireless_property_sets(model: Any) -> tuple[dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, dict[str, Any]]]]:
    """W2 wireless_property_sets(): material-level and product-level
    Pset_MaterialWirelessPropagation, converted to plain JSON-able values."""
    target = _normalize_name(WIRELESS_PSET_NAME)
    materials: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    products: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)

    for entity_type in ("IfcExtendedMaterialProperties", "IfcMaterialProperties"):
        try:
            sources = model.by_type(entity_type)
        except RuntimeError:  # absent entity in this IFC schema
            continue
        for definition in sources:
            if _normalize_name(getattr(definition, "Name", "")) != target:
                continue
            material = getattr(definition, "Material", None)
            if material is None:
                continue
            props = (getattr(definition, "ExtendedProperties", None)
                     or getattr(definition, "Properties", None) or [])
            for prop in props:
                name = _normalize_name(prop.Name)
                unit_name, unit_prefix = _prop_unit(prop)
                materials[material.id()][name] = {"value": _property_value(prop),
                                                  "unit": unit_name, "prefix": unit_prefix}
    for relation in model.by_type("IfcRelDefinesByProperties"):
        pset = getattr(relation, "RelatingPropertyDefinition", None)
        if pset is None or _normalize_name(getattr(pset, "Name", "")) != target:
            continue
        for product in (relation.RelatedObjects or []):
            for prop in (getattr(pset, "HasProperties", None) or []):
                name = _normalize_name(prop.Name)
                unit_name, unit_prefix = _prop_unit(prop)
                products[product.id()][name] = {"value": _property_value(prop),
                                                "unit": unit_name, "prefix": unit_prefix}
    return dict(materials), dict(products)


def _safe_materials(select: Any) -> tuple[list[Any], list[Any]]:
    """Former IFC_XML_PLY safe_materials(): flatten IfcMaterial(Set/List/Layer) selections."""
    if select is None:
        return [], []
    name = select.is_a()
    if name == "IfcMaterial":
        return [select], []
    if name == "IfcMaterialLayerSetUsage":
        return _safe_materials(select.ForLayerSet)
    if name == "IfcMaterialLayerSet":
        layers = list(select.MaterialLayers or [])
        return [layer.Material for layer in layers if layer.Material is not None], layers
    if name == "IfcMaterialLayer":
        return ([select.Material] if select.Material else []), [select]
    if name == "IfcMaterialList":
        return list(select.Materials or []), []
    return [], []


def _material_associations(model: Any) -> dict[int, dict[str, list[Any]]]:
    """Former IFC_XML_PLY material_associations(): material/layer links per product id."""
    associations: dict[int, dict[str, list[Any]]] = defaultdict(lambda: {"materials": [], "layers": []})
    for relation in model.by_type("IfcRelAssociatesMaterial"):
        materials, layers = _safe_materials(relation.RelatingMaterial)
        for product in relation.RelatedObjects:
            entry = associations[product.id()]
            entry["materials"].extend(materials)
            entry["layers"].extend(layers)
    for entry in associations.values():
        for key in ("materials", "layers"):
            entry[key] = list({entity.id(): entity for entity in entry[key]}.values())
    return dict(associations)


def _material_color(material: Any) -> list[float] | None:
    """P3: first valid RGB color across the material's surface styles, else None."""
    found: dict[int, Any] = {}
    for representation in (getattr(material, "HasRepresentation", None) or []):
        for itemset in (getattr(representation, "Representations", None) or []):
            for item in (getattr(itemset, "Items", None) or []):
                for style_assignment in (getattr(item, "Styles", None) or []):
                    styles = ((getattr(style_assignment, "Styles", None) or [])
                              if style_assignment.is_a("IfcPresentationStyleAssignment")
                              else [style_assignment])
                    for style in styles:
                        if style.is_a("IfcSurfaceStyle"):
                            found[style.id()] = style
    for style in (found[key] for key in sorted(found)):
        for rendering in (getattr(style, "Styles", None) or []):
            if not (rendering.is_a("IfcSurfaceStyleRendering") or rendering.is_a("IfcSurfaceStyleShading")):
                continue
            color = getattr(rendering, "SurfaceColour", None)
            if color is not None and color.is_a("IfcColourRGB"):
                return [float(color.Red), float(color.Green), float(color.Blue)]
    return None


def _mesh_label(product: Any) -> str:
    """G1-G7 label dispatch: known product types use their scene label, anything else
    keeps the IFC type token so the shared geometry kernel still exports it."""
    for label, ifc_type in GEOMETRY_LABELS:
        if product.is_a(ifc_type):
            return label
    return _safe_token(product.is_a(), "IfcObject")


def parse_ifc_building(ifc_file_path: str, project_root_path: str, construction_stage: str) -> dict[str, Any]:
    """Input: IFC file. Output: pipeline_state.json, which defines every downstream path.
    G/P/W parsing functionality (W1, material associations, wireless property sets,
    P1/P3 data extraction) runs here, inside this tool."""
    if ifcopenshell is None:
        raise RuntimeError("ifcopenshell is required for parse_ifc_building")

    root = Path(project_root_path).expanduser().resolve()
    ifc_path = Path(ifc_file_path).expanduser().resolve()
    stage = construction_stage.strip().lower()
    if stage not in {"frame", "enclosure"}:
        raise ValueError("construction_stage must be 'frame' or 'enclosure'")
    if not ifc_path.is_file():
        raise FileNotFoundError(f"IFC file not found: {ifc_path}")

    work = root / "workspace" / "bim"
    process_dir = root / "process" / "bim_data_processing"
    if work.is_symlink() or work.resolve() == root or not work.resolve().is_relative_to(root):
        raise RuntimeError(f"Unsafe BIM workspace cleanup target: {work}")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    state_file = work / "pipeline_state.json"
    radio_tables_dir = PROJECT_DATA_DIR / "radio_materials"
    radio_tables = {name: str(radio_tables_dir / filename) for name, filename in (
        ("material_class_map", "material_class_map.csv"),
        ("itu_coefficients", "itu_p2040_4_coefficients.csv"),
        ("expert_parameters", "expert_radio_parameters.csv"),
    )}
    for table_path in radio_tables.values():
        if not Path(table_path).is_file():
            raise FileNotFoundError(f"G/P/W radio material table missing: {table_path}")
    paths = {
        "state_file": str(state_file),
        "geometry_cache_dir": str(work / "geometry_cache"),
        "meshes_dir": str(work / "meshes"),
        "scene_xml": str(work / f"{stage}.xml"),
        "radio_tables": radio_tables,
        "process": {
            "parse": str(process_dir / "01_ifc_parsing.json"),
            "tessellate": str(process_dir / "02_object_tessellation.json"),
            "export_meshes": str(process_dir / "03_mesh_export.json"),
            "generate_scene": str(process_dir / "04_scene_generation.json"),
        },
    }

    model = ifcopenshell.open(str(ifc_path))
    length_scale = float(ifcopenshell.util.unit.calculate_unit_scale(model))
    associations = _material_associations(model)
    material_props, product_props = _wireless_property_sets(model)

    # W1: record the carrier frequency and check any IFC wireless Frequency property.
    frequency_checked = 0
    for props in (*material_props.values(), *product_props.values()):
        prop = props.get(_normalize_name("Frequency")) or props.get(_normalize_name("OperatingFrequency"))
        if prop is None:
            continue
        resolved = _frequency_ghz(prop["value"], prop["unit"], prop["prefix"])
        if abs(resolved - RADIO_FREQUENCY_GHZ) > max(1e-8, RADIO_FREQUENCY_GHZ * 1e-6):
            raise ValueError(f"IFC Pset Frequency {resolved:g} GHz differs from configured {RADIO_FREQUENCY_GHZ:g} GHz")
        frequency_checked += 1

    # P1/P3 extraction per material + W2/W4 wireless inputs.
    materials: dict[str, Any] = {}
    for material in model.by_type("IfcMaterial"):
        materials[str(material.id())] = {
            "ifc_id": int(material.id()),
            "name": (getattr(material, "Name", "") or "").strip() or "<Unnamed>",
            "wireless": {key: value for key, value in material_props.get(material.id(), {}).items()},
            "color": _material_color(material),
        }

    objects = []
    nonphysical_skipped = 0
    container_slots: dict[int, list[str]] = defaultdict(list)
    for index, product in enumerate(model.by_type("IfcProduct"), 1):
        if not getattr(product, "Representation", None):
            continue
        ifc_type = product.is_a()
        if ifc_type == "IfcGrid":
            nonphysical_skipped += 1  # datum lines, not radio-propagation solids
            continue
        global_id = str(getattr(product, "GlobalId", None) or product.id())
        label = _mesh_label(product)
        if label in CONTAINER_LABELS and (getattr(product, "IsDecomposedBy", None) or []):
            container_slots[product.id()].append(f"{index:04d}_{label}_{_safe_token(global_id, str(product.id()))}")
            continue  # container decomposition is exported through the constituents
        linked = associations.get(product.id(), {"materials": [], "layers": []})
        objects.append({
            "ifc_id": int(product.id()),
            "ifc_type": ifc_type,
            "object_key": f"{index:04d}_{_safe_token(ifc_type, 'IfcObject')}_{_safe_token(global_id, str(product.id()))}",
            "mesh_label": label,
            "material_ids": [int(material.id()) for material in linked["materials"]],
            "layers": [{"material_id": int(layer.Material.id()) if layer.Material is not None else None,
                        "thickness_m": round(float(layer.LayerThickness) * length_scale, 9)
                        if layer.LayerThickness is not None else None}
                       for layer in linked["layers"]],
            "wireless": {key: value for key, value in product_props.get(product.id(), {}).items()},
            "geometry_status": "pending",
            "mesh_status": "pending",
        })

    state = {
        "construction_stage": stage,
        "project_root": str(root),
        "source_ifc": str(ifc_path),
        "ifc_file_name": ifc_path.name,
        "length_scale": length_scale,
        "radio": {
            "frequency_ghz": RADIO_FREQUENCY_GHZ,
            "wavelength_m": C0 / (RADIO_FREQUENCY_GHZ * 1e9),
            "frequency_checked_in_ifc": frequency_checked,
        },
        "materials": materials,
        "containers_skipped": sum(len(v) for v in container_slots.values()),
        "nonphysical_skipped": nonphysical_skipped,
        "paths": paths,
        "scene_status": "pending",
        "objects": objects,
    }
    write_json(state_file, state)

    output = Path(paths["process"]["parse"])
    write_json(output, {
        "schema_version": "1.0", "server": SERVER_NAME, "tool": TOOL_NAME,
        "construction_stage": stage, "source_ifc": str(ifc_path),
        "ifc_schema": str(model.schema), "objects_discovered": len(objects),
        "materials": len(materials), "containers_skipped": state["containers_skipped"],
        "nonphysical_skipped": nonphysical_skipped,
        "wireless_psets_checked": frequency_checked,
        "radio_frequency_ghz": RADIO_FREQUENCY_GHZ,
        "state_file": str(state_file),
    })
    return success(root, TOOL_NAME, output, {
        "construction_stage": stage,
        "objects_discovered": len(objects),
        "materials": len(materials),
        "state_file": str(state_file),
    })
