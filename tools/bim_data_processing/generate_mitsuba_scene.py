from __future__ import annotations

import csv
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .parse_ifc_building import SERVER_NAME, load_state, success, write_json

TOOL_NAME = "generate_mitsuba_scene"

# ---------------------------------------------------------------------------
# G/P/W integration (adapted from the former IFC_XML_PLY source). This tool embeds the radio-material side of
# the G/P/W functionality, so the scene generator emits Sionna-native radio-material
# BSDFs instead of placeholder diffuse/dielectric/conductor entries:
#   W2 RelativePermittivityTransform  -> IFC Pset -> expert CSV -> ITU-R P.2040-4
#   W3 RelativePermeabilityTransform  -> mu_r = 1 enforcement (Sionna RadioMaterial)
#   W4 ConductivityTransform          -> IFC Pset (unit-checked) -> expert -> ITU
#   W5 ReflectionLossTransform        -> scattering/XPD coefficients (expert CSV)
#   W6 TransmissionLossTransform      -> IFC layer thickness (P2) -> expert -> default
#   P2 MaterialThicknessTransform     -> layer thickness sum to metres
#   P3 MaterialColorTransform         -> optional RGB color node on the BSDF
# The CSV tables live in data/radio_materials/ (paths stored in pipeline_state.json).
# ---------------------------------------------------------------------------
FALLBACK_CLASS = "concrete"       # explicit substitution for unmapped materials
DEFAULT_THICKNESS_M = 0.10        # explicit slab thickness without IFC/expert input
MIXED_LAYER_POLICY = "dominant"   # documented one-material approximation
RADIO_MATERIAL_TYPE = "radio-material"


def _normalize_name(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").casefold())


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _table_value(record: dict[str, str] | None, column: str) -> float | None:
    if record and (record.get(column) or "").strip():
        value = float(record[column])
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"Non-finite {column} in expert CSV")
        return value
    return None


def _itu_value(coefficient: dict[str, str], frequency_ghz: float, value_type: str) -> float:
    f = float(frequency_ghz)
    if not float(coefficient["min_GHz"]) <= f <= float(coefficient["max_GHz"]):
        raise ValueError(f"{coefficient['material_class']} ITU coefficients not valid at {f:g} GHz")
    a, b = (("a", "b") if value_type == "eps" else ("c", "d"))
    return float(coefficient[a]) * f ** float(coefficient[b])


def _read_tables(paths: dict[str, Any]) -> tuple[dict[tuple[str, int], dict[str, str]],
                                                dict[str, dict[str, str]],
                                                dict[tuple[str, int], dict[str, str]]]:
    by_material: dict[tuple[str, int], dict[str, str]] = {}
    for row in _load_csv(paths["material_class_map"]):
        key = (row["ifc_file"], int(row["material_step_id"]))
        if key in by_material:
            raise ValueError(f"Duplicate material mapping: {key}")
        by_material[key] = row
    coefficients = {row["material_class"]: row for row in _load_csv(paths["itu_coefficients"])}
    expert: dict[tuple[str, int], dict[str, str]] = {}
    for row in _load_csv(paths["expert_parameters"]):
        if not any((value or "").strip() for value in row.values()):
            continue
        material_id = (row.get("material_step_id") or "").strip()
        if not material_id:
            raise ValueError("expert_radio_parameters.csv needs material_step_id per row")
        key = ((row.get("ifc_file") or "").strip(), int(material_id))
        if key in expert:
            raise ValueError(f"Duplicate expert row: {key}")
        if not (row.get("source") or "").strip():
            raise ValueError(f"Expert row {key} requires source (expert or published reference)")
        expert[key] = row
    return by_material, coefficients, expert


def _choose_material(material_ids: list[int], layers: list[dict[str, Any]],
                     materials: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """Resolve heterogeneous IFC material relationships without hidden first-item fallbacks."""
    if layers:
        ids = {layer["material_id"] for layer in layers if layer["material_id"] is not None}
        if not ids:
            return None, layers, "layers without associated IfcMaterial"
        if len(ids) == 1:
            material_id = next(iter(ids))
            return materials.get(str(material_id)), layers, "single-material layer stack"
        if MIXED_LAYER_POLICY != "dominant":
            raise ValueError("Mixed-material IFC layer stack requires dominant policy (approximation)")
        main_layer = max((layer for layer in layers if layer["material_id"] is not None),
                         key=lambda layer: float(layer["thickness_m"] or 0))
        return materials.get(str(main_layer["material_id"])), layers, \
            "APPROXIMATION: dominant layer material + total stack thickness"
    if len(material_ids) == 1:
        return materials.get(str(material_ids[0])), [], "direct IFC association"
    if len(material_ids) > 1:
        if MIXED_LAYER_POLICY != "dominant":
            raise ValueError("IFC multiple-material association requires dominant policy")
        chosen = materials.get(str(min(material_ids)))
        return chosen, [], "APPROXIMATION: first material by IFC STEP id"
    return None, [], "no IFC material association"


def _from_properties(properties: dict[str, dict[str, Any]], names: list[str]) -> tuple[float | None, str]:
    for name in names:
        record = properties.get(_normalize_name(name))
        if record is not None:
            return float(record["value"]), name
    return None, ""


def _w2_permittivity(properties: dict[str, dict[str, Any]], expert_row: dict[str, str] | None,
                     itu_row: dict[str, str] | None, frequency_ghz: float) -> tuple[float, str]:
    value, _ = _from_properties(properties, ["RelativePermittivity", "DielectricConstant",
                                             "Permittivity", "epsilon_r", "Er"])
    source = "IFC Pset_MaterialWirelessPropagation" if value is not None else ""
    if value is None:
        value = _table_value(expert_row, "relative_permittivity")
        if value is not None:
            source = "expert_radio_parameters.csv"
    if value is None and itu_row is not None:
        value = _itu_value(itu_row, frequency_ghz, "eps")
        source = "ITU-R P.2040-4 class coefficient table"
    if value is None:
        raise ValueError("No relative permittivity from IFC/expert/ITU")
    if value < 1:
        raise ValueError(f"Sionna RadioMaterial requires relative permittivity >= 1, got {value}")
    return value, source


def _w3_permeability(properties: dict[str, dict[str, Any]], expert_row: dict[str, str] | None) -> tuple[float, str]:
    value, _ = _from_properties(properties, ["RelativePermeability", "Permeability", "mu_r", "Mr"])
    source = "IFC Pset_MaterialWirelessPropagation" if value is not None else ""
    if value is None:
        value = _table_value(expert_row, "relative_permeability")
        if value is not None:
            source = "expert_radio_parameters.csv"
    if value is None:
        value, source = 1.0, "explicit Sionna non-magnetic assumption"
    if abs(value - 1.0) > 1e-6:
        raise ValueError(f"Sionna built-in RadioMaterial requires mu_r=1; received {value}")
    return value, source


def _w4_conductivity(properties: dict[str, dict[str, Any]], expert_row: dict[str, str] | None,
                     itu_row: dict[str, str] | None, frequency_ghz: float) -> tuple[float, str]:
    value, prop_name = _from_properties(properties, ["Conductivity", "ElectricalConductivity"])
    source = "IFC Pset_MaterialWirelessPropagation" if value is not None else ""
    if value is not None:
        record = properties[_normalize_name(prop_name)]
        unit, prefix = record.get("unit", ""), record.get("prefix", "")
        if unit:
            if unit not in ("SIEMENS_PER_METRE", "SIEMENS_PER_METER") or prefix not in ("", "MILLI"):
                raise ValueError("Unsupported IFC conductivity unit; convert to S/m in the expert CSV")
            value *= 0.001 if prefix == "MILLI" else 1.0
        else:
            raise ValueError("IFC conductivity has no unit; declare S/m or mS/m")
    if value is None:
        value = _table_value(expert_row, "conductivity_s_m")
        if value is not None:
            source = "expert_radio_parameters.csv"
    if value is None and itu_row is not None:
        value = _itu_value(itu_row, frequency_ghz, "sigma")
        source = "ITU-R P.2040-4 class coefficient table"
    if value is None:
        raise ValueError("No conductivity from IFC/expert/ITU")
    if value < 0:
        raise ValueError("Conductivity must be >= 0")
    return value, source


def _w5_reflection_inputs(expert_row: dict[str, str] | None) -> tuple[float, float]:
    scattering = _table_value(expert_row, "scattering_coefficient")
    xpd = _table_value(expert_row, "xpd_coefficient")
    scattering = 0.0 if scattering is None else scattering
    xpd = 0.0 if xpd is None else xpd
    if not 0 <= scattering <= 1 or not 0 <= xpd <= 1:
        raise ValueError("Scattering and XPD coefficients must be in [0, 1]")
    return scattering, xpd


def _w6_thickness(layers: list[dict[str, Any]], expert_row: dict[str, str] | None) -> tuple[float, str]:
    if layers:
        values = []
        for layer in layers:
            if layer["thickness_m"] is None:
                raise ValueError("No LayerThickness on IFC material layer (P2)")
            values.append(float(layer["thickness_m"]))
        return sum(values), "IFC P2 material-layer thickness (single-slab approximation)"
    value = _table_value(expert_row, "thickness_m")
    if value is not None:
        return value, "expert_radio_parameters.csv thickness_m"
    return DEFAULT_THICKNESS_M, "explicit default-thickness-m assumption"


def _xml_comment(text: str) -> ET.Element:
    text = " ".join(str(text).split()).replace("--", "- -")
    return ET.Comment(text + " " if text.endswith("-") else text)


def _radio_material(item: dict[str, Any], material: dict[str, Any] | None, layers: list[dict[str, Any]],
                    material_note: str, mapping: dict[str, str] | None, expert_row: dict[str, str] | None,
                    coefficients: dict[str, dict[str, str]], frequency_ghz: float) -> tuple[ET.Element, str]:
    """Call the embedded W2-W6 adapters and return (radio-material BSDF element, trace)."""
    properties: dict[str, dict[str, Any]] = dict(material.get("wireless", {})) if material else {}
    properties.update(item.get("wireless", {}) or {})  # element overrides material

    material_id = int(material["ifc_id"]) if material else 0
    material_name = material.get("name", "<unassociated>") if material else "<unassociated>"
    if mapping is not None:
        expected_name = (mapping.get("material_name") or "").strip()
        current_name = str(material_name).strip() or "<Unnamed>"
        if current_name != expected_name:
            raise ValueError(f"Material mapping name mismatch: {current_name} != {expected_name}")
        status = (mapping.get("mapping_status") or "").strip()
        if status == "class_level_proxy":
            material_note += "; CLASS PROXY: " + (mapping.get("rationale") or "")
        elif status == "scenario_assumption":
            material_note += "; SCENARIO ASSUMPTION: " + (mapping.get("rationale") or "")
    if expert_row:
        measured = (expert_row.get("frequency_ghz") or "").strip()
        needs_frequency = any((expert_row.get(column) or "").strip() for column in (
            "relative_permittivity", "conductivity_s_m", "scattering_coefficient", "xpd_coefficient"))
        if needs_frequency and not measured:
            raise ValueError("Expert row needs frequency_ghz for RF values")
        if measured and abs(float(measured) - frequency_ghz) > max(1e-8, frequency_ghz * 1e-6):
            raise ValueError(f"Expert row is at {measured} GHz, not {frequency_ghz:g} GHz")
    if expert_row and expert_row.get("material_name") and \
            (expert_row["material_name"].strip() != str(material_name).strip()):
        raise ValueError("Expert table material name mismatch")

    material_class = (mapping.get("material_class") or "").strip() if mapping else ""
    if not material_class:
        material_class = FALLBACK_CLASS
        material_note += f"; EXPLICIT FALLBACK: {material_class}"
    itu_row = coefficients.get(material_class)

    eps, eps_source = _w2_permittivity(properties, expert_row, itu_row, frequency_ghz)
    mu, mu_source = _w3_permeability(properties, expert_row)
    sigma, sigma_source = _w4_conductivity(properties, expert_row, itu_row, frequency_ghz)
    scattering, xpd = _w5_reflection_inputs(expert_row)
    thickness, thickness_source = _w6_thickness(layers, expert_row)

    bsdf = ET.Element("bsdf", {"type": RADIO_MATERIAL_TYPE, "id": f"rm_{item['ifc_id']}_{material_id}"})
    ET.SubElement(bsdf, "float", {"name": "relative_permittivity", "value": f"{eps:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "conductivity", "value": f"{sigma:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "thickness", "value": f"{thickness:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "scattering_coefficient", "value": f"{scattering:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "xpd_coefficient", "value": f"{xpd:.12g}"})
    color = material.get("color") if material else None
    if color:
        ET.SubElement(bsdf, "rgb", {"name": "color", "value": " ".join(f"{v:.8g}" for v in color)})

    trace = (f"IFC material={material_id}, name={material_name}; "
             f"W2 eps={eps_source}; W3 mu_r={mu:g} ({mu_source}); W4 sigma={sigma_source}; "
             f"W5 specular/diffuse input; W6={thickness_source}; {material_note}; "
             f"static material evaluation at {frequency_ghz:g} GHz")
    return bsdf, trace


def generate_mitsuba_scene(state_file: str) -> dict[str, Any]:
    """Input: pipeline_state.json + meshes/*.ply + data/radio_materials/*.csv.
    Output: one Sionna-native Mitsuba scene XML with per-object radio-material BSDFs
    (G/P/W W2-W6 + P2/P3 functionality runs inside this tool)."""
    state_path = Path(state_file).expanduser().resolve()
    state = load_state(state_path)
    root = Path(state["project_root"])
    xml_path = Path(state["paths"]["scene_xml"])
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    frequency_ghz = float(state["radio"]["frequency_ghz"])
    wavelength_m = float(state["radio"]["wavelength_m"])
    materials = state.get("materials", {})
    mapping, coefficients, expert = _read_tables(state["paths"]["radio_tables"])
    ifc_file_name = str(state.get("ifc_file_name") or Path(state["source_ifc"]).name)

    scene = ET.Element("scene", {"version": "2.1.0"})
    scene.append(_xml_comment("Sionna RT radio-material scene; IFC -> XML/PLY through the G/P/W pipeline"))
    scene.append(_xml_comment(
        f"W1 source=SP carrier setting; f={frequency_ghz:.12g} GHz; "
        f"wavelength={wavelength_m:.12g} m; "
        "metadata ONLY: downstream Sionna must set scene.frequency explicitly"))
    scene.append(_xml_comment("W2-W4 radio-material values are frequency-specific; regenerate XML when frequency changes"))
    scene.append(_xml_comment("W5/W6 are reflection/transmission material INPUTS, never calculated loss"))
    scene.append(_xml_comment("W7 uses shared-index triangular PLY geometry and face_normals for later diffraction; "
                             "no edge/diffraction loss calculation"))

    shape_count = 0
    warnings: list[str] = []
    for item in state["objects"]:
        if item.get("mesh_status") != "completed":
            continue
        mesh_file = Path(item["mesh_file"])
        if not mesh_file.is_file():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        material_ids = [int(value) for value in item.get("material_ids", [])]
        layers = list(item.get("layers", []))
        material, chosen_layers, note = _choose_material(material_ids, layers, materials)
        key = (ifc_file_name, int(material["ifc_id"])) if material else None
        mapping_record = mapping.get(key) if key else None
        expert_row = expert.get(key) if key else None
        bsdf, trace = _radio_material(item, material, chosen_layers, note, mapping_record,
                                      expert_row, coefficients, frequency_ghz)
        scene.append(_xml_comment(trace))
        scene.append(bsdf)

        mesh_ref = os.path.relpath(mesh_file, xml_path.parent).replace("\\", "/")
        shape = ET.Element("shape", {"type": "ply", "id": f"{item['mesh_label']}_{item['ifc_id']}"})
        ET.SubElement(shape, "string", {"name": "filename", "value": mesh_ref})
        face_normals = bool((item.get("w7") or {}).get("face_normals", False))
        ET.SubElement(shape, "boolean", {"name": "face_normals", "value": "true" if face_normals else "false"})
        ET.SubElement(shape, "ref", {"name": "bsdf", "id": bsdf.get("id")})
        scene.append(shape)
        w7 = item.get("w7") or {}
        if w7.get("nonmanifold_edges"):
            warnings.append(f"#{item['ifc_id']} {item['ifc_type']}: {w7['nonmanifold_edges']} nonmanifold mesh edges")
        if ("APPROXIMATION" in note or "FALLBACK" in trace or "SCENARIO ASSUMPTION" in trace
                or "CLASS PROXY" in trace or "default-thickness" in trace):
            warnings.append(f"#{item['ifc_id']} {item['ifc_type']}: {note}; W6 may use explicit thickness assumption")
        shape_count += 1

    if not shape_count:
        raise RuntimeError("No completed meshes available to build the scene")

    ET.indent(scene, space="    ")
    ET.ElementTree(scene).write(xml_path, encoding="utf-8", xml_declaration=True)
    state["scene_status"] = "completed"
    state["scene_material_warnings"] = warnings[:200]
    write_json(state_path, state)

    output = Path(state["paths"]["process"]["generate_scene"])
    write_json(output, {
        "schema_version": "1.0", "server": SERVER_NAME, "tool": TOOL_NAME,
        "construction_stage": state["construction_stage"], "state_file": str(state_path),
        "scene_xml": str(xml_path), "shape_count": shape_count,
        "radio_material": RADIO_MATERIAL_TYPE,
        "material_resolution": "W2/W3/W4/W5/W6 with P2/P3 inputs",
        "warnings": warnings[:200],
    })
    return success(root, TOOL_NAME, output, {
        "construction_stage": state["construction_stage"],
        "shape_count": shape_count, "state_file": str(state_path),
        "scene_xml": str(xml_path), "material_warnings": len(warnings),
    })
