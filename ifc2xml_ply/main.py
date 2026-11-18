"""IFC -> Sionna-native Mitsuba scene XML + geometrical PLY meshes.

Scope stops at creating output/<IFC-stem>/{model.xml,meshes/*.ply}.
No Sionna import, ray-tracing run, or propagation-loss simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

from .functions import (
    G1_ColumnGeometryTransform, G2_BeamGeometryTransform, G3_WallGeometryTransform,
    G4_DoorGeometryTransform, G5_WindowGeometryTransform, G6_SlabGeometryTransform,
    G7_RoofGeometryTransform, P1_MaterialNameTransform, P3_MaterialColorTransform,
    W1_FrequencyTransform, W2_RelativePermittivityTransform,
    W3_RelativePermeabilityTransform, W4_ConductivityTransform,
    W5_ReflectionLossTransform, W6_TransmissionLossTransform,
    W7_DiffractionLossTransform,
)
from .functions.G1_ColumnGeometryTransform import mesh_from_product
from .functions.W2_RelativePermittivityTransform import (
    load_csv, normalize, wireless_property_sets, from_properties,
)

ROOT = Path(__file__).resolve().parent
GEOMETRY = (
    ("column", "IfcColumn", G1_ColumnGeometryTransform),
    ("beam", "IfcBeam", G2_BeamGeometryTransform),
    ("wall", "IfcWall", G3_WallGeometryTransform),
    ("door", "IfcDoor", G4_DoorGeometryTransform),
    ("window", "IfcWindow", G5_WindowGeometryTransform),
    ("slab", "IfcSlab", G6_SlabGeometryTransform),
    ("roof", "IfcRoof", G7_RoofGeometryTransform),
    # No G8+ business functions. Shared G1 geometry core handles other IFC elements.
    ("member", "IfcMember", mesh_from_product),
    ("plate", "IfcPlate", mesh_from_product),
    ("stair_flight", "IfcStairFlight", mesh_from_product),
    ("curtain_wall", "IfcCurtainWall", mesh_from_product),
    ("railing", "IfcRailing", mesh_from_product),
    ("stair", "IfcStair", mesh_from_product),
    ("proxy", "IfcBuildingElementProxy", mesh_from_product),
)


def xml_comment(text):
    """Avoid invalid XML comments even if a name is user supplied."""
    text = " ".join(str(text).split()).replace("--", "- -")
    return ET.Comment(text + " " if text.endswith("-") else text)


def safe_materials(select):
    if select is None:
        return [], []
    name = select.is_a()
    if name == "IfcMaterial":
        return [select], []
    if name == "IfcMaterialLayerSetUsage":
        return safe_materials(select.ForLayerSet)
    if name == "IfcMaterialLayerSet":
        layers = list(select.MaterialLayers or [])
        return [layer.Material for layer in layers if layer.Material is not None], layers
    if name == "IfcMaterialLayer":
        return ([select.Material] if select.Material else []), [select]
    if name == "IfcMaterialList":
        return list(select.Materials or []), []
    return [], []


def material_associations(model):
    associations = defaultdict(lambda: {"materials": [], "layers": []})
    for relation in model.by_type("IfcRelAssociatesMaterial"):
        materials, layers = safe_materials(relation.RelatingMaterial)
        for product in relation.RelatedObjects:
            entry = associations[product.id()]
            entry["materials"].extend(materials)
            entry["layers"].extend(layers)
    for entry in associations.values():
        for key in ("materials", "layers"):
            entry[key] = list({entity.id(): entity for entity in entry[key]}.values())
    return associations


def material_styles(material):
    found = {}
    for representation in (getattr(material, "HasRepresentation", None) or []):
        for itemset in (getattr(representation, "Representations", None) or []):
            for item in (getattr(itemset, "Items", None) or []):
                for style_assignment in (getattr(item, "Styles", None) or []):
                    styles = (getattr(style_assignment, "Styles", None) or []) if style_assignment.is_a("IfcPresentationStyleAssignment") else [style_assignment]
                    for style in styles:
                        if style.is_a("IfcSurfaceStyle"):
                            found[style.id()] = style
    return [found[k] for k in sorted(found)]


def read_tables(data_dir):
    by_material = {}
    for row in load_csv(data_dir/"material_class_map.csv"):
        key = (row["ifc_file"], int(row["material_step_id"]))
        if key in by_material:
            raise ValueError(f"Duplicate material mapping: {key}")
        by_material[key] = row
    coeff = {row["material_class"]: row for row in load_csv(data_dir/"itu_p2040_4_coefficients.csv")}
    expert = {}
    expert_file = data_dir/"expert_radio_parameters.csv"
    if expert_file.exists():
        for row in load_csv(expert_file):
            if not any((x or "").strip() for x in row.values()):
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
    return by_material, coeff, expert


def choose_material(linked, args):
    """Resolve heterogeneous IFC material relationships without hidden first-item fallbacks."""
    layers = linked.get("layers", [])
    materials = linked.get("materials", [])
    if layers:
        ids = {layer.Material.id() for layer in layers if layer.Material is not None}
        if not ids:
            return None, layers, "layers without associated IfcMaterial"
        if len(ids) == 1:
            return next(layer.Material for layer in layers if layer.Material), layers, "single-material layer stack"
        if args.mixed_layer_policy != "dominant":
            raise ValueError("Mixed-material IFC layer stack requires --mixed-layer-policy dominant (approximation)")
        main_layer = max((x for x in layers if x.Material is not None), key=lambda x: float(x.LayerThickness or 0))
        return main_layer.Material, layers, "APPROXIMATION: dominant layer material + total stack thickness"
    if len(materials) == 1:
        return materials[0], [], "direct IFC association"
    if len(materials) > 1:
        if args.mixed_layer_policy != "dominant":
            raise ValueError("IFC multiple-material association requires --mixed-layer-policy dominant")
        return sorted(materials, key=lambda mat: mat.id())[0], [], "APPROXIMATION: first material by IFC STEP id"
    return None, [], "no IFC material association"


def radio_material(model_name, product, material, layers, material_note,
                   props_material, props_product, mapping, coefficients, expert_table,
                   freq, args, length_scale):
    """Call all W1-W6 adapters and return (native Mitsuba BSDF element, audit note)."""
    info = W1_FrequencyTransform(freq)
    properties = dict(props_material.get(material.id(), {})) if material is not None else {}
    properties.update(props_product.get(product.id(), {}))  # element overrides material
    _, ifc_frequency = from_properties(properties, ["Frequency", "OperatingFrequency"])
    if ifc_frequency is not None:
        W1_FrequencyTransform(freq, ifc_frequency, args.ifc_frequency_unit)

    key = (model_name, material.id()) if material else None
    mapping_record = mapping.get(key) if key else None
    if key and mapping_record is not None:
        expected_name = (mapping_record.get("material_name") or "").strip()
        current_name = (getattr(material, "Name", "") or "").strip() or "<Unnamed>"
        if current_name != expected_name:
            raise ValueError(f"Material mapping name mismatch for {key}: {current_name} != {expected_name}")
    expert_row = expert_table.get(key) if key else None
    if expert_row:
        measured = (expert_row.get("frequency_ghz") or "").strip()
        needs_frequency = any((expert_row.get(col) or "").strip() for col in (
            "relative_permittivity", "conductivity_s_m", "scattering_coefficient", "xpd_coefficient"))
        if needs_frequency and not measured:
            raise ValueError(f"Expert row for {key} needs frequency_ghz for RF values")
        if measured and abs(float(measured)-freq) > max(1e-8, freq*1e-6):
            raise ValueError(f"Expert row for {key} is at {measured} GHz, not {freq:g} GHz")
    if expert_row and expert_row.get("material_name") and (expert_row["material_name"].strip() != (material.Name or "").strip()):
        raise ValueError(f"Expert table material name mismatch for {key}")
    cls = (mapping_record.get("material_class") or "").strip() if mapping_record else ""
    # Allow material-specific expert properties without an ITU category.
    if not cls and args.fallback_class:
        cls = args.fallback_class
        material_note += f"; EXPLICIT FALLBACK: {cls}"
    itu = coefficients.get(cls)
    eps, eps_source = W2_RelativePermittivityTransform(properties, expert_row, itu, freq)
    mu, mu_source = W3_RelativePermeabilityTransform(properties, expert_row)
    sigma, sigma_source = W4_ConductivityTransform(
        properties, expert_row, itu, freq, args.ifc_conductivity_unit)
    reflection = W5_ReflectionLossTransform(expert_row)
    thickness, thickness_source = W6_TransmissionLossTransform(layers, length_scale, args.default_thickness_m, expert_row)

    material_id = material.id() if material else 0
    bsdf = ET.Element("bsdf", {"type": "radio-material", "id": f"rm_{product.id()}_{material_id}"})
    ET.SubElement(bsdf, "float", {"name": "relative_permittivity", "value": f"{eps:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "conductivity", "value": f"{sigma:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "thickness", "value": f"{thickness:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "scattering_coefficient", "value": f"{reflection['scattering_coefficient']:.12g}"})
    ET.SubElement(bsdf, "float", {"name": "xpd_coefficient", "value": f"{reflection['xpd_coefficient']:.12g}"})

    color = None
    if material is not None:
        try:
            _ = P1_MaterialNameTransform(material)  # P1 actual invocation
        except (ValueError, TypeError):
            pass
        for style in material_styles(material):
            try:
                c = P3_MaterialColorTransform(style)  # P3 actual invocation
                color = [float(c.find(f"Color/{channel}").text) for channel in ("R", "G", "B")]
                break
            except (TypeError, ValueError):
                continue
    if color is not None:
        ET.SubElement(bsdf, "rgb", {"name": "color", "value": " ".join(f"{v:.8g}" for v in color)})

    # W5-W7 values from historic IFC property sets are *read*, but cannot
    # become path-independent dB coefficients in Sionna's native scene XML.
    original_losses = []
    for key in ("ReflectionLoss", "TransmissionLoss", "DiffractionLoss"):
        legacy_value, legacy_property = from_properties(properties, [key])
        if legacy_property is not None:
            original_losses.append(f"{key}={legacy_value:g} dB (IFC supplied; NOT applied to Sionna)")
    provenance = (f"IFC material={material_id}, name={(getattr(material, 'Name', '') if material else '<unassociated>')}; "
                  f"W2 eps={eps_source}; W3 mu_r={mu:g} ({mu_source}); W4 sigma={sigma_source}; "
                  f"W5 specular/diffuse input; W6={thickness_source}; {material_note}; "
                  f"expert reference={(expert_row.get('reference') or expert_row.get('source')) if expert_row else 'none'}; "
                  f"static material evaluation at {info['frequency_ghz']:g} GHz; "
                  + "; ".join(original_losses))
    return bsdf, provenance


def export_ifc(ifc_path, output_root, args, mapping, coefficients, expert):
    try:
        import ifcopenshell
        import ifcopenshell.util.unit
    except ImportError as exc:
        raise RuntimeError("Requires IfcOpenShell geometry engine: pip install -r requirements.txt") from exc
    model = ifcopenshell.open(str(ifc_path))
    length_scale = float(ifcopenshell.util.unit.calculate_unit_scale(model))
    associations = material_associations(model)
    material_props, product_props = wireless_property_sets(model)
    frequency = W1_FrequencyTransform(args.frequency_ghz)
    ifc_stem = ifc_path.stem

    # Intermediate results are removed on error. Old completed output is only
    # replaced after the conversion has finished, to avoid half-generated scenes.
    output_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{ifc_stem}_", dir=output_root))
    target = output_root/ifc_stem
    warnings = []
    failed = []
    exported = 0
    try:
        (temp_dir/"meshes").mkdir()
        root = ET.Element("scene", {"version": "2.1.0"})
        root.append(xml_comment("Sionna RT 2.2 Mitsuba scene; IFC -> XML/PLY only; no ray tracing"))
        root.append(xml_comment(
            f"W1 source=conversion setting; f={frequency['frequency_ghz']:.12g} GHz; "
            f"wavelength={frequency['wavelength_m']:.12g} m; "
            "metadata ONLY: downstream Sionna must set scene.frequency explicitly"))
        root.append(xml_comment("W2-W4 radio-material values are frequency-specific; regenerate XML when frequency changes"))
        root.append(xml_comment("W5/W6 are reflection/transmission material INPUTS, never calculated loss"))
        root.append(xml_comment("W7 uses shared-index triangular PLY geometry and face_normals for later diffraction; no edge/diffraction loss calculation"))
        shape_nodes = []
        stats = Counter()
        for label, ifc_type, converter in GEOMETRY:
            if label == "proxy" and args.exclude_proxies:
                continue
            for entity in model.by_type(ifc_type):
                # Containers decompose into separately-exported constituents.
                if label in ("roof", "curtain_wall", "stair", "railing") and (getattr(entity, "IsDecomposedBy", None) or []):
                    stats["containers_skipped"] += 1
                    continue
                try:
                    mesh = converter(entity)
                    wedge = W7_DiffractionLossTransform(mesh)
                except Exception as exc:
                    failed.append(f"#{entity.id()} {ifc_type}: {type(exc).__name__}: {exc}")
                    continue

                # Missing/ambiguous radio parameters are never treated as a
                # skippable geometry error, even under --allow-skipped-geometry.
                try:
                    linked = associations.get(entity.id(), {"materials": [], "layers": []})
                    material, layers, note = choose_material(linked, args)
                    bsdf, trace = radio_material(
                        ifc_path.name, entity, material, layers, note,
                        material_props, product_props, mapping, coefficients, expert,
                        args.frequency_ghz, args, length_scale)
                except Exception as exc:
                    raise RuntimeError(f"Radio material preparation failed for #{entity.id()} {ifc_type}: {exc}") from exc

                mesh_name = f"{label}_{entity.id()}.ply"
                (temp_dir/"meshes"/mesh_name).write_text(mesh.to_ply_ascii(), encoding="utf-8")
                root.append(xml_comment(trace))
                root.append(bsdf)
                shape = ET.Element("shape", {"type": "ply", "id": f"{label}_{entity.id()}"})
                ET.SubElement(shape, "string", {"name": "filename", "value": f"meshes/{mesh_name}"})
                ET.SubElement(shape, "boolean", {"name": "face_normals", "value": "true" if wedge["face_normals"] else "false"})
                ET.SubElement(shape, "ref", {"name": "bsdf", "id": bsdf.get("id")})
                shape_nodes.append(shape)
                if wedge["nonmanifold_edges"]:
                    warnings.append(f"#{entity.id()} {ifc_type}: {wedge['nonmanifold_edges']} nonmanifold mesh edges")
                if "APPROXIMATION" in note or "FALLBACK" in trace or "explicit --default" in trace:
                    warnings.append(f"#{entity.id()} {ifc_type}: {note}; W6 may use explicit thickness assumption")
                stats[ifc_type] += 1
                exported += 1
        if not exported:
            raise RuntimeError("No geometry exported. " + "; ".join(failed[:5]))
        if failed and not args.allow_skipped_geometry:
            raise RuntimeError(f"{len(failed)} IFC elements could not be exported:\n" + "\n".join(failed[:15]) +
                               ("\n..." if len(failed) > 15 else ""))
        # An explicitly requested partial export must be unmistakably labeled.
        if failed:
            root.insert(0, xml_comment(f"PARTIAL SCENE: {len(failed)} IFC elements failed and were omitted"))
            print(f"WARNING: {ifc_path.name}: partial scene, {len(failed)} omitted elements")
            for detail in failed[:15]: print("  OMITTED:", detail)
        for shape in shape_nodes:
            root.append(shape)
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(temp_dir/"model.xml", encoding="utf-8", xml_declaration=True)
        # Directory names and references need no Sionna installation to check.
        for shape in root.findall("shape"):
            rel = shape.find("string[@name='filename']").get("value")
            if not (temp_dir/rel).is_file():
                raise RuntimeError(f"Broken PLY reference: {rel}")
        if target.exists():
            shutil.rmtree(target)
        temp_dir.rename(target)
        print(f"CREATED {target} | meshes={exported}, containers={stats['containers_skipped']}, failures={len(failed)}")
        if warnings:
            print(f"  Notes: {len(warnings)} modelling or geometry caveats (first 8 below)")
            for warning in warnings[:8]: print("  -", warning)
        return target
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=["frame.ifc", "enclosure.ifc"],
                        help="IFC paths, or filenames searched in data/ (default: both included IFC files)")
    parser.add_argument("--output", type=Path, default=Path.cwd()/"output")
    parser.add_argument("--frequency-ghz", type=float, default=2.4,
                        help="single frequency for W2/W4 coefficient evaluation; downstream Sionna frequency must match")
    parser.add_argument("--fallback-class", default=None,
                        help="EXPLICIT substitution for unmapped/unassociated materials, e.g. concrete")
    parser.add_argument("--default-thickness-m", type=float, default=None,
                        help="EXPLICIT slab thickness for elements without IFC layers, e.g. 0.1")
    parser.add_argument("--mixed-layer-policy", choices=("error", "dominant"), default="error",
                        help="dominant makes a clearly documented one-material approximation")
    parser.add_argument("--ifc-frequency-unit", choices=("Hz", "kHz", "MHz", "GHz"), default=None,
                        help="assume this unit only for IFC Frequency property without its own unit")
    parser.add_argument("--ifc-conductivity-unit", choices=("S/m", "mS/m"), default=None,
                        help="assume this unit only for IFC property without its own unit")
    parser.add_argument("--exclude-proxies", action="store_true")
    parser.add_argument("--allow-skipped-geometry", action="store_true",
                        help="EXPLICIT partial export; omits failed elements with a warning comment")
    args = parser.parse_args()
    W1_FrequencyTransform(args.frequency_ghz)
    if args.default_thickness_m is not None and (not isfinite(args.default_thickness_m) or args.default_thickness_m <= 0):
        parser.error("--default-thickness-m must be positive and finite")
    mapping, coefficients, expert = read_tables(ROOT/"data")
    if args.fallback_class and args.fallback_class not in coefficients:
        parser.error(f"Unknown fallback class {args.fallback_class!r}; choose one of {list(coefficients)}")
    paths = []
    for name in args.inputs:
        supplied = Path(name)
        source = supplied if supplied.is_file() else ROOT/"data"/supplied.name
        if not source.is_file():
            parser.error(f"IFC input not found: {name}")
        paths.append(source.resolve())
    if len({p.stem for p in paths}) < len(paths):
        parser.error("Inputs with the same basename would overwrite each other's output folder")
    for path in paths:
        export_ifc(path, args.output, args, mapping, coefficients, expert)


if __name__ == "__main__":
    main()
