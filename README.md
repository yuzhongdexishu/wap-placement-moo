# Placement of Wireless Access Points for Smart Construction Using LLM-Driven Agents

> **MOO (Modelling-Optimization-Operation) Framework**: Integrating BIM data, Sionna RT ray-tracing, genetic algorithm optimization, and LLM-driven dual-agent workflow automation for construction-stage-aware WAP placement

---

# 1 Summary of supplemental materials

This repository contains the code, inputs, selected results, and supplementary tables for the paper. The materials are indexed below by manuscript section. Supplementary figure labels S1?S18 identify the displayed images.

### Fig. S1 BIM architecture views

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/frame_architecture.png?raw=true" width="420" alt="Frame Architecture" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/enclosure_architecture.png?raw=true" width="420" alt="Enclosure Architecture" /> |
|:--:|:--:|
| **(a)** Frame stage architecture | **(b)** Enclosure stage architecture |

The supplemental materials (`Supplementary_Tables/`) include the following tables:

| Table | Content |
|-------|---------|
| [Table S1](Supplementary_Tables/S1/) | Materials and Mapping — literature sources, material electromagnetic parameters, and IFC-to-PLY/XML object and attribute mapping |
| [Table S2](Supplementary_Tables/S2/) | Optimization and Evaluation — genetic-algorithm settings, ray-tracing and evaluation settings, receiver grid, and priority weights |
| [Table S3](Supplementary_Tables/S3/) | Scenario Configuration — Construction stages, sensor profiles (SP1–SP5), and smart construction scenes (SC1–SC8) |
| [Table S4](Supplementary_Tables/S4/) | Scheme Performance — PM vs. expert-based and geometric coverage baseline scores |
| [Table S5](Supplementary_Tables/S5/) | Six-LLM Run Records — 480-task completion audit across six LLMs |


# 2 General Introduction

2.1 This private repository provides research materials for **"Placement of Wireless Access Points for Smart Construction Using LLM-Driven Agents."** The case study concerns a primary school teaching building at the structural-frame and wall-and-enclosure stages.

2.2 We greatly appreciate the selfless spirits of the voluntary contributors of a series of open Python libraries, including **IfcOpenShell** (https://github.com/IfcOpenShell/IfcOpenShell), **Sionna RT** (https://github.com/NVlabs/sionna), **trimesh**, **shapely**, and so on. Our work stands on the shoulders of these giants.

2.3 No license file is included in this repository. Access is currently limited to authorized collaborators.


# 3 Methodology

The MOO framework comprises three core components: (i) BIM-based signal coverage modelling, (ii) simulation-based WAP placement optimization, and (iii) agent-driven WAP placement workflow operation.

## 3.1 BIM-based signal coverage modelling

The data transformation from BIM's IFC format to Sionna RT-compatible PLY meshes and XML scene files avoids manually and repeatedly establishing the building model for wireless signal simulations. This process involves three main steps: (i) extending the IFC material description with wireless-propagation properties (`Pset_MaterialWirelessPropagation`), (ii) devising the mapping schema between IFC, PLY, and XML, and (iii) developing the **IFC2PLY&XML function library**.

This library includes **17 core functions** divided into three groups: **G-series** (geometry conversion, 7 functions), **P-series** (material property conversion, 3 functions), and **W-series** (wireless-propagation property conversion, 7 functions).

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/frame_architecture.png?raw=true" width="420" alt="IFC2PLY&XML Library" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/enclosure_architecture.png?raw=true" width="420" alt="Signal Coverage Model" /> |
|:--:|:--:|
| Frame-stage BIM visualization base | Enclosure-stage BIM visualization base |

The G-series functions convert IFC building elements (columns, beams, walls, doors, windows, slabs, roofs) into vertex-based PLY meshes. The P-series functions map IFC material attributes to XML entries. The W-series functions encode wireless-propagation properties including reflection loss, transmission loss, and diffraction loss parameters.

### Fig. S2–S3 Floor-level BIM visualization bases

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/frame_1F_cutaway.png?raw=true" width="280" alt="Frame 1F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/frame_2F_cutaway.png?raw=true" width="280" alt="Frame 2F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/frame_3F_cutaway.png?raw=true" width="280" alt="Frame 3F" /> |
|:--:|:--:|:--:|
| **(a)** Frame — Floor 1 | **(b)** Frame — Floor 2 | **(c)** Frame — Floor 3 |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/enclosure_1F_cutaway.png?raw=true" width="280" alt="Enclosure 1F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/enclosure_2F_cutaway.png?raw=true" width="280" alt="Enclosure 2F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/MOO_Framework/data/visualization_bases/enclosure_3F_cutaway.png?raw=true" width="280" alt="Enclosure 3F" /> |
|:--:|:--:|:--:|
| **(a)** Enclosure — Floor 1 | **(b)** Enclosure — Floor 2 | **(c)** Enclosure — Floor 3 |


## 3.2 Simulation-based WAP placement optimization

The WAP placement optimization maximizes a weighted composite signal quality score of **RSRP**, **RSRQ**, and **SINR** across 600 formal receiver positions, using a **genetic algorithm (GA)** with real-valued chromosome encoding.

**Optimization model:**

| Component | Configuration |
|-----------|--------------|
| Objective | Maximize weighted composite F = mean(wᵢ · norm(RSRPᵢ, RSRQᵢ, SINRᵢ)) |
| Priority weight | w = 2 (priority zones), w = 1 (regular areas) |
| Decision variables | 3D WAP coordinates (x, y, z) |
| Receivers | 600 fixed positions per stage |
| Candidate WAP sites | 150 (mixed receiver-derived + scene-centre + grid anchors) |
| Minimum WAP spacing | 4.0 m |
| Obstacle clearance | ≥ 0.05 m |
| Thresholds | RSRP ≥ −140 dBm, RSRQ ≥ −19.5 dB, SINR ≥ 0 dB |

**GA configuration:**

| Operator | Setting |
|----------|---------|
| Population size | 80 |
| Generations | 35 |
| Selection | Tournament (size 4) |
| Crossover | BLX-α (α = 0.5) |
| Mutation | Gaussian, adaptive σ |
| Ray-tracing | Sionna RT RadioMapSolver, 1,000,000 samples/TX, 0.75 m cell size |
| Propagation | LoS, specular reflection, diffraction, refraction enabled |

The WAP count is determined iteratively from N = 1. The GA starts from multiple independently-found RF-feasible layouts.

### PM heatmaps — Frame stage (RSRP per floor) (Fig. S4, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_rsrp_1F.png?raw=true" width="280" alt="Frame PM RSRP 1F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_rsrp_2F.png?raw=true" width="280" alt="Frame PM RSRP 2F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_rsrp_3F.png?raw=true" width="280" alt="Frame PM RSRP 3F" /> |
|:--:|:--:|:--:|
| **(a)** Frame PM — RSRP 1F | **(b)** Frame PM — RSRP 2F | **(c)** Frame PM — RSRP 3F |

### PM heatmaps — Frame stage (RSRQ per floor) (Fig. S5, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_rsrq_1F.png?raw=true" width="280" alt="Frame PM RSRQ 1F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_rsrq_2F.png?raw=true" width="280" alt="Frame PM RSRQ 2F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_rsrq_3F.png?raw=true" width="280" alt="Frame PM RSRQ 3F" /> |
|:--:|:--:|:--:|
| **(a)** Frame PM — RSRQ 1F | **(b)** Frame PM — RSRQ 2F | **(c)** Frame PM — RSRQ 3F |

### PM heatmaps — Frame stage (SINR per floor) (Fig. S6, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_sinr_1F.png?raw=true" width="280" alt="Frame PM SINR 1F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_sinr_2F.png?raw=true" width="280" alt="Frame PM SINR 2F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/frame_PM_sinr_3F.png?raw=true" width="280" alt="Frame PM SINR 3F" /> |
|:--:|:--:|:--:|
| **(a)** Frame PM — SINR 1F | **(b)** Frame PM — SINR 2F | **(c)** Frame PM — SINR 3F |

### PM heatmaps — Enclosure stage (RSRP per floor) (Fig. S7, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/enclosure_PM_rsrp_1F.png?raw=true" width="280" alt="Enclosure PM RSRP 1F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/enclosure_PM_rsrp_2F.png?raw=true" width="280" alt="Enclosure PM RSRP 2F" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/registered_heatmaps/enclosure_PM_rsrp_3F.png?raw=true" width="280" alt="Enclosure PM RSRP 3F" /> |
|:--:|:--:|:--:|
| **(a)** Enclosure PM — RSRP 1F | **(b)** Enclosure PM — RSRP 2F | **(c)** Enclosure PM — RSRP 3F |

### PM 3D stacked overview — Frame stage (Fig. S8, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_PM_rsrp_stacked.png?raw=true" width="280" alt="Frame PM RSRP stacked" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_PM_rsrq_stacked.png?raw=true" width="280" alt="Frame PM RSRQ stacked" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_PM_sinr_stacked.png?raw=true" width="280" alt="Frame PM SINR stacked" /> |
|:--:|:--:|:--:|
| **(a)** Frame PM — RSRP | **(b)** Frame PM — RSRQ | **(c)** Frame PM — SINR |

### PM 3D stacked overview — Enclosure stage (Fig. S9, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_PM_rsrp_stacked.png?raw=true" width="280" alt="Enclosure PM RSRP stacked" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_PM_rsrq_stacked.png?raw=true" width="280" alt="Enclosure PM RSRQ stacked" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_PM_sinr_stacked.png?raw=true" width="280" alt="Enclosure PM SINR stacked" /> |
|:--:|:--:|:--:|
| **(a)** Enclosure PM — RSRP | **(b)** Enclosure PM — RSRQ | **(c)** Enclosure PM — SINR |

Below are the WAP placement optimization codes and baseline comparison codes:

```
MOO_Framework/tools/wap_layout_optimization/
├── define_problem.py        ← Optimization problem formulation
├── build_surrogate.py        ← Sionna RT surrogate model construction
├── generate_candidates.py    ← 150-site candidate dictionary generation
└── optimize.py              ← GA-based layout optimization + 600-point evaluation

MOO_Framework/baselines/
├── expert_layouts.py        ← EbM1 (Huawei), EbM2 (Akuvox), EbM3 (Siemens)
├── geometric_layouts.py     ← GcM1 (hexagonal), GcM2 (circular) coverage
├── geometry.py              ← Geometric coverage primitives
└── score.py                 ← Unified scoring on common Sionna comparison map
```

↑↑↑ Codes for simulation-based WAP placement optimization and baseline comparisons


## 3.3 Agent-driven WAP placement workflow operation

The agent system automates the complete WAP placement workflow through a **dual-agent plan-and-execute architecture** orchestrating **8 MCP-based agent-usable tools** across **2 MCP servers**.

### MCP Server and Tool Architecture

```
Human Request
     │
     ▼
┌─────────────────┐
│  Planning Agent  │  ← LLM reasoning core
│  (Preliminary    │
│   12-step plan)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execution Agent  │  ← LLM reasoning core (Thought → Action → Observation)
│  (Refine plan &  │
│   invoke tools)  │
└────────┬────────┘
         │
    ┌────┴──────────┐
    ▼               ▼
┌────────────┐ ┌────────────┐
│ BIM MCP    │ │ WAP MCP    │
│ Server     │ │ Server     │
│ (Port 8101)│ │ (Port 8102)│
│            │ │            │
│ 4 tools:   │ │ 4 tools:   │
│ 1.Parse    │ │ 5.Define   │
│ 2.Tessell. │ │ 6.Surrogate│
│ 3.Export   │ │ 7.Candidate│
│ 4.Mitsuba  │ │ 8.Optimize │
└────────────┘ └────────────┘
```

**BIM-based Scene Modelling MCP Server** (`MOO_Framework/mcp_servers/bim_server.py`):
| Tool | Function | Code |
|------|----------|------|
| 1. Parse IFC Building | Parse IFC file, extract building elements and material associations | `MOO_Framework/tools/bim_data_processing/parse_ifc_building.py` |
| 2. Tessellate IFC Objects | Triangulate IFC elements into mesh geometry | `MOO_Framework/tools/bim_data_processing/tessellate_ifc_objects.py` |
| 3. Export Object Meshes | Export PLY meshes + XML scene file with radio materials | `MOO_Framework/tools/bim_data_processing/export_object_meshes.py` |
| 4. Generate Mitsuba Scene | Produce Sionna-compatible Mitsuba XML scene | `MOO_Framework/tools/bim_data_processing/generate_mitsuba_scene.py` |

**WAP Placement Optimization MCP Server** (`MOO_Framework/mcp_servers/wap_server.py`):
| Tool | Function | Code |
|------|----------|------|
| 5. Define Problem | Set up optimization objectives, constraints, and sensor profile | `MOO_Framework/tools/wap_layout_optimization/define_problem.py` |
| 6. Build Surrogate | Build Sionna RT radio map surrogate model | `MOO_Framework/tools/wap_layout_optimization/build_surrogate.py` |
| 7. Generate Candidates | Generate and validate 150-site candidate dictionary | `MOO_Framework/tools/wap_layout_optimization/generate_candidates.py` |
| 8. Optimize | Run GA optimization + evaluate 600-receiver results | `MOO_Framework/tools/wap_layout_optimization/optimize.py` |

A **shared structured data record** maintains data continuity across the tool chain.

### Dual-Agent System

The planning agent (`MOO_Framework/agents/planning_agent.py`) and execution agent (`MOO_Framework/agents/execution_agent.py`) follow a **plan-and-execute** pattern:

1. **Planning Agent** — Receives the human task request, decomposes into an ordered preliminary plan
2. **Execution Agent** — Validates the plan against available MCP tools, produces the final execution sequence, invokes all 8 tools with **Thought–Action–Observation** logging

The six evaluated LLM configurations use the same planning and execution agent workflow. Their recorded display names appear in Section 4.2.3 and Table S5.

```
MOO_Framework/agents/
├── planning_agent.py        ← Receives task, generates 12-step preliminary plan
└── execution_agent.py       ← Validates plan, invokes 8 MCP tools, logs Thought-Action-Observation

MOO_Framework/mcp_servers/
├── bim_server.py            ← Exposes 4 BIM scene modelling tools via MCP
└── wap_server.py            ← Exposes 4 WAP optimization tools via MCP
```

↑↑↑ Codes for the dual-agent system and MCP servers


# 4 A demonstrative case study

## 4.1 Building profile and scenario settings

A primary school teaching building is studied at two construction stages. The formal six-LLM matrix combines two stages, five sensor profiles (SP1–SP5), and eight scenarios per stage (F-SC1–F-SC8 and E-SC1–E-SC8). [Table S3](Supplementary_Tables/S3/) records the profiles and priority regions. The [stage visualization index](stage_visualizations/) links all current floor overviews and scenario images.

### Fig. S10 Construction-stage priority regions by floor

The six panels place the F-SC and E-SC scenarios into the 1F, 2F, and 3F IFC floor contexts. Yellow indicates priority regions (weight 2); regular areas have weight 1. Fig. S10–S12 are repository supplementary-image labels, separate from the manuscript’s numbered Figures 1–13.

| Fig. S10 panel | Structural-frame stage | Fig. S10 panel | Wall-and-enclosure stage |
| --- | --- | --- | --- |
| **(a)** Frame 1F | <img src="stage_visualizations/frame/Frame_SC_Overview_1F.png" width="350" alt="Fig. S10(a) frame 1F" /> | **(d)** Enclosure 1F | <img src="stage_visualizations/enclosure/Enclosure_SC_Overview_1F.png" width="350" alt="Fig. S10(d) enclosure 1F" />
| **(b)** Frame 2F | <img src="stage_visualizations/frame/Frame_SC_Overview_2F.png" width="350" alt="Fig. S10(b) frame 2F" /> | **(e)** Enclosure 2F | <img src="stage_visualizations/enclosure/Enclosure_SC_Overview_2F.png" width="350" alt="Fig. S10(e) enclosure 2F" />
| **(c)** Frame 3F | <img src="stage_visualizations/frame/Frame_SC_Overview_3F.png" width="350" alt="Fig. S10(c) frame 3F" /> | **(f)** Enclosure 3F | <img src="stage_visualizations/enclosure/Enclosure_SC_Overview_3F.png" width="350" alt="Fig. S10(f) enclosure 3F" />

### Fig. S11 and Fig. S12 Individual smart-construction scenarios

Each row shows the priority-region plan (Fig. S11) and corresponding 3D view (Fig. S12) for one scenario. The panel letters (a)–(h) are F-SC1–F-SC8; (i)–(p) are E-SC1–E-SC8.

| Scenario and activity | Fig. S11 priority-region plan | Fig. S12 3D view |
| --- | --- | --- |
| **F-SC1: Column inspection** | **Fig. S11(a)**<br><img src="stage_visualizations/frame/scenarios/F-SC1_Column_inspection.png" width="300" alt="Fig. S11(a) F-SC1 plan" /> | **Fig. S12(a)**<br><img src="stage_visualizations/frame/scenarios/F-SC1_3D.png" width="300" alt="Fig. S12(a) F-SC1 3D" /> |
| **F-SC2: Beam inspection** | **Fig. S11(b)**<br><img src="stage_visualizations/frame/scenarios/F-SC2_Beam_inspection.png" width="300" alt="Fig. S11(b) F-SC2 plan" /> | **Fig. S12(b)**<br><img src="stage_visualizations/frame/scenarios/F-SC2_3D.png" width="300" alt="Fig. S12(b) F-SC2 3D" /> |
| **F-SC3: Slab-edge inspection** | **Fig. S11(c)**<br><img src="stage_visualizations/frame/scenarios/F-SC3_Slab-edge_inspection.png" width="300" alt="Fig. S11(c) F-SC3 plan" /> | **Fig. S12(c)**<br><img src="stage_visualizations/frame/scenarios/F-SC3_3D.png" width="300" alt="Fig. S12(c) F-SC3 3D" /> |
| **F-SC4: Layout marking** | **Fig. S11(d)**<br><img src="stage_visualizations/frame/scenarios/F-SC4_Layout_marking.png" width="300" alt="Fig. S11(d) F-SC4 plan" /> | **Fig. S12(d)**<br><img src="stage_visualizations/frame/scenarios/F-SC4_3D.png" width="300" alt="Fig. S12(d) F-SC4 3D" /> |
| **F-SC5: Structural-material delivery** | **Fig. S11(e)**<br><img src="stage_visualizations/frame/scenarios/F-SC5_Structural-material_delivery.png" width="300" alt="Fig. S11(e) F-SC5 plan" /> | **Fig. S12(e)**<br><img src="stage_visualizations/frame/scenarios/F-SC5_3D.png" width="300" alt="Fig. S12(e) F-SC5 3D" /> |
| **F-SC6: Structural-material stocktaking** | **Fig. S11(f)**<br><img src="stage_visualizations/frame/scenarios/F-SC6_Structural-material_stocktaking.png" width="300" alt="Fig. S11(f) F-SC6 plan" /> | **Fig. S12(f)**<br><img src="stage_visualizations/frame/scenarios/F-SC6_3D.png" width="300" alt="Fig. S12(f) F-SC6 3D" /> |
| **F-SC7: Structural-waste sorting** | **Fig. S11(g)**<br><img src="stage_visualizations/frame/scenarios/F-SC7_Structural-waste_sorting.png" width="300" alt="Fig. S11(g) F-SC7 plan" /> | **Fig. S12(g)**<br><img src="stage_visualizations/frame/scenarios/F-SC7_3D.png" width="300" alt="Fig. S12(g) F-SC7 3D" /> |
| **F-SC8: Safety monitoring** | **Fig. S11(h)**<br><img src="stage_visualizations/frame/scenarios/F-SC8_Safety_monitoring.png" width="300" alt="Fig. S11(h) F-SC8 plan" /> | **Fig. S12(h)**<br><img src="stage_visualizations/frame/scenarios/F-SC8_3D.png" width="300" alt="Fig. S12(h) F-SC8 3D" /> |
| **E-SC1: Wall plastering** | **Fig. S11(i)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC1_Wall_plastering.png" width="300" alt="Fig. S11(i) E-SC1 plan" /> | **Fig. S12(i)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC1_3D.png" width="300" alt="Fig. S12(i) E-SC1 3D" /> |
| **E-SC2: Architectural inspection** | **Fig. S11(j)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC2_Architectural_inspection.png" width="300" alt="Fig. S11(j) E-SC2 plan" /> | **Fig. S12(j)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC2_3D.png" width="300" alt="Fig. S12(j) E-SC2 3D" /> |
| **E-SC3: Finishing-material delivery** | **Fig. S11(k)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC3_Finishing-material_delivery.png" width="300" alt="Fig. S11(k) E-SC3 plan" /> | **Fig. S12(k)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC3_3D.png" width="300" alt="Fig. S12(k) E-SC3 3D" /> |
| **E-SC4: Joint filling and sealing** | **Fig. S11(l)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC4_Joint_filling_and_sealing.png" width="300" alt="Fig. S11(l) E-SC4 plan" /> | **Fig. S12(l)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC4_3D.png" width="300" alt="Fig. S12(l) E-SC4 3D" /> |
| **E-SC5: Floor surface operation** | **Fig. S11(m)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC5_Floor_surface_operation.png" width="300" alt="Fig. S11(m) E-SC5 plan" /> | **Fig. S12(m)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC5_3D.png" width="300" alt="Fig. S12(m) E-SC5 3D" /> |
| **E-SC6: Stair cleaning** | **Fig. S11(n)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC6_Stair_cleaning.png" width="300" alt="Fig. S11(n) E-SC6 plan" /> | **Fig. S12(n)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC6_3D.png" width="300" alt="Fig. S12(n) E-SC6 3D" /> |
| **E-SC7: Finishing-material stocktaking** | **Fig. S11(o)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC7_Finishing-material_stocktaking.png" width="300" alt="Fig. S11(o) E-SC7 plan" /> | **Fig. S12(o)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC7_3D.png" width="300" alt="Fig. S12(o) E-SC7 3D" /> |
| **E-SC8: Finishing-waste sorting** | **Fig. S11(p)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC8_Finishing-waste_sorting.png" width="300" alt="Fig. S11(p) E-SC8 plan" /> | **Fig. S12(p)**<br><img src="stage_visualizations/enclosure/scenarios/E-SC8_3D.png" width="300" alt="Fig. S12(p) E-SC8 3D" /> |

All 1F–3F stage views and the original-resolution scenario images are retained under [`stage_visualizations/`](stage_visualizations/).

## 4.2 Results and performance comparison

5 baseline schemes are developed for comparison with the **Proposed Method (PM)**:

**Expert-based Approaches**: Huawei-inspired (EbM1), Akuvox-inspired (EbM2), Siemens-inspired (EbM3)

**Geometric Coverage Methods**: Hexagonal deployment (GcM1), Circular deployment (GcM2)

### PM layout under SP3 / SC1

The PM comparison uses the SP3 profile at 23 dBm and the SC1 priority scenario (weight 2).

| Stage | PM WAPs | PM F (optimization map) | PM F (common comparison map) | Best baseline F | PM failures / 600 | Min. count proven |
|-------|---------|------------------------|------------------------------|-----------------|-------------------|-------------------|
| Frame | 3 | 0.997597 | 0.997597 | 0.995065 (EbM1/EbM2) | 0 | Yes |
| Enclosure | 12 | 0.987974 | 0.986763 | 0.784174 (EbM3) | 0 | Yes |

### 4.2.1 Comparison with expert-based deployment methods

| Method | Description | Frame F | Enclosure F | PM uplift (Frame) | PM uplift (Enclosure) |
|--------|-------------|---------|-------------|-------------------|----------------------|
| **PM** | Proposed Method (MOO) | **0.998** | **0.987** | — | — |
| EbM1 | Huawei-inspired | 0.995 | 0.673 | +0.3% | +46.6% |
| EbM2 | Akuvox-inspired | 0.995 | 0.679 | +0.3% | +45.4% |
| EbM3 | Siemens-inspired | 0.983 | 0.784 | +1.5% | +25.8% |

### Expert 3D stacked heatmaps — Frame stage (Fig. S13, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM1_rsrp_stacked.png?raw=true" width="280" alt="EbM1 Frame RSRP" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM2_rsrp_stacked.png?raw=true" width="280" alt="EbM2 Frame RSRP" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM3_rsrp_stacked.png?raw=true" width="280" alt="EbM3 Frame RSRP" /> |
|:--:|:--:|:--:|
| **(a)** Frame EbM1 (Huawei) — RSRP | **(b)** Frame EbM2 (Akuvox) — RSRP | **(c)** Frame EbM3 (Siemens) — RSRP |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM1_rsrq_stacked.png?raw=true" width="280" alt="EbM1 Frame RSRQ" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM2_rsrq_stacked.png?raw=true" width="280" alt="EbM2 Frame RSRQ" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM3_rsrq_stacked.png?raw=true" width="280" alt="EbM3 Frame RSRQ" /> |
|:--:|:--:|:--:|
| Frame EbM1 (Huawei) — RSRQ | Frame EbM2 (Akuvox) — RSRQ | Frame EbM3 (Siemens) — RSRQ |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM1_sinr_stacked.png?raw=true" width="280" alt="EbM1 Frame SINR" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM2_sinr_stacked.png?raw=true" width="280" alt="EbM2 Frame SINR" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_EbM3_sinr_stacked.png?raw=true" width="280" alt="EbM3 Frame SINR" /> |
|:--:|:--:|:--:|
| Frame EbM1 (Huawei) — SINR | Frame EbM2 (Akuvox) — SINR | Frame EbM3 (Siemens) — SINR |

### Expert 3D stacked heatmaps — Enclosure stage (Fig. S14, panels a–c)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM1_rsrp_stacked.png?raw=true" width="280" alt="EbM1 Enclosure RSRP" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM2_rsrp_stacked.png?raw=true" width="280" alt="EbM2 Enclosure RSRP" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM3_rsrp_stacked.png?raw=true" width="280" alt="EbM3 Enclosure RSRP" /> |
|:--:|:--:|:--:|
| **(a)** Enclosure EbM1 (Huawei) — RSRP | **(b)** Enclosure EbM2 (Akuvox) — RSRP | **(c)** Enclosure EbM3 (Siemens) — RSRP |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM1_rsrq_stacked.png?raw=true" width="280" alt="EbM1 Enclosure RSRQ" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM2_rsrq_stacked.png?raw=true" width="280" alt="EbM2 Enclosure RSRQ" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM3_rsrq_stacked.png?raw=true" width="280" alt="EbM3 Enclosure RSRQ" /> |
|:--:|:--:|:--:|
| Enclosure EbM1 (Huawei) — RSRQ | Enclosure EbM2 (Akuvox) — RSRQ | Enclosure EbM3 (Siemens) — RSRQ |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM1_sinr_stacked.png?raw=true" width="280" alt="EbM1 Enclosure SINR" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM2_sinr_stacked.png?raw=true" width="280" alt="EbM2 Enclosure SINR" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_EbM3_sinr_stacked.png?raw=true" width="280" alt="EbM3 Enclosure SINR" /> |
|:--:|:--:|:--:|
| Enclosure EbM1 (Huawei) — SINR | Enclosure EbM2 (Akuvox) — SINR | Enclosure EbM3 (Siemens) — SINR |

### Violin plots — Expert comparison (Fig. S15, panels a–b)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/violin/violin_expert_frame.png?raw=true" width="420" alt="Expert Frame Violin" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/violin/violin_expert_enclosure.png?raw=true" width="420" alt="Expert Enclosure Violin" /> |
|:--:|:--:|
| **(a)** PM vs EbM1/EbM2/EbM3 — Frame stage | **(b)** PM vs EbM1/EbM2/EbM3 — Enclosure stage |

### 4.2.2 Comparison with geometric coverage methods

| Method | Description | Frame F | Enclosure F | PM uplift (Frame) | PM uplift (Enclosure) |
|--------|-------------|---------|-------------|-------------------|----------------------|
| **PM** | Proposed Method (MOO) | **0.998** | **0.987** | — | — |
| GcM1 | Hexagonal coverage + GA | 0.860 | 0.674 | +16.0% | +46.3% |
| GcM2 | Circular coverage + GA | 0.792 | 0.682 | +25.9% | +44.7% |

### Geometric 3D stacked heatmaps — Frame stage (Fig. S16, panels a–b)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_GcM1_rsrp_stacked.png?raw=true" width="420" alt="GcM1 Frame RSRP" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_GcM2_rsrp_stacked.png?raw=true" width="420" alt="GcM2 Frame RSRP" /> |
|:--:|:--:|
| **(a)** Frame GcM1 (Hexagonal) — RSRP | **(b)** Frame GcM2 (Circular) — RSRP |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_GcM1_rsrq_stacked.png?raw=true" width="420" alt="GcM1 Frame RSRQ" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_GcM2_rsrq_stacked.png?raw=true" width="420" alt="GcM2 Frame RSRQ" /> |
|:--:|:--:|
| Frame GcM1 (Hexagonal) — RSRQ | Frame GcM2 (Circular) — RSRQ |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_GcM1_sinr_stacked.png?raw=true" width="420" alt="GcM1 Frame SINR" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/frame_GcM2_sinr_stacked.png?raw=true" width="420" alt="GcM2 Frame SINR" /> |
|:--:|:--:|
| Frame GcM1 (Hexagonal) — SINR | Frame GcM2 (Circular) — SINR |

### Geometric 3D stacked heatmaps — Enclosure stage (Fig. S17, panels a–b)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_GcM1_rsrp_stacked.png?raw=true" width="420" alt="GcM1 Enclosure RSRP" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_GcM2_rsrp_stacked.png?raw=true" width="420" alt="GcM2 Enclosure RSRP" /> |
|:--:|:--:|
| **(a)** Enclosure GcM1 (Hexagonal) — RSRP | **(b)** Enclosure GcM2 (Circular) — RSRP |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_GcM1_rsrq_stacked.png?raw=true" width="420" alt="GcM1 Enclosure RSRQ" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_GcM2_rsrq_stacked.png?raw=true" width="420" alt="GcM2 Enclosure RSRQ" /> |
|:--:|:--:|
| Enclosure GcM1 (Hexagonal) — RSRQ | Enclosure GcM2 (Circular) — RSRQ |

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_GcM1_sinr_stacked.png?raw=true" width="420" alt="GcM1 Enclosure SINR" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/stacked_registered_heatmaps/enclosure_GcM2_sinr_stacked.png?raw=true" width="420" alt="GcM2 Enclosure SINR" /> |
|:--:|:--:|
| Enclosure GcM1 (Hexagonal) — SINR | Enclosure GcM2 (Circular) — SINR |

### Violin plots — Geometric comparison (Fig. S18, panels a–b)

| <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/violin/violin_geometric_frame.png?raw=true" width="420" alt="Geometric Frame Violin" /> | <img src="https://github.com/yuzhongdexishu/wap-placement-moo/blob/main/plots/violin/violin_geometric_enclosure.png?raw=true" width="420" alt="Geometric Enclosure Violin" /> |
|:--:|:--:|
| **(a)** PM vs GcM1/GcM2 — Frame stage | **(b)** PM vs GcM1/GcM2 — Enclosure stage |

### 4.2.3 Task completion rates of six LLM agents

The six-LLM benchmark covers **480 formal tasks** (6 models × 2 stages × 5 sensor profiles × 8 scenarios). SP1–SP5 = 13, 18, 23, 28, 33 dBm at 3.5 GHz.

| Model | Family | Tasks completed | Tasks assigned | Completion rate |
|-------|--------|----------------|----------------|-----------------|
| Claude Opus-5 | Anthropic | 80 | 80 | 100% |
| DeepSeek-V4-Pro | DeepSeek | 80 | 80 | 100% |
| Qwen-3.8 Max | Alibaba | 80 | 80 | 100% |
| GPT-6-Astra | OpenAI | 80 | 80 | 100% |
| GPT-5.6-SOL | OpenAI | 80 | 80 | 100% |
| Grok-4.7 | xAI | 74 | 80 | 92.5% |
| **Total** | | **474** | **480** | **98.75%** |

Audit records: `MOO_Framework/result/six_llm/completion_rate.json`, `MOO_Framework/result/six_llm/tasks/`, [`Supplementary_Tables/S5/Table_S5_Six_LLM_Run_Records.xlsx`](Supplementary_Tables/S5/Table_S5_Six_LLM_Run_Records.xlsx).


# 5 Discussion

## 5.1 Contributions of this study

The study integrates IFC-based building and material conversion, simulation-based WAP optimization, and tool-driven agent operation in one MOO workflow. The repository keeps the dual-agent code, eight MCP tools, comparative baselines, and formal task-level records together so readers can trace the reported case study to its implementation and supplementary data.

## 5.2 Transferability of the method

- **Modelling**: IFC2PLY&XML library processes BIM elements; building geometry, materials, and radio parameters are configurable per project.
- **Optimization**: WAP parameters, receiver locations/weights, and coverage requirements are reconfigurable.
- **Operation**: Agent pattern and tool-calling workflow remain unchanged; tool inputs adapt to new buildings.

### Method and result boundaries

- The minimum-count proof applies to **600 selected receivers** and **150 candidate WAP sites**, not the continuous work domain.
- Independent dense probes can fail (PM: 6/13,818 frame; 185/12,071 enclosure positions).
- The **global optimum** over all physical sites has not been proven.
- Six-LLM completion rate is based on successful 8-tool execution, not dense-domain validation.


# Repository usage

## Environment setup

```powershell
git clone https://github.com/yuzhongdexishu/wap-placement-moo.git
cd wap-placement-moo
python -m pip install -r MOO_Framework/requirements.txt

Copy-Item MOO_Framework/config/llm_config.example.json MOO_Framework/config/llm_config.json
Copy-Item MOO_Framework/config/models.example.json MOO_Framework/config/models.json
```
The copied files are local and ignored by Git. For a regular MOO run, put the DeepSeek key in `MOO_Framework/config/llm_config.json` under `api_key`, or set the `DEEPSEEK_API_KEY` environment variable (which takes precedence). For the six-LLM benchmark, put each model?s key in its `api_key` field in `MOO_Framework/config/models.json`; also check that model names and `base_url` values match your provider. Keep `*.example.json` files free of real keys and do not commit the local JSON files.

## Run the MOO framework

```powershell
cd MOO_Framework
python run_all.py        # Terminal 1: BIM (:8101), WAP (:8102), Planning (:7001), Execution (:7002)
python send_task_moo.py  # Terminal 2: frame + enclosure, SP3 (23 dBm), SC1
```


## Run baseline comparisons

```powershell
cd MOO_Framework
python run_experiment_expert.py    # EbM1/EbM2/EbM3
python run_experiment_geometric.py  # GcM1/GcM2
python run_baseline_comparison.py   # Unified comparison on common Sionna map
```

## Generate figures

```powershell
cd MOO_Framework
python plot_registered_floor_heatmaps.py     # Per-floor heatmaps
python plot_stacked_registered_heatmaps.py   # 3D stacked overviews
python plot_group_violin.py                 # Violin plots
python plot_uplift.py                       # Uplift bars
```

## Run the six-LLM benchmark

```powershell
cd MOO_Framework
python send_task2.py --dry-run   # Preview
python send_task2.py             # Full 480 tasks
```

The task manifest and detailed run records are already provided under `MOO_Framework/result/six_llm/`.

## IFC2PLY&XML standalone library

```bash
python -m pip install -r MOO_Framework/requirements.txt
python -m ifc2xml_ply.main --inputs frame.ifc enclosure.ifc --frequency-ghz 3.5 --output ./output
```

```
ifc2xml_ply/
├── main.py             ← CLI entry point
├── functions/
│   ├── G1–G7           ← Geometry conversion (7 functions)
│   ├── P1–P3           ← Material property conversion (3 functions)
│   └── W1–W7           ← Wireless propagation conversion (7 functions)
└── data/               ← Material library & IFC samples
```

Source implementation of the IFC-to-PLY/XML conversion functions in [`ifc2xml_ply/`](ifc2xml_ply/).
