# Stage-aware WAP placement with BIM, Sionna RT, and MCP agents

This repository contains the research code for a Modelling-Optimization-Operation (MOO) workflow for wireless access point (WAP) placement during the structural frame and enclosure stages of a building project. The pipeline uses **two MCP servers and eight ordered tools**: four for BIM scene construction and four for WAP optimization. The formal six-LLM benchmark covers 480 tasks (6 models × 2 stages × 5 sensor profiles × 8 scenarios).

## Repository map

| Path | Purpose |
| --- | --- |
| `agents/`, `mcp_servers/`, `tools/` | Planning and execution agents, MCP servers, and the eight tools |
| `baselines/` | Expert-driven and geometric baseline layouts and scoring |
| `data/ifc/` | Frame and enclosure IFC input models |
| `data/radio_materials/` | Material mapping, ITU coefficients, and explicit assumptions |
| `data/regions/` | Current SC1–SC8 priority regions and volume masks |
| `config/*.example.json` | Credential-free model configuration examples |
| `plot_*.py`, `render_floor_bases.py` | Figure generation |
| `result/six_llm/tasks/` | Small formal task-level JSON records retained for audit |
| `docs/` | Legacy project notes and supplementary run-record workbook |

Generated radio maps, working results, logs, and most plots are intentionally excluded from Git. Recreate them by running the pipeline with the supplied input data. The complete local research working directory may contain additional unpublished results and manuscript drafts.

## Setup

Use Python 3.11 with the dependencies in `requirements.txt`. Sionna RT also requires a compatible compute setup; the exact environment depends on the local GPU/driver installation.

```powershell
python -m pip install -r requirements.txt
Copy-Item config/llm_config.example.json config/llm_config.json
Copy-Item config/models.example.json config/models.json
```

Supply credentials through the environment variables or your **local, ignored** JSON files. The example files contain empty `api_key` fields. Never commit actual credentials.

## Run

```powershell
python run_all.py
python send_task_moo.py
python run_experiment_expert.py
python run_experiment_geometric.py
python run_baseline_comparison.py
python plot_uplift.py
```

The full six-model task matrix is launched by `send_task2.py`; inspect its command-line options before running a long or paid experiment. `send_task2.py --dry-run` previews the task plan. The current power profiles SP1–SP5 are 13, 18, 23, 28, and 33 dBm at 3.5 GHz. The PM-versus-baseline comparisons use SP3 and SC1.

## Method and result boundaries

The WAP count search and genetic algorithm use 600 selected receiver positions and a dictionary of 150 candidate WAP sites. A minimum-count proof applies to this finite set. The final layout must satisfy the RSRP, RSRQ, and SINR thresholds at all 600 formal receivers; separate dense-domain probes can still fail. The formal six-LLM completion rate is based on successful eight-tool execution, not dense-domain validation.

The repository includes the formal task JSON records and completion summary. Of 480 assigned tasks, 474 succeeded; six Grok-4.7 tasks failed at the LLM/API layer under the single-attempt policy. See `result/six_llm/completion_rate.json` and `docs/Table_S5_Six_LLM_Run_Records.xlsx` for the detailed audit.

The earlier extended Chinese project notes are preserved in [`docs/legacy_README.md`](docs/legacy_README.md).
