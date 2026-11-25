# MOO framework and comparison experiments

This directory keeps the runnable research project at a single project root so its relative paths and Python imports work together. It includes the two MCP servers and eight ordered tools, the six-LLM task runner, and the expert and geometric comparison methods.

## Layout

| Path | Purpose |
| --- | --- |
| `agents/`, `mcp_servers/`, `tools/` | Planning and execution agents, MCP servers, and BIM/WAP tools |
| `baselines/` | Expert and geometric layouts, geometry, and scoring |
| `data/` | IFC inputs, radio materials, priority regions, and visualization bases |
| `config/*.example.json` | Credential-free configuration examples |
| `result/six_llm/` | Formal task records, task manifest, and completion summary |
| `result/comparison_table.csv` | MOO versus baseline comparison summary |

## Setup and run

Run from this directory with Python 3.11 and a compatible Sionna RT installation:

```powershell
python -m pip install -r requirements.txt
Copy-Item config/llm_config.example.json config/llm_config.json
Copy-Item config/models.example.json config/models.json
python run_all.py
python send_task_moo.py
python run_experiment_expert.py
python run_experiment_geometric.py
python run_baseline_comparison.py
```

Add the regular-run DeepSeek credential to `config/llm_config.json` as `api_key`, or set `DEEPSEEK_API_KEY` (the environment variable takes precedence). For the six-LLM benchmark, add each model credential to the matching `api_key` field in `config/models.json` and verify its `model_name` and `base_url`. Both local JSON files are ignored by Git; leave the `.example.json` templates empty. `send_task2.py --dry-run` previews the six-LLM task matrix before a long or paid run.

The formal optimization checks RSRP, RSRQ, and SINR at 600 selected receiver positions and searches 150 candidate WAP sites. Its minimum-count claim is limited to that finite candidate and receiver set. Separate dense-domain checks may have failures. The six-LLM completion metric measures successful execution of the eight-tool sequence, not dense-domain validation.

Of the 480 formal six-LLM tasks, 474 succeeded and six Grok-4.7 tasks failed at the LLM/API layer under the single-attempt policy. The detailed workbook is in [`../Supplementary_Tables/S5/`](../Supplementary_Tables/S5/).
