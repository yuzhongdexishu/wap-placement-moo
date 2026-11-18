from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import time
import yaml
from openai import OpenAI
from python_a2a import A2AServer, Message, MessageRole, TextContent, agent, run_server, skill

# Windows: keep the selector event-loop policy required by the MCP stack under load.
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except RuntimeError:
        pass

# ---------------------------------------------------------------------------
# Embedded logging helpers
# ---------------------------------------------------------------------------
def _safe_token(text: str, limit: int = 48) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    if not token:
        token = "task"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{token[:limit]}_{digest}"


def make_log_paths(log_root: str | Path, task_text: str) -> tuple[str, str]:
    root = Path(log_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{_safe_token(task_text)}"
    return str(root / f"{stem}.txt"), str(root / f"{stem}.jsonl")


def append_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Embedded deterministic workflow contract.
# Planning owns task parsing and preliminary-plan construction. All 8 tools
# are state_file-driven (the current V17 tool signatures).
# ---------------------------------------------------------------------------
SERVER_ORDER = (
    "bim_data_processing",
    "wap_layout_optimization",
)

TOOL_SEQUENCE = (
    ("bim_data_processing", "parse_ifc_building"),
    ("bim_data_processing", "tessellate_ifc_objects"),
    ("bim_data_processing", "export_object_meshes"),
    ("bim_data_processing", "generate_mitsuba_scene"),
    ("wap_layout_optimization", "define_optimization_problem"),
    ("wap_layout_optimization", "build_sionna_surrogate"),
    ("wap_layout_optimization", "generate_and_evaluate_candidate_layouts"),
    ("wap_layout_optimization", "optimize_wap_layout"),

)

BENCHMARK_MARKER = "WAP_LLM_COMPLETION_TASK"


def extract_yaml_block(text: str) -> str:
    match = re.search(r"```(?:yaml)?\s*(.*?)```", text, flags=re.S | re.I)
    return match.group(1).strip() if match else text.strip()


def _benchmark_payload(task_text: str) -> dict[str, Any] | None:
    """Return the benchmark JSON payload, or None for ordinary task texts."""
    if BENCHMARK_MARKER not in task_text:
        return None
    json_start = task_text.find("{", task_text.find(BENCHMARK_MARKER))
    strict_pos = task_text.find("\nSTRICT_REQUIREMENTS", json_start)
    if json_start < 0 or strict_pos < 0:
        raise ValueError("Benchmark task JSON block is missing")
    payload = json.loads(task_text[json_start:strict_pos].strip())
    return payload if isinstance(payload, dict) else None


def task_stages(task_text: str) -> list[str]:
    lower = task_text.lower()
    has_frame = "frame" in lower
    has_enclosure = "enclosure" in lower
    if has_frame and not has_enclosure:
        return ["frame"]
    if has_enclosure and not has_frame:
        return ["enclosure"]
    return ["frame", "enclosure"]


def task_scope(task_text: str) -> str:
    lower = task_text.lower()
    if "only" in lower:
        if any(token in lower for token in ("parse", "ifc", "mesh", "bim")):
            if not any(token in lower for token in ("ga", "optimization", "ray", "sionna", "audit")):
                return "bim"
        if any(token in lower for token in ("ga", "optimization")):
            if not any(token in lower for token in ("ray", "sionna", "audit")):
                return "optimization"
    return "full"


def _int_option(task_text: str, names: list[str], default: int) -> int:
    for name in names:
        match = re.search(rf"\b{re.escape(name)}\s*[=:：]?\s*(\d+)", task_text, flags=re.I)
        if match:
            return int(match.group(1))
    return default


def _float_option(task_text: str, names: list[str], default: float) -> float:
    for name in names:
        match = re.search(rf"\b{re.escape(name)}\s*[=:：]?\s*(-?\d+(?:\.\d+)?)", task_text, flags=re.I)
        if match:
            return float(match.group(1))
    return default


def _bool_option(task_text: str, names: list[str], default: bool) -> bool:
    lower = task_text.lower()
    for name in names:
        match = re.search(rf"\b{re.escape(name.lower())}\s*[=:：]?\s*(true|false|yes|no|1|0)", lower, flags=re.I)
        if match:
            return match.group(1).lower() in {"true", "yes", "1"}
    return default


def parse_task_conditions(task_text: str) -> dict[str, Any]:
    """Resolve the executable conditions of a task.

    Benchmark tasks (WAP_LLM_COMPLETION_TASK) carry model_key, require_llm,
    run_root_path, and a single construction stage. Ordinary tasks use the
    English tokens in the natural-language text. The sensor profile and the
    smart scenario are real execution conditions: they are passed through to
    define_optimization_problem, so the GA and final results simulate the
    task-selected profile/scenario instead of a fixed SP3/SC1 default.
    """
    sensor_match = re.search(r"\bSP([1-5])\b", task_text, flags=re.I)
    scenario_match = re.search(r"\bSC(10|[1-9])\b", task_text, flags=re.I)
    payload = _benchmark_payload(task_text)

    if payload is not None:
        raw_stage = str(payload.get("construction_stage") or "").strip().lower()
        stage = "enclosure" if raw_stage in {"envelope", "enclosure"} else ("frame" if raw_stage == "frame" else "")
        md = _int_option(task_text, ["max_depth", "max-depth"], 3)
        return {
            "stages": [stage] if stage else task_stages(task_text),
            "scope": "full",
            "sensor_profile": str(payload.get("sensor_profile")
                                  or (f"SP{sensor_match.group(1)}" if sensor_match else "SP3")),
            "smart_scenario": str(payload.get("scenario")
                                  or (f"SC{scenario_match.group(1)}" if scenario_match else "SC1")),
            "run_root_path": str(payload.get("run_root_path") or "").strip(),
            "model_key": str(payload.get("model_key") or "").strip(),
            "require_llm": bool(payload.get("require_llm", False)),
            "population_size": _int_option(task_text, ["population", "population_size"], 80),
            "generations": _int_option(task_text, ["generations"], 35),
            "minimum_wap_distance_m": _float_option(task_text, ["minimum_wap_distance_m"], 4.0),
            "min_rsrp_dbm": _float_option(task_text, ["min_rsrp_dbm"], -140.0),
            "min_rsrq_db": _float_option(task_text, ["min_rsrq_db"], -19.5),
            "min_sinr_db": _float_option(task_text, ["min_sinr_db"], 0.0),
            "random_seed": int(payload.get("random_seed") or _int_option(task_text, ["random_seed"], 20260717)),
            "max_depth": md,
            "los": _bool_option(task_text, ["los"], True),
            "specular_reflection": _bool_option(task_text, ["specular_reflection", "reflection"], True),
            "diffraction": _bool_option(task_text, ["diffraction"], True),
            "refraction": _bool_option(task_text, ["refraction"], True),
            "ga_max_depth": _int_option(task_text, ["ga_max_depth"], md),
            "ga_samples_per_tx": _int_option(task_text, ["ga_samples_per_tx", "samples_per_tx"], 1000000),
                    }

    run_root_match = re.search(r"(?:run_root_path|result_root)\s*[=:]\s*(\S+)", task_text, flags=re.I)
    md = _int_option(task_text, ["max_depth", "max-depth"], 3)
    return {
        "stages": task_stages(task_text),
        "scope": task_scope(task_text),
        "sensor_profile": f"SP{sensor_match.group(1)}" if sensor_match else "SP3",
        "smart_scenario": f"SC{scenario_match.group(1)}" if scenario_match else "SC1",
        "run_root_path": run_root_match.group(1) if run_root_match else "",
        "model_key": "",
        "require_llm": False,
        "population_size": _int_option(task_text, ["population", "population_size"], 80),
        "generations": _int_option(task_text, ["generations"], 35),
        "minimum_wap_distance_m": _float_option(task_text, ["minimum_wap_distance_m"], 4.0),
        "min_rsrp_dbm": _float_option(task_text, ["min_rsrp_dbm"], -140.0),
        "min_rsrq_db": _float_option(task_text, ["min_rsrq_db"], -19.5),
        "min_sinr_db": _float_option(task_text, ["min_sinr_db"], 0.0),
        "random_seed": _int_option(task_text, ["random_seed"], 20260717),
        "max_depth": md,
        "los": _bool_option(task_text, ["los"], True),
        "specular_reflection": _bool_option(task_text, ["specular_reflection", "reflection"], True),
        "diffraction": _bool_option(task_text, ["diffraction"], True),
        "refraction": _bool_option(task_text, ["refraction"], True),
        "ga_max_depth": _int_option(task_text, ["ga_max_depth"], md),
        "ga_samples_per_tx": _int_option(task_text, ["ga_samples_per_tx", "samples_per_tx"], 1000000),
            }


def _stage_run_root(conditions: dict[str, Any], stage: str) -> Path:
    configured = str(conditions.get("run_root_path") or "").strip()
    if configured:
        base = Path(configured)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base / stage
    return PROJECT_ROOT / "result" / stage


def _stage_steps(stage: str, conditions: dict[str, Any]) -> list[dict[str, Any]]:
    run_root = _stage_run_root(conditions, stage)
    ifc_file = PROJECT_ROOT / "data" / "ifc" / f"{stage}.ifc"
    state_file = run_root / "workspace" / "bim" / "pipeline_state.json"

    raw: list[tuple[str, str, dict[str, Any], str]] = [
        (
            "bim_data_processing",
            "parse_ifc_building",
            {
                "ifc_file_path": str(ifc_file),
                "project_root_path": str(run_root),
                "construction_stage": stage,
            },
            f"Parse the {stage} IFC model and initialize the pipeline state.",
        ),
        (
            "bim_data_processing",
            "tessellate_ifc_objects",
            {"state_file": str(state_file)},
            "Tessellate the represented IFC products into independent triangle meshes.",
        ),
        (
            "bim_data_processing",
            "export_object_meshes",
            {"state_file": str(state_file)},
            "Export one PLY file for each successfully tessellated object.",
        ),
        (
            "bim_data_processing",
            "generate_mitsuba_scene",
            {"state_file": str(state_file)},
            "Generate and validate the Sionna-compatible Mitsuba scene.",
        ),
        (
            "wap_layout_optimization",
            "define_optimization_problem",
            {
                "state_file": str(state_file),
                "population_size": conditions["population_size"],
                "generations": conditions["generations"],
                "crossover_probability": 0.75,
                "mutation_probability": 0.30,
                "minimum_wap_distance_m": conditions["minimum_wap_distance_m"],
                "random_seed": conditions["random_seed"],
                "sensor_profile": conditions["sensor_profile"],
                "smart_scenario": conditions["smart_scenario"],
                "min_rsrp_dbm": conditions["min_rsrp_dbm"],
                "min_rsrq_db": conditions["min_rsrq_db"],
                "min_sinr_db": conditions["min_sinr_db"],
            },
            "Define the GA problem with the task-selected sensor profile and smart scenario "
            "(the scenario's zones get the priority weight; all other zones weight 1).",
        ),
        (
            "wap_layout_optimization",
            "build_sionna_surrogate",
            {
                "state_file": str(state_file),
                "max_depth": conditions["ga_max_depth"],
                "los": conditions["los"],
                "specular_reflection": conditions["specular_reflection"],
                "diffraction": conditions["diffraction"],
                "refraction": conditions["refraction"],
                "samples_per_tx": conditions["ga_samples_per_tx"],
            },
            "Use Sionna to build the 600-receiver RadioMapSolver surrogate.",
        ),
        (
            "wap_layout_optimization",
            "generate_and_evaluate_candidate_layouts",
            {"state_file": str(state_file)},
            "Solve the full-coverage WAP count and generate the GA population.",
        ),
        (
            "wap_layout_optimization",
            "optimize_wap_layout",
            {"state_file": str(state_file)},
            "Refine the WAP coordinates through selection, crossover, mutation, and constraint repair.",
        ),
    ]

    limits = {"bim": 4, "optimization": 8, "full": 8}
    selected = raw[: limits[conditions["scope"]]]
    steps: list[dict[str, Any]] = []
    previous_id: str | None = None
    for index, (server, tool, args, why) in enumerate(selected, start=1):
        step_id = f"{stage}_{index:02d}"
        steps.append({
            "step_id": step_id,
            "stage": stage,
            "server": server,
            "tool": tool,
            "args": args,
            "depends_on": [previous_id] if previous_id else [],
            "why": why,
        })
        previous_id = step_id
    return steps


def build_workflow_contract(task_text: str, plan_type: str = "preliminary") -> dict[str, Any]:
    conditions = parse_task_conditions(task_text)
    steps: list[dict[str, Any]] = []
    for stage in conditions["stages"]:
        steps.extend(_stage_steps(stage, conditions))

    required_inputs = [
        str(PROJECT_ROOT / "data" / "ifc" / f"{stage}.ifc")
        for stage in conditions["stages"]
    ]
    expected_outputs = []
    for stage in conditions["stages"]:
        run_root = _stage_run_root(conditions, stage)
        expected_outputs.extend([
            str(run_root / "process"),
            str(run_root / "final_results"),
            str(run_root / "run_summary.json"),
        ])

    return {
        "schema_version": "2.1",
        "plan_type": plan_type,
        "original_task": task_text,
        "task_analysis": {
            "required_inputs": required_inputs,
            "engineering_constraints": [
                f"The {conditions['sensor_profile']} sensor profile (SP1-SP5: 13/18/23/28/33 dBm TX power) drives the Sionna surrogate and exported results",
                f"The {conditions['smart_scenario']} construction-scenario zones carry the priority weight 2; every other zone carries weight 1",
                "GA receiver set: exactly 600 points in the current scenario valid domains",
                "Every one of the 600 receivers must have a valid surrogate link",
                "Search WAP counts from N=1 and select the first layout meeting all 600 RF quality thresholds",
                f"Minimum WAP spacing: {conditions['minimum_wap_distance_m']} m",
                "Tools must be invoked through MCP servers; direct tool imports are forbidden in agents",
            ],
            "expected_outputs": expected_outputs,
            "resolved_conditions": conditions,
            "assumptions": [
                "The named IFC files exist under data/ifc.",
                "The two MCP servers are available before either agent processes the task.",
                "Sionna RT is installed and builds the GA surrogate.",
            ],
        },
        "strategy": "Planning Agent preliminary decomposition -> Execution Agent refinement -> MCP-only tool execution",
        "steps": steps,
    }


def tool_catalog_text(tools: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in tools:
        schema = row.get("input_schema") or {}
        properties = schema.get("properties") or {}
        required = schema.get("required") or []
        args = ", ".join(
            f"{name}{'*' if name in required else ''}"
            for name in properties
        )
        lines.append(
            f"- server={row.get('server')} tool={row.get('tool')} args=[{args}] "
            f"description={row.get('description', '')}"
        )
    return "\n".join(lines)


def _candidate_step_map(candidate: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str, str], dict[str, Any]] = {}
    for step in candidate.get("steps") or []:
        if not isinstance(step, dict):
            continue
        key = (str(step.get("stage", "")), str(step.get("server", "")), str(step.get("tool", "")))
        if all(key):
            mapping[key] = step
    return mapping


def normalize_plan(
    candidate: dict[str, Any] | None,
    contract: dict[str, Any],
    available_tools: list[dict[str, Any]],
    plan_type: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate an LLM plan against the actual MCP contract and repair unsafe fields.

    The LLM controls task analysis, rationale, and refinement notes. Concrete file paths,
    required arguments, tool order, and dependencies are restored from the deterministic
    workflow contract so the plan remains executable.
    """
    candidate = candidate if isinstance(candidate, dict) else {}
    available = {(str(row.get("server")), str(row.get("tool"))) for row in available_tools}
    canonical_steps = contract["steps"]
    candidate_map = _candidate_step_map(candidate)
    notes: list[str] = []
    normalized_steps: list[dict[str, Any]] = []

    for canonical in canonical_steps:
        pair = (canonical["server"], canonical["tool"])
        if pair not in available:
            raise ValueError(f"Required MCP tool is unavailable: {pair[0]}/{pair[1]}")
        key = (canonical["stage"], canonical["server"], canonical["tool"])
        proposed = candidate_map.get(key, {})
        normalized = copy.deepcopy(canonical)
        if proposed.get("why"):
            normalized["why"] = str(proposed["why"])
        normalized_steps.append(normalized)

    proposed_count = len(candidate.get("steps") or [])
    if proposed_count != len(canonical_steps):
        notes.append(
            f"The LLM proposed {proposed_count} steps; the executable contract restored {len(canonical_steps)} required steps."
        )

    task_analysis = copy.deepcopy(contract["task_analysis"])
    proposed_analysis = candidate.get("task_analysis")
    if isinstance(proposed_analysis, dict):
        for key in ("required_inputs", "engineering_constraints", "expected_outputs", "assumptions"):
            value = proposed_analysis.get(key)
            if isinstance(value, list) and value:
                task_analysis[key] = value
        task_analysis["resolved_conditions"] = contract["task_analysis"]["resolved_conditions"]

    result = {
        "schema_version": "2.1",
        "plan_type": plan_type,
        "original_task": contract["original_task"],
        "task_analysis": task_analysis,
        "strategy": str(candidate.get("strategy") or contract["strategy"]),
        "steps": normalized_steps,
    }
    refinement_notes = candidate.get("refinement_notes")
    if isinstance(refinement_notes, list):
        notes.extend(str(item) for item in refinement_notes)
    if plan_type == "refined":
        result["refinement_notes"] = notes or [
            "Verified every server/tool pair against the live MCP catalog.",
            "Restored mandatory arguments and explicit data dependencies from the workflow contract.",
        ]
    return result, notes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LLM_CONFIG_PATH = PROJECT_ROOT / "config" / "llm_config.json"
MODELS_CONFIG_PATH = PROJECT_ROOT / "config" / "models.json"


def _model_configs() -> list[dict[str, Any]]:
    if not MODELS_CONFIG_PATH.is_file():
        return []
    try:
        payload = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("models") if isinstance(payload, dict) else payload
    return rows if isinstance(rows, list) else []


def _task_model_key(task_text: str) -> str:
    payload = _benchmark_payload(task_text)
    key = str((payload or {}).get("model_key") or "").strip()
    if key:
        return key
    match = re.search(r"model_key\s*[=:]\s*([A-Za-z0-9_-]+)", task_text, flags=re.I)
    return match.group(1) if match else ""


def _llm_settings(task_text: str) -> dict[str, Any]:
    """Resolve the LLM settings for the task.

    Benchmark tasks carry model_key and select one of the six configured LLMs
    from config/models.json. Ordinary tasks use config/llm_config.json.
    """
    model_key = _task_model_key(task_text)
    if model_key:
        for cfg in _model_configs():
            if str(cfg.get("key") or "") == model_key:
                api_key = str(cfg.get("api_key") or "").strip()
                base_url = str(cfg.get("base_url") or "").strip()
                model = str(cfg.get("model_name") or cfg.get("model") or "").strip()
                if not api_key or not base_url or not model:
                    raise ValueError(f"Model '{model_key}' is incomplete in config/models.json")
                return {"api_key": api_key, "base_url": base_url, "model": model, "model_key": model_key}
        raise ValueError(f"Unknown model_key '{model_key}' (not in config/models.json)")

    config: dict[str, Any] = {}
    if LLM_CONFIG_PATH.is_file():
        try:
            config = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            config = {}
    env_name = str(config.get("api_key_env") or "DEEPSEEK_API_KEY")
    api_key = os.getenv(env_name, "").strip() or str(config.get("api_key") or "").strip()
    base_url = str(config.get("base_url") or "https://api.deepseek.com").strip()
    model = str(config.get("model") or "deepseek-reasoner").strip()
    return {"api_key": api_key, "base_url": base_url, "model": model,
            "model_key": str(config.get("provider") or "deepseek")}


EXEC_AGENT_URL = os.getenv("EXECUTION_AGENT_URL", "http://127.0.0.1:7002/a2a")
LOG_ROOT = PROJECT_ROOT / "result" / "logs" / "planning"


def _message_text(message: Message) -> str:
    content = message.content
    return getattr(content, "text", "") if content is not None else ""


def _planned_tools_from_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the planning-time tool catalog from the deterministic workflow contract.

    Planning deliberately does not connect to MCP. Live MCP discovery and validation
    are centralized in the Execution Agent immediately before execution.
    """
    catalog: dict[tuple[str, str], dict[str, Any]] = {}
    for step in contract.get("steps") or []:
        server = str(step.get("server") or "")
        tool = str(step.get("tool") or "")
        if not server or not tool:
            continue
        args = dict(step.get("args") or {})
        catalog[(server, tool)] = {
            "server": server,
            "tool": tool,
            "full_name": f"{server}/{tool}",
            "description": str(step.get("why") or ""),
            "input_schema": {
                "properties": {name: {} for name in args},
                "required": list(args),
            },
        }
    return list(catalog.values())


def _generate_preliminary_plan(
    task_text: str,
    tools: list[dict[str, Any]],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    """Use the assigned LLM to identify inputs, constraints, outputs, and an ordered
    preliminary plan. LLM usage is mandatory: if no LLM plan can be produced the
    error is raised and the task fails (no deterministic fallback plan exists)."""
    catalog = tool_catalog_text(tools)
    contract_yaml = yaml.safe_dump(contract, allow_unicode=True, sort_keys=False)
    system = (
        "You are the Planning Agent in a dual-agent BIM-to-WAP system. "
        "Identify required input data, engineering constraints, expected outputs, and data dependencies. "
        "Then decompose the request into an ordered preliminary MCP plan. "
        "You may only use server/tool pairs from the supplied live MCP catalog. "
        "Return exactly one YAML code block with: schema_version, plan_type=preliminary, original_task, "
        "task_analysis, strategy, and steps. Each step must contain step_id, stage, server, tool, args, "
        "depends_on, and why. Do not execute tools and do not add prose outside the YAML block."
    )
    prompt = (
        f"Natural-language task:\n{task_text}\n\n"
        f"Live MCP tool catalog:\n{catalog}\n\n"
        "The following deterministic workflow contract documents the correct file paths, "
        "arguments, stage order, and data dependencies. Improve the task analysis and rationale without inventing "
        "new tools or breaking dependencies.\n\n"
        f"Workflow contract:\n```yaml\n{contract_yaml}\n```"
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            settings = _llm_settings(task_text)
            if not settings["api_key"]:
                raise RuntimeError(f"No API key configured for model '{settings['model_key']}'")
            client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"])
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            if attempt > 1:
                messages.append({"role": "user", "content": (
                    f"Your previous answer could not be parsed as YAML ({type(last_error).__name__}: {last_error}). "
                    "Return exactly one valid YAML code block."
                )})
            response = client.chat.completions.create(model=settings["model"], messages=messages)
            if response is None:
                raise RuntimeError("LLM API returned an empty response (transient failure)")
            raw = response.choices[0].message.content or ""
            candidate = yaml.safe_load(extract_yaml_block(raw)) or {}
            plan, _ = normalize_plan(candidate, contract, tools, "preliminary")
            return plan, f"llm_preliminary_plan:{settings['model_key']}", raw
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(10 * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def _post_to_execution_agent(task_text: str, preliminary_plan: dict[str, Any]) -> dict[str, Any]:
    plan_yaml = yaml.safe_dump(preliminary_plan, allow_unicode=True, sort_keys=False)
    payload = {
        "role": "user",
        "content": {
            "type": "text",
            "text": (
                "Review, refine, and execute this preliminary MCP workflow.\n\n"
                f"original_task:\n{task_text}\n\n"
                f"preliminary_plan:\n```yaml\n{plan_yaml}\n```"
            ),
        },
    }
    response = requests.post(EXEC_AGENT_URL, json=payload, timeout=21600)
    response.raise_for_status()
    body = response.json()
    text = body.get("content", {}).get("text")
    if not text:
        return {"status": "failed", "error": "Execution Agent returned no text content.", "raw": body}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"status": "failed", "raw": parsed}
    except json.JSONDecodeError:
        return {"status": "failed", "error": "Execution Agent returned non-JSON content.", "raw": text}


@agent(
    name="Planning Agent",
    description=(
        "Interprets the natural-language request, identifies inputs/constraints/outputs, creates a preliminary "
        "MCP plan, dispatches it to the Execution Agent, and returns the refined workflow and results."
    ),
    version="3.0.0",
)
class PlanningAgent(A2AServer):
    @skill(
        name="plan_and_dispatch",
        description="Generate a preliminary dependency-aware MCP plan and dispatch it to the Execution Agent.",
        tags=["planning", "MCP", "BIM", "WAP"],
    )
    def plan_and_dispatch(self, task_text: str) -> dict[str, Any]:
        return self._process_task(task_text)

    def _process_task(self, task_text: str) -> dict[str, Any]:
        task_text = task_text.strip()
        txt_log, jsonl_log = make_log_paths(LOG_ROOT, task_text or "empty_task")
        append_text(txt_log, "=== Planning Agent Start ===")
        append_jsonl(jsonl_log, {"type": "meta", "event": "planning_start"})

        try:
            if not task_text:
                raise ValueError("The task text is empty.")
            contract = build_workflow_contract(task_text, "preliminary")
            tools = _planned_tools_from_contract(contract)
            preliminary_plan, planning_mode, llm_trace = _generate_preliminary_plan(task_text, tools, contract)
            preliminary_yaml = yaml.safe_dump(preliminary_plan, allow_unicode=True, sort_keys=False)
            append_text(txt_log, f"Preliminary Plan:\n```yaml\n{preliminary_yaml}\n```")
            append_jsonl(jsonl_log, {
                "type": "preliminary_plan",
                "planning_mode": planning_mode,
                "plan": preliminary_plan,
            })

            execution_response = _post_to_execution_agent(task_text, preliminary_plan)
            status = "success" if execution_response.get("status") == "success" else "failed"
            result = {
                "status": status,
                "original_task": task_text,
                "planning_mode": planning_mode,
                "preliminary_plan": preliminary_plan,
                "refined_plan": execution_response.get("refined_plan"),
                "refinement_mode": execution_response.get("refinement_mode"),
                "failed_step": execution_response.get("failed_step"),
                "aggregated_results": execution_response.get("aggregated_results"),
                "execution_records": execution_response.get("execution_records"),
                "execution_logs": execution_response.get("logs"),
                "planning_logs": {"txt": txt_log, "jsonl": jsonl_log},
            }
            if status != "success":
                result["execution_error"] = execution_response.get("error") or execution_response
            append_text(txt_log, f"Execution Agent Status: {execution_response.get('status')}")
            append_jsonl(jsonl_log, {
                "type": "execution_response",
                "status": execution_response.get("status"),
                "aggregated_results": execution_response.get("aggregated_results"),
                "logs": execution_response.get("logs"),
            })
            append_jsonl(jsonl_log, {
                "type": "planning_llm_trace",
                "mode": planning_mode,
                "trace_length": len(llm_trace),
            })
        except Exception as exc:
            result = {
                "status": "failed",
                "original_task": task_text,
                "error": f"{type(exc).__name__}: {exc}",
                "failed_step": None,
                "planning_logs": {"txt": txt_log, "jsonl": jsonl_log},
            }
            append_text(txt_log, f"Planning Error: {result['error']}")
            append_jsonl(jsonl_log, {"type": "error", "error": result["error"]})

        append_text(txt_log, "=== Planning Agent End ===")
        append_jsonl(jsonl_log, {"type": "meta", "event": "planning_end", "status": result["status"]})
        return result

    def handle_message(self, message: Message) -> Message:
        result = self.plan_and_dispatch(_message_text(message))
        return Message(
            content=TextContent(text=json.dumps(result, ensure_ascii=False, indent=2, default=str)),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )


if __name__ == "__main__":
    run_server(
        PlanningAgent(url="http://127.0.0.1:7001/a2a", google_a2a_compatible=False),
        host=os.getenv("PLANNING_AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("PLANNING_AGENT_PORT", "7001")),
    )
