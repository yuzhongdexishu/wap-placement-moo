from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import time
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI
from python_a2a import A2AServer, Message, MessageRole, TextContent, agent, run_server, skill
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

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
# Execution owns live validation and final executable-plan normalization.
# All 8 tools are state_file-driven (the current V17 tool signatures).
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


# ---------------------------------------------------------------------------
# Embedded MCP client hub.
# All computational Agent -> MCP calls pass through this class.
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINTS = {
    "bim_data_processing": os.getenv("BIM_MCP_URL", "http://127.0.0.1:8101/mcp"),
    "wap_layout_optimization": os.getenv("WAP_MCP_URL", "http://127.0.0.1:8102/mcp"),
}


class MCPHub:
    """One execution-side client for the two independent MCP servers."""

    def __init__(self, endpoints: dict[str, str] | None = None) -> None:
        self.endpoints = endpoints or dict(DEFAULT_ENDPOINTS)
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}

    async def __aenter__(self) -> "MCPHub":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for server, url in self.endpoints.items():
            streams = await self._stack.enter_async_context(
                streamablehttp_client(url, timeout=120, sse_read_timeout=7200)
            )
            read_stream, write_stream, _ = streams
            session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            self._sessions[server] = session
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc, tb)
        self._sessions.clear()

    async def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for server, session in self._sessions.items():
            response = await session.list_tools()
            for tool in response.tools:
                tools.append({
                    "server": server,
                    "tool": tool.name,
                    "full_name": f"{server}/{tool.name}",
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                })
        return tools

    @staticmethod
    def _decode_result(result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        items = []
        for part in getattr(result, "content", []) or []:
            text = getattr(part, "text", None)
            if text is None:
                items.append(str(part))
                continue
            try:
                items.append(json.loads(text))
            except Exception:
                items.append(text)
        if len(items) == 1:
            return items[0]
        return items

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        if server not in self._sessions:
            raise KeyError(f"Unknown MCP server: {server}")
        result = await self._sessions[server].call_tool(tool, arguments=args)
        if getattr(result, "isError", False):
            decoded = self._decode_result(result)
            raise RuntimeError(f"MCP tool failed: {server}/{tool}: {decoded}")
        return self._decode_result(result)


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


LOG_ROOT = PROJECT_ROOT / "result" / "logs" / "execution"


def _message_text(message: Message) -> str:
    content = message.content
    return getattr(content, "text", "") if content is not None else ""


def _extract_request(text: str) -> tuple[str, dict[str, Any]]:
    task_match = re.search(
        r"original_task:\s*(.*?)\s*preliminary_plan:\s*```(?:yaml)?\s*(.*?)```",
        text,
        flags=re.S | re.I,
    )
    if task_match:
        original_task = task_match.group(1).strip()
        plan_block = task_match.group(2).strip()
    else:
        plan_match = re.search(r"(?:preliminary_)?plan:\s*```(?:yaml)?\s*(.*?)```", text, flags=re.S | re.I)
        if not plan_match:
            raise ValueError("The request does not contain a preliminary YAML plan.")
        plan_block = plan_match.group(1).strip()
        original_task = ""
    plan = yaml.safe_load(plan_block) or {}
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise ValueError("The preliminary plan does not contain a valid steps list.")
    original_task = original_task or str(plan.get("original_task") or "")
    if not original_task:
        raise ValueError("The original task is missing from the execution request.")
    return original_task, plan


def _refine_plan(
    original_task: str,
    preliminary_plan: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    """Review the preliminary plan against live tools and task conditions before execution.

    LLM usage is mandatory: if no refined plan can be produced the error is raised
    and the task fails (no deterministic fallback plan exists)."""
    contract = build_workflow_contract(original_task, "refined")
    catalog = tool_catalog_text(tools)
    preliminary_yaml = yaml.safe_dump(preliminary_plan, allow_unicode=True, sort_keys=False)
    system = (
        "You are the Execution Agent in a dual-agent BIM-to-WAP system. "
        "Review the Planning Agent's preliminary plan against the live MCP tool catalog, actual task conditions, "
        "engineering constraints, required arguments, file dependencies, and expected outputs. Produce the final "
        "execution sequence before any tool is called. You may improve rationales and record refinement notes, but "
        "must not call Python tool functions directly. Every computational action must use server + tool through MCP. "
        "Return exactly one YAML code block with plan_type=refined, task_analysis, strategy, refinement_notes, and steps."
    )
    prompt = (
        f"Original task:\n{original_task}\n\n"
        f"Live MCP tool catalog:\n{catalog}\n\n"
        f"Planning Agent preliminary plan:\n```yaml\n{preliminary_yaml}\n```\n\n"
        "Check all data dependencies, stage order, paths, required arguments, receiver grids, WAP constraints, "
        "and Sionna surrogate options. Return the refined plan only."
    )
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            settings = _llm_settings(original_task)
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
            refined, _ = normalize_plan(candidate, contract, tools, "refined")
            return refined, f"llm_refined_plan:{settings['model_key']}", raw
        except Exception as exc:
            last_error = exc
            if attempt < 5:
                time.sleep(10 * (2 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def _compact(value: Any, limit: int = 3500) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return text if len(text) <= limit else text[:limit] + f"\n... truncated {len(text) - limit} characters"


def _result_summary(result: Any) -> Any:
    if isinstance(result, dict):
        return {
            key: result.get(key)
            for key in ("status", "server", "tool", "process_file", "summary")
            if key in result
        }
    return str(result)[:1000]


def _write_run_summaries(refined_plan: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_stage.setdefault(str(row.get("stage", "unknown")), []).append(row)

    stage_summaries: list[dict[str, Any]] = []
    conditions = parse_task_conditions(str(refined_plan.get("original_task") or ""))
    for stage, stage_records in by_stage.items():
        run_root = _stage_run_root(conditions, stage)
        run_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "construction_stage": stage,
            "run_root": str(run_root),
            "tool_count": len(stage_records),
            "all_tools_success": all(row.get("status") == "success" for row in stage_records),
            "execution_records": stage_records,
            "final_results_directory": str(run_root / "final_results"),
            "process_directory": str(run_root / "process"),
        }
        (run_root / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        stage_summaries.append(summary)

    overview = {
        "architecture": "Planning Agent -> Execution Agent -> MCP Server -> Tool",
        "refined_strategy": refined_plan.get("strategy"),
        "stage_count": len(stage_summaries),
        "all_stages_success": all(row["all_tools_success"] for row in stage_summaries),
        "stages": [
            {
                "construction_stage": row["construction_stage"],
                "run_root": row["run_root"],
                "tool_count": row["tool_count"],
                "all_tools_success": row["all_tools_success"],
                "process_directory": row["process_directory"],
                "final_results_directory": row["final_results_directory"],
            }
            for row in stage_summaries
        ],
    }
    overview_path = PROJECT_ROOT / "result" / "results_overview.json"
    overview_path.parent.mkdir(parents=True, exist_ok=True)
    overview["overview_file"] = str(overview_path)
    overview_path.write_text(
        json.dumps(overview, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return overview


async def _refine_and_execute(
    original_task: str,
    preliminary_plan: dict[str, Any],
    txt_log: str,
    jsonl_log: str,
) -> dict[str, Any]:
    async with MCPHub() as hub:
        tools = await hub.list_tools()
        refined_plan, refinement_mode, llm_trace = await asyncio.to_thread(
            _refine_plan,
            original_task,
            preliminary_plan,
            tools,
        )
        refined_yaml = yaml.safe_dump(refined_plan, allow_unicode=True, sort_keys=False)
        append_text(txt_log, f"Refined Plan:\n```yaml\n{refined_yaml}\n```")
        append_jsonl(jsonl_log, {
            "type": "refined_plan",
            "refinement_mode": refinement_mode,
            "plan": refined_plan,
            "llm_trace_length": len(llm_trace),
        })

        available = {(str(row["server"]), str(row["tool"])) for row in tools}
        records: list[dict[str, Any]] = []
        completed_ids: set[str] = set()
        failed_step: dict[str, Any] | None = None

        def record_failure(index: int, step_id: str, stage: str, server: str,
                           tool: str, error: str) -> dict[str, Any]:
            """Log a failed step, append a failed record, and capture the failed_step."""
            nonlocal failed_step
            failed_step = {
                "step_id": step_id,
                "index": index,
                "stage": stage,
                "server": server,
                "tool": tool,
                "error": error,
            }
            append_text(txt_log, f"Error #{index}: {error}")
            append_jsonl(jsonl_log, {
                "type": "error",
                "index": index,
                "step_id": step_id,
                "stage": stage,
                "server": server,
                "tool": tool,
                "error": error,
                "timestamp": time.time(),
            })
            records.append({
                "index": index,
                "step_id": step_id,
                "stage": stage,
                "server": server,
                "tool": tool,
                "status": "failed",
                "error": error,
            })
            return failed_step

        for index, step in enumerate(refined_plan["steps"], start=1):
            step_id = str(step["step_id"])
            stage = str(step["stage"])
            server = str(step["server"])
            tool = str(step["tool"])
            args = dict(step.get("args") or {})
            dependencies = [str(item) for item in step.get("depends_on") or []]
            why = str(step.get("why") or f"Call {server}/{tool}")

            missing_dependencies = [item for item in dependencies if item not in completed_ids]
            if missing_dependencies:
                record_failure(index, step_id, stage, server, tool,
                               f"incomplete dependencies: {missing_dependencies}")
                break
            if (server, tool) not in available:
                record_failure(index, step_id, stage, server, tool,
                               f"unknown MCP tool: {server}/{tool}")
                break

            started = time.time()
            append_text(txt_log, f"Step rationale #{index} [{step_id}]: {why}")
            append_text(txt_log, f"MCP call #{index}: {server}/{tool}({_compact(args, 1800)})")
            append_jsonl(jsonl_log, {
                "type": "mcp_call",
                "index": index,
                "step_id": step_id,
                "stage": stage,
                "server": server,
                "tool": tool,
                "args": args,
                "depends_on": dependencies,
                "why": why,
                "timestamp": started,
            })

            try:
                # This is the only computational invocation path used by the agent.
                result = await hub.call(server, tool, args)
            except Exception as exc:
                record_failure(index, step_id, stage, server, tool,
                               f"{type(exc).__name__}: {exc}")
                break

            elapsed = time.time() - started
            summary = _result_summary(result)
            append_text(txt_log, f"Tool result #{index}: {_compact(summary)}")
            append_jsonl(jsonl_log, {
                "type": "tool_result",
                "index": index,
                "step_id": step_id,
                "stage": stage,
                "server": server,
                "tool": tool,
                "result_summary": summary,
                "elapsed_seconds": elapsed,
                "timestamp": time.time(),
            })
            records.append({
                "index": index,
                "step_id": step_id,
                "stage": stage,
                "server": server,
                "tool": tool,
                "status": "success",
                "elapsed_seconds": elapsed,
                "result_summary": summary,
            })
            completed_ids.add(step_id)

    overview = _write_run_summaries(refined_plan, records)
    status = "failed" if failed_step is not None else "success"
    return {
        "status": status,
        "original_task": original_task,
        "preliminary_plan": preliminary_plan,
        "refined_plan": refined_plan,
        "refinement_mode": refinement_mode,
        "failed_step": failed_step,
        "aggregated_results": {
            "step_count": len(records),
            "all_steps_success": failed_step is None,
            "overview_file": overview.get("overview_file", str(PROJECT_ROOT / "result" / "results_overview.json")),
            "stage_summary": overview["stages"],
        },
        "execution_records": records,
    }


@agent(
    name="Execution Agent",
    description=(
        "Reviews the preliminary workflow against live MCP tools and task conditions, creates the final execution "
        "sequence, invokes all computations through MCP servers, and returns results and execution records."
    ),
    version="3.0.0",
)
class ExecutionAgent(A2AServer):
    @skill(
        name="refine_and_execute",
        description="Refine a preliminary plan against live MCP tools and execute it exclusively through MCPHub.",
        tags=["execution", "MCP", "Sionna"],
    )
    def refine_and_execute(self, request_text: str) -> dict[str, Any]:
        return self._process_request(request_text)

    def _process_request(self, text: str) -> dict[str, Any]:
        txt_log, jsonl_log = make_log_paths(LOG_ROOT, text or "empty_execution_request")
        append_text(txt_log, "=== Execution Agent Start ===")
        append_jsonl(jsonl_log, {"type": "meta", "event": "execution_start", "timestamp": time.time()})

        try:
            original_task, preliminary_plan = _extract_request(text)
            preliminary_yaml = yaml.safe_dump(preliminary_plan, allow_unicode=True, sort_keys=False)
            append_text(txt_log, f"Preliminary Plan Received:\n```yaml\n{preliminary_yaml}\n```")
            append_jsonl(jsonl_log, {"type": "preliminary_plan_received", "plan": preliminary_plan})
            result = asyncio.run(
                _refine_and_execute(original_task, preliminary_plan, txt_log, jsonl_log)
            )
            result["logs"] = {"txt": txt_log, "jsonl": jsonl_log}
            append_text(txt_log, "=== Execution Agent End ===")
            append_jsonl(jsonl_log, {
                "type": "meta",
                "event": "execution_end",
                "status": "success",
                "timestamp": time.time(),
            })
        except Exception as exc:
            result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_step": None,
                "execution_records": [],
                "logs": {"txt": txt_log, "jsonl": jsonl_log},
            }
            append_text(txt_log, f"Final Error: {result['error']}")
            append_jsonl(jsonl_log, {
                "type": "meta",
                "event": "execution_end",
                "status": "failed",
                "error": result["error"],
                "timestamp": time.time(),
            })
        return result

    def handle_message(self, message: Message) -> Message:
        result = self.refine_and_execute(_message_text(message))
        return Message(
            content=TextContent(text=json.dumps(result, ensure_ascii=False, indent=2, default=str)),
            role=MessageRole.AGENT,
            parent_message_id=message.message_id,
            conversation_id=message.conversation_id,
        )


if __name__ == "__main__":
    run_server(
        ExecutionAgent(url="http://127.0.0.1:7002/a2a", google_a2a_compatible=False),
        host=os.getenv("EXECUTION_AGENT_HOST", "127.0.0.1"),
        port=int(os.getenv("EXECUTION_AGENT_PORT", "7002")),
    )
