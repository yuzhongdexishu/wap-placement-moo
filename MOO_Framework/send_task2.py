from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

# Six-LLM completion benchmark, V17 edition. Self-contained: the 480-task matrix
# (6 models x 2 stages x 5 sensor profiles x 8 scenarios), task dispatch to the
# V17 Planning Agent, per-task JSON results, and the per-model completion-rate
# summary. Model credentials are read from config/models.json.

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_CONFIG_PATH = PROJECT_ROOT / "config" / "models.json"
RESULTS_ROOT = PROJECT_ROOT / "result" / "six_llm"

STAGES = ("frame", "enclosure")
SENSOR_PROFILES = ("SP1", "SP2", "SP3", "SP4", "SP5")
from tools.wap_layout_optimization.define_problem import get_sensor_profile
from tools.wap_layout_optimization.define_problem import available_scenarios
BASE_RANDOM_SEED = 20260727


def load_models() -> list[dict[str, Any]]:
    if not MODELS_CONFIG_PATH.is_file():
        raise SystemExit(f"Missing {MODELS_CONFIG_PATH}; populate it before running the six-LLM benchmark.")
    try:
        payload = json.loads(MODELS_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Cannot read {MODELS_CONFIG_PATH}: {exc}")
    rows = payload.get("models") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != 6:
        raise SystemExit(f"{MODELS_CONFIG_PATH} must contain exactly 6 models")
    return rows


def build_task_matrix(*, repetitions: int = 1, backend: str = "sionna", require_llm: bool = True) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    tasks: list[dict[str, Any]] = []
    for model in load_models():
        for stage in STAGES:
            for sensor_key in SENSOR_PROFILES:
                for scenario_key in available_scenarios(stage):
                    for repetition in range(1, repetitions + 1):
                        task_id = f"{model['key']}__{stage}__{sensor_key.lower()}__{scenario_key.lower()}__r{repetition:02d}"
                        matched_seed = (BASE_RANDOM_SEED + STAGES.index(stage) * 1000
                                        + SENSOR_PROFILES.index(sensor_key) * 100
                                        + int(scenario_key[2:]) * 10 + repetition)
                        tasks.append({
                            "schema_version": "1.0-completion-only",
                            "task_id": task_id,
                            "model_key": model["key"],
                            "model_display_name": model.get("display_name", model["key"]),
                            "model_name": model.get("model_name", ""),
                            "construction_stage": stage,
                            "sensor_profile": sensor_key,
                            "tx_power_dbm": get_sensor_profile(sensor_key).tx_power_dbm,
                            "scenario": scenario_key,
                            "repetition": repetition,
                            "random_seed": matched_seed,
                            "run_root_path": str(RESULTS_ROOT / "work" / task_id),
                            "backend": backend,
                            "require_llm": require_llm,
                        })
    return tasks


def to_agent_text(task: dict[str, Any]) -> str:
    return (
        "WAP_LLM_COMPLETION_TASK\n"
        + json.dumps(task, ensure_ascii=False, separators=(",", ":"))
        + "\nSTRICT_REQUIREMENTS\n"
        "Use the assigned model in both Planning and Execution. "
        "Execute all 8 tools through the two MCP servers in canonical order. "
        "Parameters: max_depth=3, ga_samples_per_tx=1000000, population=80, "
        "generations=35, minimum_wap_distance_m=4.0, los=true, "
        "specular_reflection=true, diffraction=true, refraction=true. "
        f"backend={task['backend']}. Return task completion status only."
    )


def filter_tasks(
    tasks: list[dict[str, Any]], *, model_keys: set[str] | None = None, stages: set[str] | None = None,
    sensor_profiles: set[str] | None = None, scenarios: set[str] | None = None, task_id: str | None = None,
) -> list[dict[str, Any]]:
    return [
        task for task in tasks
        if (not model_keys or task["model_key"] in model_keys)
        and (not stages or task["construction_stage"] in stages)
        and (not sensor_profiles or task["sensor_profile"] in sensor_profiles)
        and (not scenarios or task["scenario"] in scenarios)
        and (not task_id or task["task_id"] == task_id)
    ]


def export_manifest(tasks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not tasks:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tasks[0]))
        writer.writeheader(); writer.writerows(tasks)


def _csv_set(value: str | None) -> set[str] | None:
    if not value or value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _normalize_a2a(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid Planning Agent URL: {url}")
    path = parts.path.rstrip("/")
    if not path:
        path = "/a2a"
    elif not path.endswith("/a2a"):
        path += "/a2a"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _extract_text(value: Any, depth: int = 0) -> str:
    if depth > 8 or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"].strip()
        for key in ("message", "result", "response", "text", "output"):
            if key in value:
                text = _extract_text(value[key], depth + 1)
                if text:
                    return text
        return ""
    for attr in ("content", "message", "result", "response", "text", "output"):
        if hasattr(value, attr):
            text = _extract_text(getattr(value, attr), depth + 1)
            if text:
                return text
    return ""


def _send(task: dict[str, Any], url: str, timeout: int) -> dict[str, Any]:
    endpoint = _normalize_a2a(url)
    payload = {"role": "user", "content": {"type": "text", "text": to_agent_text(task)}}
    session = requests.Session(); session.trust_env = False
    try:
        response = session.post(endpoint, json=payload, timeout=(10, max(timeout, 1)))
        response.raise_for_status()
        body = response.json()
    finally:
        session.close()
    text = _extract_text(body)
    if not text:
        raise RuntimeError(f"Planning Agent returned no text payload: {body}")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise RuntimeError("Planning Agent response is not a JSON object")
    return value


def _assess(
    task: dict[str, Any],
    response: dict[str, Any],
    *,
    started_at: datetime,
    ended_at: datetime,
    validation_mode: bool = False,
) -> dict[str, Any]:
    records = response.get("execution_records") or []
    failed_step = response.get("failed_step")
    tool_success = len(records) == 8 and all(isinstance(row, dict) and row.get("status") == "success" for row in records)
    planning_mode = str(response.get("planning_mode") or "")
    execution_mode = str(response.get("refinement_mode") or response.get("execution_refinement_mode") or "")
    llm_ok = ("llm" in planning_mode.lower() and "llm" in execution_mode.lower()) if task["require_llm"] else True
    status_ok = response.get("status") in {"success", "completed"}
    success = bool(status_ok and tool_success and llm_ok)

    # Locate the failed step (explicit failed_step first; otherwise the first
    # non-success record), so the manifest records exactly where the agent stopped.
    failed_step_id = failed_server = failed_tool = ""
    reason = ""
    if not status_ok:
        if isinstance(failed_step, dict):
            failed_step_id = str(failed_step.get("step_id") or "")
            failed_server = str(failed_step.get("server") or "")
            failed_tool = str(failed_step.get("tool") or "")
            reason = str(failed_step.get("error") or "")
        if not reason:
            raw_error = response.get("error") or response.get("execution_error")
            reason = str(raw_error)[:500] if raw_error else "planning/execution failed"
    elif not tool_success:
        if isinstance(failed_step, dict):
            failed_step_id = str(failed_step.get("step_id") or "")
            failed_server = str(failed_step.get("server") or "")
            failed_tool = str(failed_step.get("tool") or "")
            reason = (f"failed at {failed_step_id} ({failed_server}/{failed_tool}): "
                      f"{failed_step.get('error') or 'tool did not succeed'}")
        else:
            bad = next((row for row in records if isinstance(row, dict) and row.get("status") != "success"), None)
            if bad is not None:
                failed_step_id = str(bad.get("step_id") or "")
                failed_server = str(bad.get("server") or "")
                failed_tool = str(bad.get("tool") or "")
                reason = (f"failed at {failed_step_id} ({failed_server}/{failed_tool}): "
                          f"{bad.get('error') or 'tool did not succeed'}")
            else:
                reason = f"expected 8 successful MCP tool calls, got {len(records)}"
    elif not llm_ok:
        reason = f"assigned model was not used in both agent phases: planning={planning_mode!r}, execution={execution_mode!r}"

    category = "none" if success else _failure_category(reason, failed_tool)
    final_record = next((row for row in reversed(records)
                         if isinstance(row, dict) and row.get("tool") == "optimize_wap_layout"), {})
    final_summary = (final_record.get("result_summary") or {}).get("summary") or {}

    elapsed = (ended_at - started_at).total_seconds()
    return {
        "task_id": task["task_id"],
        "benchmark_mode": "compatibility_smoke" if validation_mode else "formal",
        "model_key": task["model_key"],
        "model_display_name": task["model_display_name"],
        "construction_stage": task["construction_stage"],
        "sensor_profile": task["sensor_profile"],
        "tx_power_dbm": task["tx_power_dbm"],
        "scenario": task["scenario"],
        "success": success,
        "completion_percent": 100.0 if success else 0.0,
        "tool_calls_completed": len(records),
        "planning_mode": planning_mode,
        "execution_mode": execution_mode,
        "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": round(elapsed, 3),
        "failed_step_id": failed_step_id,
        "failed_server": failed_server,
        "failed_tool": failed_tool,
        "failure_category": category,
        "domain_validation_status": final_summary.get("domain_validation_status", ""),
        "domain_failed_probe_count": final_summary.get("domain_failed_probe_count"),
        "actual_wap_count": final_summary.get("actual_wap_count"),
        "reason": reason,
    }


def _failure_category(reason: str, failed_tool: str = "") -> str:
    lower = reason.lower()
    if any(marker in lower for marker in (
        "no 600-point rf-feasible layout found", "search exhausted its time budget",
        "minimum rf-quality count was not proven", "was not solved")):
        return "search_unresolved"
    if any(marker in lower for marker in (
        "cannot meet rsrp", "cannot meet the rf thresholds", "no rsrp-covering layout",
        "candidate site", "candidate dictionary")):
        return "model_or_design_infeasible"
    if any(marker in lower for marker in (
        "llm", "api key", "model_key", "rate limit",
        "insufficient_balance", "permissiondeniederror", "internalservererror")):
        return "llm_or_api"
    return "tool_or_runtime"


def task_result_path(task_id: str, *, validation: bool = False) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in task_id)
    result_dir = RESULTS_ROOT / ("validation" if validation else "") / "tasks"
    return result_dir / f"{safe}.json"


def load_task_result(task_id: str, *, validation: bool = False) -> dict[str, Any] | None:
    path = task_result_path(task_id, validation=validation)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def save_task_result(task_id: str, result: dict[str, Any], *, validation: bool = False) -> Path:
    path = task_result_path(task_id, validation=validation)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)
        temp = Path(handle.name)
    os.replace(temp, path)
    return path


def all_task_results(*, validation: bool = False) -> list[dict[str, Any]]:
    result_dir = RESULTS_ROOT / ("validation" if validation else "") / "tasks"
    if not result_dir.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(result_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(value, dict):
                rows.append(value)
        except Exception:
            pass
    return rows


def result_matches_current_profile(row: dict[str, Any]) -> bool:
    """Old SP IDs had different powers; never count or resume them as current runs."""
    key = str(row.get("sensor_profile") or "")
    try:
        return float(row["tx_power_dbm"]) == get_sensor_profile(key).tx_power_dbm
    except (KeyError, TypeError, ValueError):
        return False


def _active_scope_ids() -> set[str] | None:
    path = RESULTS_ROOT / "active_scope.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data.get("task_ids") if isinstance(data, dict) else None
    return set(ids) if isinstance(ids, list) else None


def export_completion_summary(output_dir: Path | None = None) -> dict:
    output_dir = output_dir or RESULTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_by_model: dict[str, set[str]] = defaultdict(set)
    active_ids = _active_scope_ids()
    for task in build_task_matrix(repetitions=1, backend="sionna", require_llm=True):
        if active_ids is None or task["task_id"] in active_ids:
            expected_by_model[task["model_key"]].add(task["task_id"])

    successful_by_model: dict[str, set[str]] = defaultdict(set)
    for row in all_task_results(validation=False):
        model_key = str(row.get("model_key", ""))
        task_id = str(row.get("task_id", ""))
        if (row.get("success") is True and result_matches_current_profile(row)
                and task_id in expected_by_model.get(model_key, set())):
            successful_by_model[model_key].add(task_id)

    summary_rows = []
    failure_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    attempted_by_model: dict[str, set[str]] = defaultdict(set)
    dense_by_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in all_task_results(validation=False):
        model_key = str(row.get("model_key"))
        if (row.get("task_id") not in expected_by_model.get(model_key, set())
                or not result_matches_current_profile(row)):
            continue
        attempted_by_model[model_key].add(str(row.get("task_id")))
        if row.get("success") is not True:
            failure_counts[model_key][str(row.get("failure_category") or "unclassified")] += 1
        if row.get("domain_validation_status") in ("passed", "failed"):
            dense_by_model[model_key][str(row["domain_validation_status"])] += 1
    for model in load_models():
        expected = len(expected_by_model[model["key"]])
        completed = len(successful_by_model[model["key"]])
        summary_rows.append({
            "model_key": model["key"],
            "model_display_name": model.get("display_name", model["key"]),
            "completion_rate_percent": round(100.0 * completed / expected, 2) if expected else 0.0,
            "assigned_tasks": expected,
            "attempted_tasks": len(attempted_by_model[model["key"]]),
            "failure_categories": dict(failure_counts[model["key"]]),
            "dense_validation": dict(dense_by_model[model["key"]]),
        })

    csv_path = output_dir / "completion_rate.csv"
    fields = ["model_key", "model_display_name", "completion_rate_percent",
              "assigned_tasks", "attempted_tasks", "failure_categories", "dense_validation"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(summary_rows)

    payload = {
        "metric": "task_completion_rate_percent",
        "definition": "successful formal tasks / assigned tasks in the active scope; dense-domain validation is reported separately",
        "scope_task_count": sum(len(ids) for ids in expected_by_model.values()),
        "models": summary_rows,
    }
    json_path = output_dir / "completion_rate.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "summary": payload}


# ---------------------------------------------------------------------------
# task_manifest.xlsx — one row per task: timing, success, failed step.
# ---------------------------------------------------------------------------
MANIFEST_XLSX_COLUMNS = [
    ("task_id", "Task ID", 34),
    ("model_key", "Model Key", 16),
    ("model_display_name", "Model", 18),
    ("construction_stage", "Stage", 11),
    ("sensor_profile", "Sensor", 8),
    ("scenario", "Scenario", 9),
    ("success", "Success", 9),
    ("start_time", "Start Time", 20),
    ("end_time", "End Time", 20),
    ("elapsed_seconds", "Elapsed (s)", 12),
    ("tool_calls_completed", "Tools Done", 11),
    ("failed_step_id", "Failed Step", 12),
    ("failed_server", "Failed Server", 20),
    ("failed_tool", "Failed Tool", 30),
    ("failure_category", "Failure Category", 25),
    ("domain_validation_status", "Dense Validation", 20),
    ("domain_failed_probe_count", "Dense Failures", 14),
    ("actual_wap_count", "WAP Count", 11),
    ("planning_mode", "Planning Mode", 26),
    ("execution_mode", "Execution Mode", 26),
    ("reason", "Reason / Error", 60),
]


def task_manifest_xlsx_path(*, validation: bool = False) -> Path:
    if validation:
        return RESULTS_ROOT / "validation" / "task_manifest.xlsx"
    return PROJECT_ROOT / "task_manifest.xlsx"


def _scenario_number(text: str) -> int:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    return int(digits) if digits else 0


def export_task_manifest_xlsx(*, validation: bool = False) -> Path:
    """Rebuild task_manifest.xlsx from the task matrix + per-task result JSONs.

    One row per matrix task (all 480): start/end time, elapsed seconds,
    success flag (TRUE/FALSE/PENDING), and the exact step (server/tool) at
    which the agent failed. Result values come from the saved JSON results
    (never fabricated); tasks without a result are PENDING with empty timing.
    """
    results_by_id = {
        str(row.get("task_id")): row
        for row in all_task_results(validation=validation)
        if result_matches_current_profile(row)
    }
    # task_id is identical regardless of require_llm/backend, so the default
    # matrix gives the complete ordered list of tasks for the manifest.
    matrix = build_task_matrix(repetitions=1, backend="sionna", require_llm=True)
    active_ids = _active_scope_ids()
    if active_ids is not None:
        matrix = [task for task in matrix if task["task_id"] in active_ids]

    path = task_manifest_xlsx_path(validation=validation)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "task_manifest"

    header_fill = PatternFill("solid", fgColor="2F4156")
    header_font = Font(bold=True, color="FFFFFF")
    ok_fill = PatternFill("solid", fgColor="DDF3DD")
    bad_fill = PatternFill("solid", fgColor="F9DDDB")
    pending_fill = PatternFill("solid", fgColor="EFEFEF")

    for col, (_key, label, width) in enumerate(MANIFEST_XLSX_COLUMNS, start=1):
        cell = ws.cell(row=1, column=col, value=label)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.freeze_panes = "A2"

    ordered = sorted(
        matrix,
        key=lambda t: (str(t.get("model_key", "")), str(t.get("construction_stage", "")),
                       str(t.get("sensor_profile", "")), _scenario_number(t.get("scenario", ""))),
    )
    for r, task in enumerate(ordered, start=2):
        result = results_by_id.get(str(task["task_id"]))
        if result is None:
            state = "PENDING"
            row_values = {}
        else:
            state = "TRUE" if result.get("success") is True else "FALSE"
            row_values = result
        for col, (key, _label, _width) in enumerate(MANIFEST_XLSX_COLUMNS, start=1):
            if key == "success":
                value = state
            elif key in ("task_id", "model_key", "model_display_name", "construction_stage",
                         "sensor_profile", "scenario"):
                value = task.get(key)
            else:
                value = row_values.get(key)
            cell = ws.cell(row=r, column=col, value=value)
            if key == "reason":
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            if key == "success":
                cell.alignment = Alignment(horizontal="center", vertical="center")
            if state == "TRUE":
                cell.fill = ok_fill
            elif state == "FALSE":
                cell.fill = bad_fill
            else:
                cell.fill = pending_fill

    wb.save(path)
    return path


# Infrastructure failures that may succeed on retry: the task itself never ran, so a
# retry does not change benchmark semantics (a task only counts as success when the
# assigned LLM was used in both agent phases and all 8 MCP tool calls completed).
TRANSIENT_MARKERS = (
    "connectionerror", "remotedisconnected", "max retries exceeded", "newconnectionerror",
    "connection aborted", "apiconnectionerror", "apitimeouterror", "readtimeout",
    "connecttimeout", "connection reset", "timeouterror", "timed out", "remote end closed",
    "exceptiongroup", "taskgroup", "internalservererror", "error code: 503",
)


def _is_transient(reason: str) -> bool:
    text = (reason or "").lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Six-LLM completion-only benchmark (V17): 6 x 5 x 8 x 2 = 480 tasks")
    p.add_argument("--models", default="all")
    p.add_argument("--stages", default="all")
    p.add_argument("--profiles", default="all")
    p.add_argument("--scenarios", default="all")
    p.add_argument("--task-id", default="")
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit; use 1 for one-task smoke run")
    p.add_argument("--backend", choices=["sionna"], default="sionna")
    p.add_argument("--resume", action="store_true", help="Resume mode: skip successful tasks and retry failed tasks")
    p.add_argument("--skip-attempted", action="store_true",
                   help="Skip every task with a saved result, including failures")
    p.add_argument("--start-index", type=int, default=1, help="Run task range start index after matrix filtering (1-based)")
    p.add_argument("--end-index", type=int, default=0, help="Run task range end index after matrix filtering (inclusive); 0 means no limit")
    p.add_argument("--timeout", type=int, default=21600)
    p.add_argument("--retry", type=int, default=3, help="Extra attempts per task for transient transport/LLM-API failures (0 disables)")
    p.add_argument("--worker-id", default="worker")
    p.add_argument("--planning-agent-url", default="http://127.0.0.1:7001")
    p.add_argument("--no-finalize", action="store_true")
    p.add_argument("--allow-llm-fallback", action="store_true",
                   help="Validation/compatibility mode: results are marked compatibility_smoke "
                        "and stored under validation/ (the LLM itself is always mandatory; "
                        "the agents no longer fall back to a deterministic plan)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--keep-work", action="store_true", help="Keep per-task intermediate tool outputs; default removes them after each task")
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows console/redirected stdout defaults to the locale ANSI codepage
    # (GBK here); failure reasons returned by LLM proxies can contain emoji or
    # other non-ANSI characters, which would raise UnicodeEncodeError inside
    # print() and crash the worker. Write UTF-8 with replacement fallbacks.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    model_keys = _csv_set(args.models)
    known_keys = {model["key"] for model in load_models()}
    unknown = (model_keys or set()) - known_keys
    if unknown:
        raise SystemExit(f"Unknown model keys: {sorted(unknown)}")
    stages = _csv_set(args.stages); profiles = _csv_set(args.profiles); scenarios = _csv_set(args.scenarios)
    tasks = filter_tasks(
        build_task_matrix(repetitions=args.repetitions, backend=args.backend, require_llm=not args.allow_llm_fallback),
        model_keys=model_keys, stages=stages, sensor_profiles=profiles, scenarios=scenarios, task_id=args.task_id or None,
    )
    if args.limit > 0:
        tasks = tasks[:args.limit]
    if args.start_index < 1:
        raise SystemExit("--start-index must be >= 1")
    start = args.start_index - 1
    end = args.end_index if args.end_index > 0 else len(tasks)
    tasks = tasks[start:end]
    if (not args.dry_run and args.limit == 0 and args.start_index == 1
            and args.end_index == 0):
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
        scope_path = RESULTS_ROOT / "active_scope.json"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=RESULTS_ROOT,
                                         delete=False, suffix=".tmp") as stream:
            json.dump({"scope": {"stages": args.stages, "profiles": args.profiles,
                                  "scenarios": args.scenarios, "models": args.models},
                       "task_ids": [task["task_id"] for task in tasks]}, stream,
                      ensure_ascii=False, indent=2)
            temporary_scope = Path(stream.name)
        os.replace(temporary_scope, scope_path)
    manifest_root = RESULTS_ROOT / ("validation" if args.allow_llm_fallback else "")
    manifest = manifest_root / f"task_manifest_{args.worker_id}.csv"
    export_manifest(tasks, manifest)
    print(json.dumps({"worker_id": args.worker_id, "selected_tasks": len(tasks), "manifest": str(manifest), "backend": args.backend}, ensure_ascii=False))
    if args.dry_run:
        return 0
    failed = 0
    for index, task in enumerate(tasks, start=1):
        existing = (load_task_result(task["task_id"], validation=args.allow_llm_fallback)
                    if args.resume or args.skip_attempted else None)
        if existing and args.skip_attempted:
            print(f"[{index}/{len(tasks)}] SKIP RECORDED {task['task_id']} success={existing.get('success', False)}")
            continue
        if existing and existing.get("success") is True and result_matches_current_profile(existing):
            print(f"[{index}/{len(tasks)}] SKIP FINAL {task['task_id']} success={existing.get('success', False)}")
            continue
        print(f"[{index}/{len(tasks)}] RUN  {task['task_id']}", flush=True)
        max_attempts = max(1, args.retry + 1)
        result: dict[str, Any] | None = None
        for attempt in range(1, max_attempts + 1):
            started_at = datetime.now()
            try:
                response = _send(task, args.planning_agent_url, args.timeout)
                result = _assess(task, response, started_at=started_at, ended_at=datetime.now(),
                                 validation_mode=args.allow_llm_fallback)
            except Exception as exc:
                ended_at = datetime.now()
                result = {
                    "task_id": task["task_id"], "benchmark_mode": "compatibility_smoke" if args.allow_llm_fallback else "formal",
                    "model_key": task["model_key"], "model_display_name": task["model_display_name"],
                    "construction_stage": task["construction_stage"], "sensor_profile": task["sensor_profile"], "scenario": task["scenario"],
                    "tx_power_dbm": task["tx_power_dbm"],
                    "success": False, "completion_percent": 0.0, "tool_calls_completed": 0,
                    "planning_mode": "", "execution_mode": "",
                    "start_time": started_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": ended_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_seconds": round((ended_at - started_at).total_seconds(), 3),
                    "failed_step_id": "", "failed_server": "", "failed_tool": "",
                    "failure_category": _failure_category(f"{type(exc).__name__}: {exc}"),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            assert result is not None
            if result["success"] or attempt >= max_attempts or not _is_transient(str(result.get("reason") or "")):
                break
            delay = 20 * attempt
            print(f"[{index}/{len(tasks)}] RETRY {attempt}/{max_attempts - 1} {task['task_id']} -> {str(result.get('reason') or '')[:120]} (waiting {delay}s)", flush=True)
            time.sleep(delay)
        path = save_task_result(task["task_id"], result, validation=args.allow_llm_fallback)
        run_root = Path(task["run_root_path"])
        if not args.keep_work and run_root.exists():
            allowed = (RESULTS_ROOT / "work").resolve()
            resolved = run_root.resolve()
            if not resolved.is_relative_to(allowed) or run_root.is_symlink():
                raise RuntimeError(f"Refusing to remove work outside {allowed}: {resolved}")
            shutil.rmtree(run_root, ignore_errors=True)
        # Refresh the Excel manifest after every task so timing/results survive a crash.
        manifest_xlsx = export_task_manifest_xlsx(validation=args.allow_llm_fallback)
        print(json.dumps({"task": task["task_id"], "success": result["success"], "result": str(path),
                          "manifest": str(manifest_xlsx), "reason": result.get("reason", "")}, ensure_ascii=False), flush=True)
        if not result["success"]:
            failed += 1
            if "insufficient_balance" in str(result.get("reason", "")).lower():
                print("STOP: shared LLM API balance is insufficient; resume after top-up", flush=True)
                break
            if args.stop_on_error:
                break
    if not args.no_finalize:
        summary = export_completion_summary(RESULTS_ROOT)
        manifest_xlsx = export_task_manifest_xlsx(validation=args.allow_llm_fallback)
        print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
        print(f"Task manifest (Excel): {manifest_xlsx}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
