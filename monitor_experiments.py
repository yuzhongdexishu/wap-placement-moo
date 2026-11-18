"""Write a compact experiment checkpoint; --loop repeats every 30 minutes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tools.wap_layout_optimization.define_problem import get_sensor_profile

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "result"
CHINA = timezone(timedelta(hours=8))


def _json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def checkpoint(expected_tasks: int = 480) -> dict:
    tasks = sorted((RESULT / "six_llm/tasks").glob("*.json"))
    scope = _json(RESULT / "six_llm/active_scope.json")
    allowed = set(scope.get("task_ids", [])) if scope else None
    if allowed is not None:
        expected_tasks = len(allowed)
    selected = [(path, row) for path in tasks if (row := _json(path)) is not None
                and (allowed is None or row.get("task_id") in allowed)
                and row.get("sensor_profile") in {"SP1", "SP2", "SP3", "SP4", "SP5"}
                and row.get("tx_power_dbm") == get_sensor_profile(row["sensor_profile"]).tx_power_dbm]
    records = [row for _, row in selected]
    success = sum(row.get("success") is True for row in records)
    by_model: dict[str, dict[str, int]] = {}
    for row in records:
        model = str(row.get("model_key") or "unknown")
        counts = by_model.setdefault(model, {"attempted": 0, "successful": 0, "failed": 0})
        counts["attempted"] += 1
        counts["successful" if row.get("success") is True else "failed"] += 1
    moo = _json(RESULT / "last_agent_response.json")
    baselines = [p for stage in ("frame", "enclosure")
                 for p in (RESULT / stage).glob("*/case_manifest.json")]
    exact = _json(RESULT / "baseline_comparison_summary.json")
    latest = max((path for path, _ in selected), key=lambda p: p.stat().st_mtime) if selected else None
    return {"checked_at_china": datetime.now(CHINA).isoformat(timespec="seconds"),
            "moo_status": (moo or {}).get("status", "pending"),
            "baseline_cases": len(baselines), "baseline_expected": 10,
            "baseline_exact_comparison_complete": bool(
                exact and len(exact.get("cases", [])) == 10
                and set(exact.get("pm_joint", {})) == {"frame", "enclosure"}),
            "six_llm": {"attempted": len(records), "expected": expected_tasks,
                        "successful": success, "failed": len(records) - success,
                        "by_model": by_model,
                        "last_task_update": datetime.fromtimestamp(latest.stat().st_mtime, CHINA).isoformat(timespec="seconds")
                                            if latest else None,
                        "recent_failures": [{"task_id": row.get("task_id"),
                                             "failed_tool": row.get("failed_tool"),
                                             "reason": str(row.get("reason") or "")[:250]}
                                            for row in records if row.get("success") is not True][-3:]}}


def write_checkpoint(expected_tasks: int = 480) -> dict:
    RESULT.mkdir(exist_ok=True)
    payload = checkpoint(expected_tasks)
    target = RESULT / "experiment_status.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=RESULT,
                                     delete=False, suffix=".tmp") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        temp = Path(stream.name)
    os.replace(temp, target)
    with (RESULT / "experiment_status_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--expected-tasks", type=int, default=480)
    args = parser.parse_args()
    if args.interval_seconds < 60:
        raise ValueError("Minimum monitoring interval is 60 seconds")
    while True:
        payload = write_checkpoint(args.expected_tasks)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if (RESULT / "six_llm/active_scope.json").is_file():
            try:
                from send_task2 import export_completion_summary
                export_completion_summary()
                subprocess.run([sys.executable, str(ROOT / "plot_six_llm_completion.py")],
                               cwd=ROOT, check=True)
            except Exception as exc:
                print(f"completion plot refresh failed: {exc}", flush=True)
        if not args.loop or payload["six_llm"]["attempted"] >= args.expected_tasks:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
