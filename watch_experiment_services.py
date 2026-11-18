"""Restart the local MCP/A2A service group if it exits during the 480-task run."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent
PORTS = (7001, 7002, 8101, 8102)
LOGS = ROOT / "result/logs"


def open_ports() -> int:
    count = 0
    for port in PORTS:
        with socket.socket() as connection:
            connection.settimeout(.5)
            count += connection.connect_ex(("127.0.0.1", port)) == 0
    return count


def attempts() -> int:
    scope_file = ROOT / "result/six_llm/active_scope.json"
    if not scope_file.is_file():
        return 0
    allowed = set(json.loads(scope_file.read_text(encoding="utf-8"))["task_ids"])
    count = 0
    for path in (ROOT / "result/six_llm/tasks").glob("*.json"):
        try:
            count += json.loads(path.read_text(encoding="utf-8")).get("task_id") in allowed
        except (OSError, ValueError):
            pass
    return count


def main() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    while attempts() < 480:
        task_files = list((ROOT / "result/six_llm/tasks").glob("*.json"))
        if task_files:
            newest = max(task_files, key=lambda path: path.stat().st_mtime)
            if newest.stat().st_mtime > started_at:
                try:
                    result = json.loads(newest.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    result = {}
                if "insufficient_balance" in str(result.get("reason", "")).lower():
                    worker_file = LOGS / "six_llm_full480.pid"
                    worker_id = int(worker_file.read_text().strip())
                    if psutil.pid_exists(worker_id):
                        worker = psutil.Process(worker_id)
                        if "send_task2.py" in " ".join(worker.cmdline()):
                            worker.terminate()
                    print("STOP: shared LLM API balance exhausted; worker paused", flush=True)
                    break
        if open_ports() == 0:
            # run_all.py stops the remaining children when one service exits.
            with (LOGS / "service_watch_group.out.log").open("a", encoding="utf-8") as out, \
                 (LOGS / "service_watch_group.err.log").open("a", encoding="utf-8") as err:
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                process = subprocess.Popen([sys.executable, "-u", "run_all.py"],
                                           cwd=ROOT, stdout=out, stderr=err,
                                           creationflags=flags)
                (LOGS / "service_watch_service.pid").write_text(str(process.pid), encoding="utf-8")
                print(f"restarted service group pid={process.pid}", flush=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
