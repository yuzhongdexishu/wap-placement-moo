"""Check the staged Git release for credentials and unintended artifacts."""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
names = [item.decode("utf-8") for item in subprocess.check_output(
    ["git", "ls-files", "-z"], cwd=ROOT).split(b"\0") if item]
patterns = {
    "api_token": re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "private_key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "nonempty_api_key": re.compile(rb'"api_key"\s*:\s*"[^"]+"'),
}
hits = defaultdict(list)
bytes_total = 0
large = []
for name in names:
    path = ROOT / name
    if not path.is_file():
        continue
    size = path.stat().st_size
    bytes_total += size
    if size > 50 * 1024 * 1024:
        large.append({"path": name, "bytes": size})
    if size < 40 * 1024 * 1024:
        blob = path.read_bytes()
        for category, pattern in patterns.items():
            if pattern.search(blob):
                hits[category].append(name)

for name in ("models.example.json", "llm_config.example.json"):
    data = json.loads((ROOT / "config" / name).read_text(encoding="utf-8"))
    keys = [model.get("api_key") for model in data.get("models", [])] if "models" in data else [data.get("api_key")]
    if any(keys):
        hits["example_key"].append("config/" + name)

forbidden = [name for name in names if name.endswith((".wbk", ".docx", ".pdf", ".zip"))
             or name in {"config/models.json", "config/llm_config.json"}]
report = {
    "staged_files": len(names),
    "staged_bytes": bytes_total,
    "formal_task_json": sum(name.startswith("result/six_llm/tasks/") and name.endswith(".json") for name in names),
    "secret_hit_paths": dict(hits),
    "forbidden_paths": forbidden,
    "files_over_50_mb": large,
}
print(json.dumps(report, indent=2))
if hits or forbidden or large or report["formal_task_json"] != 480:
    raise SystemExit(1)
