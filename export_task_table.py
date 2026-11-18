"""Export the six-model matrix from the current 16 stage scenarios."""
from __future__ import annotations

import csv
from pathlib import Path

from send_task2 import build_task_matrix
from tools.wap_layout_optimization.define_problem import region_data, scenario_key

ROOT = Path(__file__).resolve().parent


def main() -> None:
    tasks = build_task_matrix()
    if len(tasks) != 480:
        raise RuntimeError(f"Expected 480 tasks, got {len(tasks)}")
    counters: dict[str, int] = {}
    rows = []
    for task in tasks:
        model = task["model_key"]
        counters[model] = counters.get(model, 0) + 1
        stage, scenario = task["construction_stage"], task["scenario"]
        rows.append({"task_name": f"{model}-{counters[model]:02d}",
            "model_key": model, "construction_stage": stage,
            "sensor_profile": task["sensor_profile"], "scenario": scenario,
            "scenario_name": region_data()["scenarios"][scenario_key(stage, scenario)]["name"],
            "task_id": task["task_id"]})
    if any(n != 80 for n in counters.values()):
        raise RuntimeError(f"Expected 80 tasks per model: {counters}")
    columns = list(rows[0])
    with (ROOT / "six_llm_task_matrix.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Six-model task matrix", "", "480 tasks: 6 models × 2 stages × 5 profiles × 8 scenarios per stage.", "",
             "| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines.extend("| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows)
    (ROOT / "six_llm_task_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported {len(rows)} tasks")


if __name__ == "__main__":
    main()
