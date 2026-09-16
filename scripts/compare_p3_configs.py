#!/usr/bin/env python3
"""Freeze the bounded seed-17 P3 optimization audit as JSON and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("default", "p3_natural_seed17"),
    ("c2_dual01", "p3_natural_seed17_c2_dual01"),
    ("c3_dual05", "p3_natural_seed17_c3_dual05"),
    ("c4_dual10", "p3_natural_seed17_c4_dual10"),
    ("c5_dual05_long", "p3_natural_seed17_c5_dual05_long"),
    ("c6_dual10_long", "p3_natural_seed17_c6_dual10_long"),
]


def main() -> int:
    rows: list[dict[str, object]] = []
    for config_id, run_id in RUNS:
        run_dir = REPO_ROOT / "runs" / run_id
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
        m1 = metrics["methods"]["M1"]
        b5 = metrics["methods"]["B5"]
        rows.append(
            {
                "config_id": config_id,
                "run_id": run_id,
                "dual_lr": config["training"]["dual_lr"],
                "max_epochs": config["training"]["gate_epochs"],
                "patience": config["training"]["patience"],
                "m1_feasible": m1["feasible"],
                "m1_constraint_violation": m1["constraint_violation"],
                "m1_selected_macro_ap": m1["selected_macro_ap"],
                "b5_selected_macro_ap": b5["selected_macro_ap"],
                "m1_minus_b5_ap": m1["selected_macro_ap"] - b5["selected_macro_ap"],
                "passes_p3": bool(
                    m1["feasible"]
                    and m1["selected_macro_ap"] > b5["selected_macro_ap"]
                    and m1["selected_macro_ap"] > metrics["methods"]["B4"]["selected_macro_ap"]
                ),
            }
        )
    report = {
        "schema_version": 1,
        "seed": 17,
        "candidate_run": "p2d_seed17",
        "test_accessed": False,
        "planned_configuration_count": len(RUNS),
        "all_configs_infeasible": all(not row["m1_feasible"] for row in rows),
        "any_config_passes_p3": any(row["passes_p3"] for row in rows),
        "configs": rows,
    }
    output_json = REPO_ROOT / "reports" / "p3_seed17_optimization_audit.json"
    output_csv = REPO_ROOT / "reports" / "p3_seed17_optimization_audit.csv"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
