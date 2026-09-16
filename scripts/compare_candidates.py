#!/usr/bin/env python3
"""Create a compact, tracked comparison of audited P2 candidate versions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["p2_seed17", "p2b_seed17", "p2c_seed17", "p2d_seed17", "p2e_seed17"],
    )
    parser.add_argument("--output", default="reports/p2_candidate_comparison.json")
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for run_id in args.runs:
        metrics = json.loads(
            (REPO_ROOT / "runs" / run_id / "metrics.json").read_text(encoding="utf-8")
        )
        opportunity = metrics["opportunity"]
        rows.append(
            {
                "run_id": run_id,
                "method_version": metrics.get("method_version", "p2_v1"),
                "base_views": metrics.get("base_views", "all"),
                "context_selection": metrics.get("context_selection", "ap"),
                "context_signal": metrics.get("context_signal", "probability"),
                "b1_all_map": metrics["b1_all_map"],
                "b2_all_map": metrics["b2_all_map"],
                "selected_macro_ap_delta": (
                    metrics["b2_selected_macro_ap"] - metrics["b1_selected_macro_ap"]
                ),
                "oracle_relative_improvement": opportunity["oracle_relative_improvement"],
                "beneficial_fraction": opportunity["beneficial_fraction"],
                "harmful_fraction": opportunity["harmful_fraction"],
                "ignored_fraction": opportunity["ignored_fraction"],
                "passes_preregistered_threshold": opportunity["passes_preregistered_threshold"],
            }
        )
    report = {
        "schema_version": 1,
        "split": "modelval",
        "seed": 17,
        "test_accessed": False,
        "rows": rows,
    }
    output = REPO_ROOT / args.output
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
