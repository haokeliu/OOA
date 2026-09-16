#!/usr/bin/env python3
"""Aggregate the completed Q12 test metrics without reopening model selection."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
METRICS = (
    "selected_macro_ap",
    "macro_cfpr",
    "macro_recall",
    "macro_weak_recall",
    "macro_coco_small_recall",
)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    reports: list[dict[str, Any]] = []
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        path = REPO_ROOT / "runs" / f"p3e_groupsup_seed{seed}" / "test_metrics.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["protocol_freeze"] != "Q12" or not report["test_accessed"]:
            raise ValueError(f"seed {seed} is not a completed Q12 test result")
        reports.append(report)
        for method, metrics in report["methods"].items():
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "diagnostic_only": method in report["diagnostic_only_methods"],
                    **{metric: metrics[metric] for metric in METRICS},
                }
            )
    methods = list(reports[0]["frozen_methods"])
    aggregate = {
        method: {
            metric: summarize(
                [float(row[metric]) for row in rows if row["method"] == method]
            )
            for metric in METRICS
        }
        for method in methods
    }
    b1_cfpr = aggregate["B1"]["macro_cfpr"]["mean"]
    summary = {
        "schema_version": 1,
        "protocol_freeze": "Q12",
        "test_accessed": True,
        "seeds": list(SEEDS),
        "methods": aggregate,
        "relative_cfpr_reduction_vs_b1": {
            method: (b1_cfpr - metrics["macro_cfpr"]["mean"]) / b1_cfpr
            for method, metrics in aggregate.items()
        },
        "bootstrap": json.loads(
            (REPO_ROOT / "reports/final_test_bootstrap.json").read_text(encoding="utf-8")
        )["comparisons"],
    }
    output_json = REPO_ROOT / "reports/final_test_summary.json"
    output_csv = REPO_ROOT / "reports/final_test_summary.csv"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "diagnostic_only"] + [
            f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sample_std")
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for method in methods:
            writer.writerow(
                {
                    "method": method,
                    "diagnostic_only": method == "O1",
                    **{
                        f"{metric}_{stat}": aggregate[method][metric][stat]
                        for metric in METRICS
                        for stat in ("mean", "sample_std")
                    },
                }
            )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
