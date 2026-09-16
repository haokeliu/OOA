#!/usr/bin/env python3
"""Aggregate V2-F dense spatial observability across registered seeds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main() -> int:
    reports = [
        json.loads(
            (REPO_ROOT / f"runs/v2f_dense_seed{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    downstream = [
        json.loads(
            (REPO_ROOT / f"runs/v2f_dense_seed{seed}/downstream_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    advances = [
        {
            "seed": report["seed"],
            "beyond_constant_fraction_gain_over_rcvi": report[
                "beyond_constant_gain_over_rcvi"
            ],
            "achieved_improvement": report["aggregate"]["achieved_improvement"],
        }
        for report in reports
    ]
    methods = {}
    for method in ("B1", "B5", "M2", "RCVI-H", "DenseMapGain-H"):
        methods[method] = {
            field: summary([row["methods"][method][field] for row in downstream])
            for field in (
                "selected_macro_ap",
                "macro_cfpr",
                "macro_recall",
                "macro_weak_recall",
            )
        }
    output = {
        "schema_version": 1,
        "protocol": "v2-F",
        "development_only": True,
        "test_accessed": False,
        "seeds": list(SEEDS),
        "dense_observable_opportunity": {
            "achieved_improvement": summary(
                [report["aggregate"]["achieved_improvement"] for report in reports]
            ),
            "recovered_oracle_fraction": summary(
                [report["aggregate"]["recovered_oracle_fraction"] for report in reports]
            ),
            "recovered_beyond_constant_fraction": summary(
                [
                    report["aggregate"]["recovered_beyond_constant_fraction"]
                    for report in reports
                ]
            ),
        },
        "advancement": {
            "per_seed": advances,
            "seed_count_gain_at_least_0_05": sum(
                row["beyond_constant_fraction_gain_over_rcvi"] >= 0.05
                for row in advances
            ),
            "all_seeds_positive_achieved_improvement": all(
                row["achieved_improvement"] > 0 for row in advances
            ),
            "passes_dense_risk_frontier_rule": sum(
                row["beyond_constant_fraction_gain_over_rcvi"] >= 0.05
                for row in advances
            )
            >= 2
            and all(row["achieved_improvement"] > 0 for row in advances),
        },
        "downstream_secondary": methods,
    }
    path = REPO_ROOT / "reports/v2f_dense_spatial_observability.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
