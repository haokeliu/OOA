#!/usr/bin/env python3
"""Aggregate the preregistered V2-A observable-opportunity audit."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
FEATURE_SETS = ("confidence", "summary", "view_logits")
FIELDS = (
    "achieved_improvement",
    "recovered_oracle_fraction",
    "recovered_beyond_constant_fraction",
    "macro_weighted_action_auroc",
)


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main() -> int:
    reports = [
        json.loads(
            (REPO_ROOT / f"runs/v2a_observable_seed{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    aggregate = {
        feature_set: {
            field: summary(
                [report["feature_sets"][feature_set]["aggregate"][field] for report in reports]
            )
            for field in FIELDS
        }
        for feature_set in FEATURE_SETS
    }
    advances = []
    for report in reports:
        feature_sets = report["feature_sets"]
        delta = (
            feature_sets["view_logits"]["aggregate"]["recovered_beyond_constant_fraction"]
            - feature_sets["summary"]["aggregate"]["recovered_beyond_constant_fraction"]
        )
        advances.append(
            {
                "seed": report["seed"],
                "beyond_constant_fraction_gain": delta,
                "achieved_improvement_gain": report[
                    "view_logits_minus_summary_achieved_improvement"
                ],
            }
        )
    output = {
        "schema_version": 1,
        "protocol": "v2-A",
        "development_only": True,
        "test_accessed": False,
        "seeds": list(SEEDS),
        "feature_sets": aggregate,
        "advancement": {
            "per_seed": advances,
            "seed_count_above_0_05": sum(
                row["beyond_constant_fraction_gain"] >= 0.05 for row in advances
            ),
            "all_seeds_non_decreasing_achieved_improvement": all(
                row["achieved_improvement_gain"] >= 0 for row in advances
            ),
            "passes_v2b_rule": sum(
                row["beyond_constant_fraction_gain"] >= 0.05 for row in advances
            )
            >= 2
            and all(row["achieved_improvement_gain"] >= 0 for row in advances),
        },
    }
    path = REPO_ROOT / "reports/v2a_observable_opportunity.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
