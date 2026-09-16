#!/usr/bin/env python3
"""Aggregate the preregistered V2-B neural observable-gain study."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
MODELS = ("LogitMLP", "RCVI")
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
            (REPO_ROOT / f"runs/v2b_gain_seed{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    aggregate = {
        model: {
            field: summary(
                [report["models"][model]["aggregate"][field] for report in reports]
            )
            for field in FIELDS
        }
        for model in MODELS
    }
    advances = [
        {
            "seed": report["seed"],
            "beyond_constant_fraction_gain_over_v2a": report[
                "rcvi_beyond_constant_gain_over_v2a"
            ],
            "rcvi_achieved_improvement": report["models"]["RCVI"]["aggregate"][
                "achieved_improvement"
            ],
        }
        for report in reports
    ]
    output = {
        "schema_version": 1,
        "protocol": "v2-B",
        "development_only": True,
        "test_accessed": False,
        "seeds": list(SEEDS),
        "models": aggregate,
        "advancement": {
            "per_seed": advances,
            "seed_count_above_0_10": sum(
                row["beyond_constant_fraction_gain_over_v2a"] >= 0.10
                for row in advances
            ),
            "all_seeds_positive_achieved_improvement": all(
                row["rcvi_achieved_improvement"] > 0 for row in advances
            ),
            "passes_method_candidate_rule": sum(
                row["beyond_constant_fraction_gain_over_v2a"] >= 0.10
                for row in advances
            )
            >= 2
            and all(row["rcvi_achieved_improvement"] > 0 for row in advances),
        },
    }
    path = REPO_ROOT / "reports/v2b_relation_conditioned_views.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
