#!/usr/bin/env python3
"""Aggregate the preregistered V2-C downstream selector diagnostic."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
METHODS = ("B1", "B2", "B5", "M2", "LogitMLP-H", "RCVI-H")
FIELDS = ("selected_macro_ap", "macro_cfpr", "macro_recall", "macro_weak_recall")


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main() -> int:
    reports = [
        json.loads(
            (REPO_ROOT / f"runs/v2c_selector_seed{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    aggregate = {
        method: {
            field: summary([report["methods"][method][field] for report in reports])
            for field in FIELDS
        }
        for method in METHODS
    }
    per_seed = []
    for report in reports:
        base = report["methods"]["B1"]
        rcvi = report["methods"]["RCVI-H"]
        per_seed.append(
            {
                "seed": report["seed"],
                "ap_gain": rcvi["selected_macro_ap"] - base["selected_macro_ap"],
                "cfpr_change": rcvi["macro_cfpr"] - base["macro_cfpr"],
                "recall_change": rcvi["macro_recall"] - base["macro_recall"],
                "weak_recall_change": (
                    rcvi["macro_weak_recall"] - base["macro_weak_recall"]
                ),
            }
        )
    b1 = aggregate["B1"]
    rcvi = aggregate["RCVI-H"]
    checks = {
        "mean_ap_gain_at_least_0_002": (
            rcvi["selected_macro_ap"]["mean"] - b1["selected_macro_ap"]["mean"]
            >= 0.002
        ),
        "mean_cfpr_ratio_at_most_0_90": (
            rcvi["macro_cfpr"]["mean"] / b1["macro_cfpr"]["mean"] <= 0.90
        ),
        "absolute_mean_recall_gap_at_most_0_02": (
            abs(rcvi["macro_recall"]["mean"] - b1["macro_recall"]["mean"]) <= 0.02
        ),
        "mean_weak_recall_not_below_0_02": (
            rcvi["macro_weak_recall"]["mean"]
            >= b1["macro_weak_recall"]["mean"] - 0.02
        ),
        "two_seeds_ap_non_decreasing_and_cfpr_decreasing": sum(
            row["ap_gain"] >= 0 and row["cfpr_change"] < 0 for row in per_seed
        )
        >= 2,
    }
    output = {
        "schema_version": 1,
        "protocol": "v2-C",
        "development_only": True,
        "test_accessed": False,
        "seeds": list(SEEDS),
        "methods": aggregate,
        "rcvi_h_vs_b1": {
            "per_seed": per_seed,
            "mean_ap_gain": (
                rcvi["selected_macro_ap"]["mean"] - b1["selected_macro_ap"]["mean"]
            ),
            "mean_cfpr_change": rcvi["macro_cfpr"]["mean"]
            - b1["macro_cfpr"]["mean"],
            "mean_cfpr_ratio": rcvi["macro_cfpr"]["mean"]
            / b1["macro_cfpr"]["mean"],
            "checks": checks,
            "passes_direct_expansion_rule": all(checks.values()),
        },
    }
    path = REPO_ROOT / "reports/v2c_downstream_selector.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
