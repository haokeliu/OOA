#!/usr/bin/env python3
"""Aggregate V2-D balanced risk-aligned opportunity and downstream metrics."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
METHODS = ("B1", "B5", "M2", "RCVI-H", "Balanced-RCVI-H")
FIELDS = ("selected_macro_ap", "macro_cfpr", "macro_recall", "macro_weak_recall")


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main() -> int:
    opportunity = [
        json.loads(
            (REPO_ROOT / f"runs/v2d_balanced_seed{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    downstream = [
        json.loads(
            (
                REPO_ROOT / f"runs/v2d_balanced_seed{seed}/downstream_metrics.json"
            ).read_text(encoding="utf-8")
        )
        for seed in SEEDS
    ]
    methods = {
        method: {
            field: summary([report["methods"][method][field] for report in downstream])
            for field in FIELDS
        }
        for method in METHODS
    }
    b1 = methods["B1"]
    b5 = methods["B5"]
    balanced = methods["Balanced-RCVI-H"]
    per_seed = [
        {
            "seed": report["seed"],
            "ap_gain": report["methods"]["Balanced-RCVI-H"]["selected_macro_ap"]
            - report["methods"]["B1"]["selected_macro_ap"],
            "cfpr_change": report["methods"]["Balanced-RCVI-H"]["macro_cfpr"]
            - report["methods"]["B1"]["macro_cfpr"],
        }
        for report in downstream
    ]
    checks = {
        "mean_cfpr_ratio_at_most_0_90": balanced["macro_cfpr"]["mean"]
        / b1["macro_cfpr"]["mean"]
        <= 0.90,
        "mean_ap_at_least_b5": balanced["selected_macro_ap"]["mean"]
        >= b5["selected_macro_ap"]["mean"],
        "absolute_mean_recall_gap_at_most_0_02": abs(
            balanced["macro_recall"]["mean"] - b1["macro_recall"]["mean"]
        )
        <= 0.02,
        "mean_weak_recall_not_below_b1_minus_0_02": balanced["macro_weak_recall"][
            "mean"
        ]
        >= b1["macro_weak_recall"]["mean"] - 0.02,
        "two_seeds_ap_non_decreasing_and_cfpr_decreasing": sum(
            row["ap_gain"] >= 0 and row["cfpr_change"] < 0 for row in per_seed
        )
        >= 2,
    }
    output = {
        "schema_version": 1,
        "protocol": "v2-D",
        "development_only": True,
        "test_accessed": False,
        "seeds": list(SEEDS),
        "opportunity": {
            "achieved_improvement": summary(
                [report["aggregate"]["achieved_improvement"] for report in opportunity]
            ),
            "recovered_oracle_fraction": summary(
                [report["aggregate"]["recovered_oracle_fraction"] for report in opportunity]
            ),
            "context_negative_achieved_improvement": summary(
                [
                    report["strata"]["context_negative"]["achieved_improvement"]
                    for report in opportunity
                ]
            ),
            "weak_target_recovered_oracle_fraction": summary(
                [
                    report["strata"]["weak_target"]["recovered_oracle_fraction"]
                    for report in opportunity
                ]
            ),
        },
        "methods": methods,
        "balanced_rcvi_h_vs_b1": {
            "per_seed": per_seed,
            "mean_cfpr_ratio": balanced["macro_cfpr"]["mean"]
            / b1["macro_cfpr"]["mean"],
            "checks": checks,
            "passes_expansion_rule": all(checks.values()),
        },
    }
    path = REPO_ROOT / "reports/v2d_balanced_risk.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
