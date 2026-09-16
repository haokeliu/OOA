#!/usr/bin/env python3
"""Aggregate the preregistered three-seed multi-risk opportunity frontier."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
FIELDS = ("selected_macro_ap", "macro_cfpr", "macro_recall", "macro_weak_recall")


def summary(values: list[float]) -> dict[str, float]:
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values)}


def main() -> int:
    reports = [
        json.loads(
            (REPO_ROOT / f"runs/v2e_frontier_seed{seed}/metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    ]
    baselines = {
        method: {
            field: summary([report["baselines"][method][field] for report in reports])
            for field in FIELDS
        }
        for method in ("B1", "B5", "M2")
    }
    point_maps = [
        {point["point_id"]: point for point in report["points"]} for report in reports
    ]
    points = []
    for point_id in point_maps[0]:
        rows = [mapping[point_id] for mapping in point_maps]
        metrics = {
            field: summary([row["metrics"][field] for row in rows]) for field in FIELDS
        }
        b1 = baselines["B1"]
        b5 = baselines["B5"]
        qualifies = (
            metrics["macro_cfpr"]["mean"] <= 0.90 * b1["macro_cfpr"]["mean"]
            and metrics["selected_macro_ap"]["mean"]
            >= b5["selected_macro_ap"]["mean"]
            and abs(metrics["macro_recall"]["mean"] - b1["macro_recall"]["mean"])
            <= 0.02
            and metrics["macro_weak_recall"]["mean"]
            >= b1["macro_weak_recall"]["mean"] - 0.02
        )
        points.append(
            {
                "point_id": point_id,
                "lambda_negative": rows[0]["lambda_negative"],
                "lambda_weak": rows[0]["lambda_weak"],
                "metrics": metrics,
                "mean_context_action_fraction": statistics.mean(
                    row["context_action_fraction"] for row in rows
                ),
                "qualifies": qualifies,
            }
        )

    feasible_for_other_metrics = [
        point
        for point in points
        if point["metrics"]["selected_macro_ap"]["mean"]
        >= baselines["B5"]["selected_macro_ap"]["mean"]
        and abs(
            point["metrics"]["macro_recall"]["mean"]
            - baselines["B1"]["macro_recall"]["mean"]
        )
        <= 0.02
        and point["metrics"]["macro_weak_recall"]["mean"]
        >= baselines["B1"]["macro_weak_recall"]["mean"] - 0.02
    ]
    best_cfpr = min(feasible_for_other_metrics, key=lambda row: row["metrics"]["macro_cfpr"]["mean"])
    output = {
        "schema_version": 1,
        "protocol": "v2-E",
        "development_only": True,
        "test_accessed": False,
        "seeds": list(SEEDS),
        "baselines": baselines,
        "points": points,
        "qualification": {
            "qualifying_point_ids": [point["point_id"] for point in points if point["qualifies"]],
            "representation_feasible_region_exists": any(
                point["qualifies"] for point in points
            ),
            "best_cfpr_point_subject_to_other_metrics": best_cfpr,
            "best_cfpr_ratio_to_b1": best_cfpr["metrics"]["macro_cfpr"]["mean"]
            / baselines["B1"]["macro_cfpr"]["mean"],
        },
    }
    path = REPO_ROOT / "reports/v2e_multi_risk_frontier.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["qualification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
