#!/usr/bin/env python3
"""Post hoc descriptive robustness analysis for the frozen V6 result."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "reports/v6_openimages_test_confirmation.json"
OUTPUT_PATH = REPO_ROOT / "reports/v6_openimages_robustness.json"


def main() -> int:
    report = json.loads(INPUT_PATH.read_text())
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in report["rows"]:
        grouped[row["pair_id"]].append(row["soft_minus_probability_hard"])
    relation_rows = [
        {
            "pair_id": pair_id,
            "task_count": len(values),
            "mean_soft_minus_probability_hard": float(np.mean(values)),
            "min_task_effect": float(np.min(values)),
            "max_task_effect": float(np.max(values)),
        }
        for pair_id, values in grouped.items()
    ]
    relation_rows.sort(key=lambda row: row["mean_soft_minus_probability_hard"])
    relation_means = np.asarray(
        [row["mean_soft_minus_probability_hard"] for row in relation_rows]
    )
    leave_one_relation_out = np.asarray(
        [
            (relation_means.sum() - value) / (len(relation_means) - 1)
            for value in relation_means
        ]
    )
    primary = report["primary"]
    constant = primary["mean_constant_achieved"]
    hard = primary["mean_probability_hard_achieved"]
    soft = primary["mean_soft_achieved"]
    oracle = primary["mean_endpoint_oracle"]
    positive_count = int(np.sum(relation_means > 0))
    output = {
        "schema_version": 1,
        "protocol": "v6-openimages-posthoc-descriptive-robustness",
        "confirmatory_result_unchanged": True,
        "relation_count": len(relation_rows),
        "positive_relation_count": positive_count,
        "negative_relation_count": int(np.sum(relation_means < 0)),
        "exact_one_sided_relation_sign_test_p": float(
            binomtest(
                positive_count,
                len(relation_rows),
                p=0.5,
                alternative="greater",
            ).pvalue
        ),
        "median_relation_effect": float(np.median(relation_means)),
        "leave_one_relation_out_mean_range": [
            float(leave_one_relation_out.min()),
            float(leave_one_relation_out.max()),
        ],
        "soft_relative_improvement_over_hard_total_gain": (soft - hard) / hard,
        "soft_relative_improvement_over_hard_beyond_constant_gain": (
            (soft - hard) / (hard - constant)
        ),
        "hard_recovered_beyond_constant_fraction": (
            (hard - constant) / (oracle - constant)
        ),
        "soft_recovered_beyond_constant_fraction": (
            (soft - constant) / (oracle - constant)
        ),
        "relation_rows": relation_rows,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "relation_rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
