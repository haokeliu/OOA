#!/usr/bin/env python3
"""Post hoc robustness summaries for the V7 cross-candidate development result."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "reports/v7_cross_candidate_development.json"
OUTPUT_PATH = REPO_ROOT / "reports/v7_cross_candidate_robustness.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    report = json.loads(INPUT_PATH.read_text())
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in report["rows"]:
        grouped[row["pair_id"]].append(row)
    relation_rows = []
    for pair_id, rows in grouped.items():
        relation_rows.append(
            {
                "pair_id": pair_id,
                "soft_minus_probability_hard": float(
                    np.mean([row["soft_minus_probability_hard"] for row in rows])
                ),
                "endpoint_oracle": float(
                    np.mean([row["endpoint_oracle"] for row in rows])
                ),
                "soft_interior_fraction": float(
                    np.mean([row["soft_interior_fraction"] for row in rows])
                ),
            }
        )
    relation_rows.sort(key=lambda row: row["soft_minus_probability_hard"])
    effects = np.asarray(
        [row["soft_minus_probability_hard"] for row in relation_rows],
        dtype=np.float64,
    )
    oracle = np.asarray(
        [row["endpoint_oracle"] for row in relation_rows], dtype=np.float64
    )
    interior = np.asarray(
        [row["soft_interior_fraction"] for row in relation_rows], dtype=np.float64
    )
    overall = report["overall"]
    constant = overall["constant_achieved"]
    hard = overall["probability_hard_achieved"]
    soft = overall["soft_achieved"]
    endpoint_oracle = overall["endpoint_oracle"]
    leave_one_out = np.asarray(
        [(effects.sum() - value) / (len(effects) - 1) for value in effects]
    )
    output = {
        "schema_version": 1,
        "protocol": "v7-cross-candidate-posthoc-robustness",
        "development_only": True,
        "input_sha256": sha256(INPUT_PATH),
        "relation_count": len(effects),
        "median_relation_effect": float(np.median(effects)),
        "trimmed_mean_drop_one_each_tail": float(np.mean(np.sort(effects)[1:-1])),
        "leave_one_relation_out_mean_range": [
            float(leave_one_out.min()),
            float(leave_one_out.max()),
        ],
        "soft_relative_improvement_over_hard_total_gain": (soft - hard) / hard,
        "soft_relative_improvement_over_hard_beyond_constant_gain": (
            (soft - hard) / (hard - constant)
        ),
        "hard_recovered_beyond_constant_fraction": (
            (hard - constant) / (endpoint_oracle - constant)
        ),
        "soft_recovered_beyond_constant_fraction": (
            (soft - constant) / (endpoint_oracle - constant)
        ),
        "spearman_effect_vs_endpoint_oracle": float(
            spearmanr(effects, oracle).statistic
        ),
        "spearman_effect_vs_soft_interior_fraction": float(
            spearmanr(effects, interior).statistic
        ),
        "relation_rows": relation_rows,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in output.items() if key != "relation_rows"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
