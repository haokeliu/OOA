#!/usr/bin/env python3
"""Paired post hoc analyses for the fixed V8 development report."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_v8_calibration_baselines import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED

INPUT_PATH = REPO_ROOT / "reports/v8_calibration_baselines.json"
OUTPUT_PATH = REPO_ROOT / "reports/v8_calibration_baselines_analysis.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_summary(
    rows: list[dict[str, object]], key: str, generator
) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(float(row[key]))
    blocks = [
        np.asarray(grouped[pair_id], dtype=np.float64) for pair_id in sorted(grouped)
    ]
    values = np.concatenate(blocks)
    replicates = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        replicates[replicate] = np.concatenate(
            [blocks[index] for index in selected]
        ).mean()
    return {
        "mean": float(values.mean()),
        "relation_cluster_95_percentile_interval": np.quantile(
            replicates, (0.025, 0.975)
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
    }


def main() -> int:
    report = json.loads(INPUT_PATH.read_text())
    calibration = {
        (row["seed"], row["representation"], row["pair_id"], row["calibration"]): row
        for row in report["calibration_rows"]
    }
    baselines = {
        (row["seed"], row["representation"], row["pair_id"], row["method"]): row
        for row in report["baseline_rows"]
    }
    candidate_only = {
        (row["seed"], row["pair_id"], row["method"]): row
        for row in report["baseline_rows"]
        if row["representation"] is None
    }
    generator = np.random.default_rng(BOOTSTRAP_SEED + 1)

    calibration_differences = {}
    for method in ("temperature", "platt", "isotonic"):
        paired_rows = []
        for key, method_row in calibration.items():
            seed, representation, pair_id, calibration_method = key
            if calibration_method != method:
                continue
            raw = calibration[(seed, representation, pair_id, "raw")]
            paired_rows.append(
                {
                    "pair_id": pair_id,
                    "nll_change_vs_raw": method_row["nll"] - raw["nll"],
                    "brier_change_vs_raw": method_row["brier"] - raw["brier"],
                    "ece_change_vs_raw": method_row["ece_10_bin"]
                    - raw["ece_10_bin"],
                }
            )
        calibration_differences[method] = {
            metric: cluster_summary(paired_rows, metric, generator)
            for metric in (
                "nll_change_vs_raw",
                "brier_change_vs_raw",
                "ece_change_vs_raw",
            )
        }

    comparison_specs = {
        "soft_platt_minus_direct_gain_hard": (
            "analytic_soft_platt",
            "direct_gain_hard",
            "same_representation",
        ),
        "soft_platt_minus_probability_hard_platt": (
            "analytic_soft_platt",
            "probability_hard_platt",
            "same_representation",
        ),
        "soft_platt_minus_static_interpolation": (
            "analytic_soft_platt",
            "nested_selected_static_interpolation",
            "candidate_only",
        ),
        "soft_platt_minus_logistic_stacking": (
            "analytic_soft_platt",
            "logistic_endpoint_stacking",
            "candidate_only",
        ),
    }
    baseline_comparisons = {}
    for comparison, (left_method, right_method, alignment) in comparison_specs.items():
        paired_rows = []
        for (seed, representation, pair_id, method), left in baselines.items():
            if method != left_method:
                continue
            if alignment == "same_representation":
                right = baselines[(seed, representation, pair_id, right_method)]
            else:
                right = candidate_only[(seed, pair_id, right_method)]
            paired_rows.append(
                {"pair_id": pair_id, "difference": left["gain"] - right["gain"]}
            )
        baseline_comparisons[comparison] = cluster_summary(
            paired_rows, "difference", generator
        )

    raw_nll = np.asarray(
        [
            row["nll"]
            for row in report["calibration_rows"]
            if row["calibration"] == "raw"
        ]
    )
    isotonic_nll = np.asarray(
        [
            row["nll"]
            for row in report["calibration_rows"]
            if row["calibration"] == "isotonic"
        ]
    )
    output = {
        "schema_version": 1,
        "protocol": "v8-posthoc-paired-analysis",
        "development_only": True,
        "input_sha256": sha256(INPUT_PATH),
        "calibration_differences": calibration_differences,
        "baseline_comparisons": baseline_comparisons,
        "isotonic_task_fraction_worse_nll_than_raw": float(
            np.mean(isotonic_nll > raw_nll)
        ),
        "isotonic_task_fraction_nll_above_one": float(np.mean(isotonic_nll > 1.0)),
        "scope_interpretation": {
            "best_candidate_constrained_method": "analytic_soft_platt",
            "unconstrained_controls_leave_candidate_portfolio": True,
            "logistic_stacking_is_not_a_candidate_selection_method": True,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
