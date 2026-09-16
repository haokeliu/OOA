#!/usr/bin/env python3
"""Summarize nested versus legacy V3 neural checkpoint behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]


def spearman(first: list[float], second: list[float]) -> float:
    return float(spearmanr(first, second).statistic)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    nested_checkpoint = [float(row["checkpoint_observable"]) for row in rows]
    nested_modelval = [float(row["modelval_realized"]) for row in rows]
    legacy_checkpoint = [float(row["legacy_checkpoint_observable"]) for row in rows]
    legacy_modelval = [float(row["legacy_modelval_realized"]) for row in rows]
    return {
        "task_count": len(rows),
        "nested": {
            "checkpoint_mean": float(np.mean(nested_checkpoint)),
            "modelval_mean": float(np.mean(nested_modelval)),
            "checkpoint_to_modelval_spearman": spearman(
                nested_checkpoint, nested_modelval
            ),
            "checkpoint_to_modelval_mae": float(
                np.mean(
                    np.abs(
                        np.asarray(nested_checkpoint) - np.asarray(nested_modelval)
                    )
                )
            ),
        },
        "legacy": {
            "checkpoint_mean": float(np.mean(legacy_checkpoint)),
            "modelval_mean": float(np.mean(legacy_modelval)),
            "checkpoint_to_modelval_spearman": spearman(
                legacy_checkpoint, legacy_modelval
            ),
            "checkpoint_to_modelval_mae": float(
                np.mean(
                    np.abs(
                        np.asarray(legacy_checkpoint) - np.asarray(legacy_modelval)
                    )
                )
            ),
        },
        "nested_minus_legacy": {
            "checkpoint_mean": float(
                np.mean(nested_checkpoint) - np.mean(legacy_checkpoint)
            ),
            "modelval_mean": float(np.mean(nested_modelval) - np.mean(legacy_modelval)),
        },
    }


def main() -> int:
    source = json.loads(
        (REPO_ROOT / "reports/v3d_nested_neural_checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    rows = source["rows"]
    report = {
        "schema_version": 1,
        "protocol": "v3-D-nested-neural-analysis",
        "development_only": True,
        "test_accessed": False,
        "outer_checkpoint_labels_used_for_training_or_selection": False,
        "overall": summarize(rows),
        "by_representation": {
            representation: summarize(
                [row for row in rows if row["representation"] == representation]
            )
            for representation in ("LogitMLP", "RCVI")
        },
        "by_seed": {
            str(seed): summarize([row for row in rows if row["seed"] == seed])
            for seed in (17, 29, 43)
        },
    }
    output_path = REPO_ROOT / "reports/v3d_nested_neural_analysis.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
