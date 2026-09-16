#!/usr/bin/env python3
"""Combined relation-block analysis of the two prospective VOC confirmations."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260911


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def main() -> int:
    sources = [
        (
            "voc2007_test",
            "test_realized",
            json.loads(
                (REPO_ROOT / "reports/v3b_voc_external_confirmation.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
        (
            "voc2012_val",
            "voc2012_realized",
            json.loads(
                (REPO_ROOT / "reports/v3c_voc2012_external_confirmation.json").read_text(
                    encoding="utf-8"
                )
            ),
        ),
    ]
    combined_rows = []
    for dataset, realized_name, source in sources:
        for row in source["rows"]:
            combined_rows.append(
                {
                    "dataset": dataset,
                    "pair_id": row["pair_id"],
                    "seed": row["seed"],
                    "representation": row["representation"],
                    "checkpoint_oracle": row["checkpoint_oracle"],
                    "checkpoint_observable": row["checkpoint_observable"],
                    "realized": row[realized_name],
                }
            )
    relations = sorted({row["pair_id"] for row in combined_rows})
    tasks = sorted(
        {(row["dataset"], row["seed"], row["representation"]) for row in combined_rows}
    )
    lookup = {
        (row["pair_id"], row["dataset"], row["seed"], row["representation"]): row
        for row in combined_rows
    }
    arrays = {
        field: np.asarray(
            [
                [lookup[(relation, *task)][field] for task in tasks]
                for relation in relations
            ],
            dtype=np.float64,
        )
        for field in ("checkpoint_oracle", "checkpoint_observable", "realized")
    }
    oracle_flat = arrays["checkpoint_oracle"].reshape(-1)
    observable_flat = arrays["checkpoint_observable"].reshape(-1)
    realized_flat = arrays["realized"].reshape(-1)
    oracle_correlation = spearman(oracle_flat, realized_flat)
    observable_correlation = spearman(observable_flat, realized_flat)
    observed_advantage = observable_correlation - oracle_correlation

    swap_advantages = []
    for swap_bits in itertools.product((False, True), repeat=len(relations)):
        swap = np.asarray(swap_bits)[:, None]
        first = np.where(swap, arrays["checkpoint_observable"], arrays["checkpoint_oracle"])
        second = np.where(swap, arrays["checkpoint_oracle"], arrays["checkpoint_observable"])
        swap_advantages.append(
            spearman(second.reshape(-1), realized_flat)
            - spearman(first.reshape(-1), realized_flat)
        )
    swap_p = float(np.mean(np.asarray(swap_advantages) >= observed_advantage))

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_observable = np.empty(BOOTSTRAP_REPLICATES)
    bootstrap_advantage = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, len(relations), size=len(relations))
        oracle = arrays["checkpoint_oracle"][selected].reshape(-1)
        observable = arrays["checkpoint_observable"][selected].reshape(-1)
        realized = arrays["realized"][selected].reshape(-1)
        bootstrap_observable[replicate] = spearman(observable, realized)
        bootstrap_advantage[replicate] = spearman(observable, realized) - spearman(
            oracle, realized
        )

    per_dataset_realized = {}
    for dataset, _realized_name, _source in sources:
        per_dataset_realized[dataset] = np.asarray(
            [
                lookup[(relation, dataset, seed, representation)]["realized"]
                for relation in relations
                for seed in (17, 29, 43)
                for representation in (
                    "confidence",
                    "summary",
                    "view_logits",
                    "LogitMLP",
                    "RCVI",
                )
            ]
        )
    report = {
        "schema_version": 1,
        "protocol": "v3-BC-secondary-combined-voc-analysis",
        "analysis_status": "secondary analysis specified before V3-C label reveal",
        "datasets": [dataset for dataset, _name, _source in sources],
        "relation_count": len(relations),
        "task_count": len(combined_rows),
        "observed": {
            "oracle_spearman": oracle_correlation,
            "observable_spearman": observable_correlation,
            "advantage": observed_advantage,
            "observable_mean_absolute_calibration_error": float(
                np.mean(np.abs(observable_flat - realized_flat))
            ),
            "cross_vintage_realized_spearman": spearman(
                per_dataset_realized["voc2007_test"],
                per_dataset_realized["voc2012_val"],
            ),
        },
        "exact_relation_block_predictor_swap": {
            "assignment_count": len(swap_advantages),
            "one_sided_p_advantage": swap_p,
        },
        "relation_cluster_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "observable_spearman_95_percentile_interval": np.quantile(
                bootstrap_observable, [0.025, 0.975]
            ).tolist(),
            "advantage_95_percentile_interval": np.quantile(
                bootstrap_advantage, [0.025, 0.975]
            ).tolist(),
            "fraction_advantage_positive": float(np.mean(bootstrap_advantage > 0)),
        },
    }
    (REPO_ROOT / "reports/v3bc_combined_voc_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
