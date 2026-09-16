#!/usr/bin/env python3
"""Relation-cluster robustness for the V3-A observability law."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_v3_observability_law import design_matrix

BOOTSTRAP_REPLICATES = 20_000
PERMUTATION_REPLICATES = 100_000
RANDOM_SEED = 20260910


def cross_validated_prediction(
    rows: list[dict[str, object]], include_observable: bool
) -> np.ndarray:
    target = np.asarray([row["modelval_realized"] for row in rows], dtype=np.float64)
    prediction = np.zeros_like(target)
    for relation_id in sorted({str(row["pair_id"]) for row in rows}):
        test = np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] == relation_id]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] != relation_id]
        )
        train_x, test_x = design_matrix(rows, train, test, include_observable)
        model = Ridge(alpha=1.0).fit(train_x, target[train])
        prediction[test] = model.predict(test_x)
    return prediction


def main() -> int:
    source = json.loads(
        (REPO_ROOT / "reports/v3a_observability_law.json").read_text(encoding="utf-8")
    )
    rows = source["rows"]
    target = np.asarray([row["modelval_realized"] for row in rows], dtype=np.float64)
    oracle_prediction = cross_validated_prediction(rows, include_observable=False)
    observable_prediction = cross_validated_prediction(rows, include_observable=True)
    relations = sorted({str(row["pair_id"]) for row in rows})
    relation_indices = [
        np.asarray([index for index, row in enumerate(rows) if row["pair_id"] == relation])
        for relation in relations
    ]
    observed = {
        "oracle_r_squared": float(r2_score(target, oracle_prediction)),
        "observable_r_squared": float(r2_score(target, observable_prediction)),
        "r_squared_gain": float(
            r2_score(target, observable_prediction) - r2_score(target, oracle_prediction)
        ),
        "oracle_spearman": float(spearmanr(target, oracle_prediction).statistic),
        "observable_spearman": float(spearmanr(target, observable_prediction).statistic),
        "mean_absolute_error_reduction": float(
            np.mean(np.abs(target - oracle_prediction))
            - np.mean(np.abs(target - observable_prediction))
        ),
    }
    generator = np.random.default_rng(RANDOM_SEED)
    r2_gain = np.empty(BOOTSTRAP_REPLICATES)
    mae_reduction = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_relations = generator.integers(0, len(relations), size=len(relations))
        selected = np.concatenate([relation_indices[index] for index in selected_relations])
        selected_target = target[selected]
        selected_oracle = oracle_prediction[selected]
        selected_observable = observable_prediction[selected]
        r2_gain[replicate] = r2_score(selected_target, selected_observable) - r2_score(
            selected_target, selected_oracle
        )
        mae_reduction[replicate] = np.mean(
            np.abs(selected_target - selected_oracle)
        ) - np.mean(np.abs(selected_target - selected_observable))

    relation_sse_difference = np.asarray(
        [
            np.sum((target[index] - oracle_prediction[index]) ** 2)
            - np.sum((target[index] - observable_prediction[index]) ** 2)
            for index in relation_indices
        ]
    )
    observed_sse_improvement = float(relation_sse_difference.sum())
    signs = generator.choice(
        (-1.0, 1.0), size=(PERMUTATION_REPLICATES, len(relations))
    )
    null_sse_improvement = signs @ relation_sse_difference
    permutation_p = float(
        (1 + np.sum(null_sse_improvement >= observed_sse_improvement))
        / (PERMUTATION_REPLICATES + 1)
    )
    report = {
        "schema_version": 1,
        "protocol": "v3-A-posthoc-relation-cluster-robustness",
        "development_only": True,
        "test_accessed": False,
        "analysis_status": "posthoc dependence robustness; primary rule unchanged",
        "relation_count": len(relations),
        "task_count": len(rows),
        "observed": observed,
        "relation_cluster_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RANDOM_SEED,
            "r_squared_gain_95_percentile_interval": np.quantile(
                r2_gain, [0.025, 0.975]
            ).tolist(),
            "fraction_r_squared_gain_positive": float(np.mean(r2_gain > 0)),
            "mae_reduction_95_percentile_interval": np.quantile(
                mae_reduction, [0.025, 0.975]
            ).tolist(),
        },
        "relation_block_sign_flip": {
            "replicates": PERMUTATION_REPLICATES,
            "observed_sse_improvement": observed_sse_improvement,
            "one_sided_p": permutation_p,
        },
    }
    output_path = REPO_ROOT / "reports/v3a_cluster_robustness.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
