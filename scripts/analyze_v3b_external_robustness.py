#!/usr/bin/env python3
"""Post-freeze dependence-aware robustness analyses for V3-B."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260910


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def rank_correlation(x_rank: np.ndarray, y_rank: np.ndarray) -> float:
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    return float(
        np.dot(x_centered, y_centered)
        / np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )


def main() -> int:
    source = json.loads(
        (REPO_ROOT / "reports/v3b_voc_external_confirmation.json").read_text(
            encoding="utf-8"
        )
    )
    rows = source["rows"]
    relations = sorted({row["pair_id"] for row in rows})
    tasks = sorted({(row["seed"], row["representation"]) for row in rows})
    lookup = {
        (row["pair_id"], row["seed"], row["representation"]): row for row in rows
    }
    arrays = {}
    for field in ("checkpoint_oracle", "checkpoint_observable", "test_realized"):
        arrays[field] = np.asarray(
            [
                [lookup[(relation, seed, representation)][field] for seed, representation in tasks]
                for relation in relations
            ],
            dtype=np.float64,
        )
    oracle_flat = arrays["checkpoint_oracle"].reshape(-1)
    observable_flat = arrays["checkpoint_observable"].reshape(-1)
    realized_flat = arrays["test_realized"].reshape(-1)
    observed_oracle = correlation(oracle_flat, realized_flat)
    observed_observable = correlation(observable_flat, realized_flat)
    observed_advantage = observed_observable - observed_oracle

    oracle_rank = rankdata(oracle_flat)
    observable_rank = rankdata(observable_flat)
    realized_rank_blocks = rankdata(realized_flat).reshape(len(relations), len(tasks))
    null_observable = []
    null_advantage = []
    for permutation in itertools.permutations(range(len(relations))):
        permuted_rank = realized_rank_blocks[list(permutation)].reshape(-1)
        oracle_correlation = rank_correlation(oracle_rank, permuted_rank)
        observable_correlation = rank_correlation(observable_rank, permuted_rank)
        null_observable.append(observable_correlation)
        null_advantage.append(observable_correlation - oracle_correlation)
    exact_p_observable = np.mean(
        np.asarray(null_observable) >= observed_observable
    )
    exact_p_advantage = np.mean(
        np.asarray(null_advantage) >= observed_advantage
    )

    swap_advantages = []
    for swap_bits in itertools.product((False, True), repeat=len(relations)):
        swap = np.asarray(swap_bits)[:, None]
        first = np.where(swap, arrays["checkpoint_observable"], arrays["checkpoint_oracle"])
        second = np.where(swap, arrays["checkpoint_oracle"], arrays["checkpoint_observable"])
        first_correlation = correlation(first.reshape(-1), realized_flat)
        second_correlation = correlation(second.reshape(-1), realized_flat)
        swap_advantages.append(second_correlation - first_correlation)
    swap_p_advantage = np.mean(np.asarray(swap_advantages) >= observed_advantage)

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_observable = np.empty(BOOTSTRAP_REPLICATES)
    bootstrap_advantage = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, len(relations), size=len(relations))
        oracle = arrays["checkpoint_oracle"][selected].reshape(-1)
        observable = arrays["checkpoint_observable"][selected].reshape(-1)
        realized = arrays["test_realized"][selected].reshape(-1)
        oracle_correlation = correlation(oracle, realized)
        observable_correlation = correlation(observable, realized)
        bootstrap_observable[replicate] = observable_correlation
        bootstrap_advantage[replicate] = observable_correlation - oracle_correlation

    leave_one_relation_out = {}
    for relation_index, relation in enumerate(relations):
        keep = np.arange(len(relations)) != relation_index
        oracle = arrays["checkpoint_oracle"][keep].reshape(-1)
        observable = arrays["checkpoint_observable"][keep].reshape(-1)
        realized = arrays["test_realized"][keep].reshape(-1)
        oracle_correlation = correlation(oracle, realized)
        observable_correlation = correlation(observable, realized)
        leave_one_relation_out[relation] = {
            "oracle_spearman": oracle_correlation,
            "observable_spearman": observable_correlation,
            "advantage": observable_correlation - oracle_correlation,
        }

    relation_aggregated = {
        field: arrays[field].mean(axis=1)
        for field in ("checkpoint_oracle", "checkpoint_observable", "test_realized")
    }
    report = {
        "schema_version": 1,
        "protocol": "v3-B-post-freeze-dependence-robustness",
        "confirmatory_primary_unchanged": True,
        "test_labels_accessed": True,
        "analysis_status": "post-freeze robustness; not used for model selection",
        "cluster_unit": "relation; seed and representation remain within each block",
        "relation_count": len(relations),
        "within_relation_task_count": len(tasks),
        "observed": {
            "oracle_spearman": observed_oracle,
            "observable_spearman": observed_observable,
            "advantage": observed_advantage,
            "observable_mean_absolute_calibration_error": float(
                np.mean(np.abs(observable_flat - realized_flat))
            ),
        },
        "exact_relation_block_permutation": {
            "permutation_count": len(null_observable),
            "one_sided_p_observable_spearman": float(exact_p_observable),
            "one_sided_p_advantage": float(exact_p_advantage),
            "advantage_null_note": (
                "relation-label permutation tests alignment, not paired predictor superiority"
            ),
        },
        "exact_relation_block_predictor_swap": {
            "assignment_count": len(swap_advantages),
            "one_sided_p_advantage": float(swap_p_advantage),
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
        "relation_aggregated": {
            "oracle_spearman": correlation(
                relation_aggregated["checkpoint_oracle"],
                relation_aggregated["test_realized"],
            ),
            "observable_spearman": correlation(
                relation_aggregated["checkpoint_observable"],
                relation_aggregated["test_realized"],
            ),
        },
        "leave_one_relation_out": leave_one_relation_out,
    }
    output_path = REPO_ROOT / "reports/v3b_voc_external_robustness.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
