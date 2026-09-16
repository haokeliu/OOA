#!/usr/bin/env python3
"""Relation-block uncertainty for the V3-E tree sample-efficiency study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260911


def main() -> int:
    source = json.loads(
        (REPO_ROOT / "reports/v3e_tree_sample_efficiency.json").read_text(
            encoding="utf-8"
        )
    )
    rows = source["rows"]
    relations = sorted({str(row["pair_id"]) for row in rows})
    fractions = source["fractions"]
    seeds = (17, 29, 43)
    lookup = {
        (
            str(row["pair_id"]),
            int(row["seed"]),
            float(row["fraction"]),
            str(row["representation"]),
        ): float(row["modelval_realized"])
        for row in rows
    }
    generator = np.random.default_rng(RANDOM_SEED)
    paired_differences = {}
    difference_arrays = {}
    for fraction in fractions:
        values = np.asarray(
            [
                [
                    lookup[(relation, seed, fraction, "view_logits")]
                    - lookup[(relation, seed, fraction, "summary")]
                    for seed in seeds
                ]
                for relation in relations
            ]
        )
        difference_arrays[fraction] = values
        replicates = np.empty(BOOTSTRAP_REPLICATES)
        for replicate in range(BOOTSTRAP_REPLICATES):
            selected = generator.integers(0, len(relations), size=len(relations))
            replicates[replicate] = values[selected].mean()
        paired_differences[str(fraction)] = {
            "mean_view_logits_minus_summary": float(values.mean()),
            "relation_cluster_95_percentile_interval": np.quantile(
                replicates, [0.025, 0.975]
            ).tolist(),
            "fraction_positive": float(np.mean(replicates > 0)),
        }

    interaction = difference_arrays[1.0] - difference_arrays[0.1]
    interaction_replicates = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, len(relations), size=len(relations))
        interaction_replicates[replicate] = interaction[selected].mean()
    report = {
        "schema_version": 1,
        "protocol": "v3-E-tree-sample-efficiency-analysis",
        "development_only": True,
        "test_accessed": False,
        "relation_count": len(relations),
        "seeds": list(seeds),
        "paired_view_logits_minus_summary": paired_differences,
        "sample_size_interaction": {
            "contrast": "(view_logits-summary at 100%) - (view_logits-summary at 10%)",
            "mean": float(interaction.mean()),
            "relation_cluster_95_percentile_interval": np.quantile(
                interaction_replicates, [0.025, 0.975]
            ).tolist(),
            "fraction_positive": float(np.mean(interaction_replicates > 0)),
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RANDOM_SEED,
            "cluster": "relation; all seeds retained within a relation",
        },
    }
    (REPO_ROOT / "reports/v3e_tree_sample_efficiency_analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
