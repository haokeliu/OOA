#!/usr/bin/env python3
"""Probability-matched hard-versus-soft action mechanism ablation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import candidate_gain, feature_matrices
from run_v4b_action_grid import bce_gain, soft_action, tree_classifier, tree_regressor
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
)

from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
DATASETS = ("modelval", "calib")
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260911


def expected_endpoint_gain(
    probability: np.ndarray, base: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    base_risk = np.logaddexp(0.0, base) - probability * base
    full = base + residual
    full_risk = np.logaddexp(0.0, full) - probability * full
    return base_risk - full_risk


def bootstrap(values: np.ndarray, blocks: list[np.ndarray], generator) -> dict:
    replicates = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen_blocks = generator.integers(0, len(blocks), size=len(blocks))
        chosen = np.concatenate([blocks[index] for index in chosen_blocks])
        replicates[replicate] = values[chosen].mean()
    return {
        "mean": float(values.mean()),
        "relation_cluster_95_percentile_interval": np.quantile(
            replicates, [0.025, 0.975]
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
    }


def main() -> int:
    started = time.perf_counter()
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    relation_rows = json.loads(
        (REPO_ROOT / "data/manifests/v3_relation_panel.json").read_text(
            encoding="utf-8"
        )
    )
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    evaluation_data = {
        split: load_feature_shards_with_ids(split) for split in DATASETS
    }
    gate_features = torch.from_numpy(gate_features_np)
    train_mask, _checkpoint_mask = internal_split(gate_ids)
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
        gate_gain = candidate_gain(gate_tensors)
        gate_sets = feature_matrices(candidate, gate_features, gate_tensors)
        evaluation = {}
        for split, (_ids, features_np, labels_np) in evaluation_data.items():
            features = torch.from_numpy(features_np)
            tensors = build_tensors(candidate, features, labels_np)
            evaluation[split] = {
                "features": features,
                "tensors": tensors,
                "gain": candidate_gain(tensors),
                "sets": feature_matrices(candidate, features, tensors),
            }
        for representation in REPRESENTATIONS:
            for relation, relation_row in enumerate(relation_rows):
                classifier = tree_classifier(seed)
                classifier.fit(
                    gate_sets[representation][train_mask, relation],
                    gate_tensors.target_labels.numpy()[train_mask, relation],
                )
                gain_model = tree_regressor(seed)
                gain_model.fit(
                    gate_sets[representation][train_mask, relation],
                    gate_gain[train_mask, relation],
                )
                for split in DATASETS:
                    item = evaluation[split]
                    features = item["sets"][representation][:, relation]
                    probability = classifier.predict_proba(features)[:, 1]
                    base = item["tensors"].base.numpy()[:, relation]
                    residual = item["tensors"].residual.numpy()[:, relation]
                    labels = item["tensors"].target_labels.numpy()[:, relation]
                    q_soft = soft_action(probability, base, residual)
                    q_hard = (expected_endpoint_gain(probability, base, residual) > 0).astype(
                        np.float64
                    )
                    predicted_gain = gain_model.predict(features)
                    q_direct = (predicted_gain > 0).astype(np.float64)
                    soft_gain = bce_gain(base, residual, labels, q_soft)
                    hard_gain = bce_gain(base, residual, labels, q_hard)
                    direct_gain = item["gain"][:, relation] * q_direct
                    rows.append(
                        {
                            "dataset": split,
                            "seed": seed,
                            "representation": representation,
                            "pair_id": relation_row["pair_id"],
                            "soft_achieved": float(soft_gain.mean()),
                            "probability_hard_achieved": float(hard_gain.mean()),
                            "direct_gain_hard_achieved": float(direct_gain.mean()),
                            "soft_minus_probability_hard": float(
                                soft_gain.mean() - hard_gain.mean()
                            ),
                            "probability_hard_minus_direct_gain_hard": float(
                                hard_gain.mean() - direct_gain.mean()
                            ),
                            "soft_interior_fraction": float(
                                np.mean((q_soft > 0) & (q_soft < 1))
                            ),
                        }
                    )

    relations = sorted({str(row["pair_id"]) for row in rows})
    generator = np.random.default_rng(RANDOM_SEED)

    def analyze(selected_rows: list[dict[str, object]]) -> dict:
        blocks = [
            np.asarray(
                [
                    index
                    for index, row in enumerate(selected_rows)
                    if row["pair_id"] == relation
                ]
            )
            for relation in relations
        ]
        soft_hard = np.asarray(
            [row["soft_minus_probability_hard"] for row in selected_rows]
        )
        hard_direct = np.asarray(
            [
                row["probability_hard_minus_direct_gain_hard"]
                for row in selected_rows
            ]
        )
        return {
            "task_count": len(selected_rows),
            "soft_minus_probability_hard": bootstrap(
                soft_hard, blocks, generator
            ),
            "probability_hard_minus_direct_gain_hard": bootstrap(
                hard_direct, blocks, generator
            ),
            "mean_soft_interior_fraction": float(
                np.mean([row["soft_interior_fraction"] for row in selected_rows])
            ),
        }

    report = {
        "schema_version": 1,
        "protocol": "v4-D-posthoc-probability-matched-soft-mechanism-ablation",
        "analysis_status": "locked post hoc mechanism analysis",
        "development_only": True,
        "test_accessed": False,
        "combined": analyze(rows),
        "by_dataset": {
            dataset: analyze([row for row in rows if row["dataset"] == dataset])
            for dataset in DATASETS
        },
        "by_representation": {
            representation: analyze(
                [row for row in rows if row["representation"] == representation]
            )
            for representation in REPRESENTATIONS
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (REPO_ROOT / "reports/v4d_soft_mechanism_ablation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
