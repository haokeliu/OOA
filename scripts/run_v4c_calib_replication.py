#!/usr/bin/env python3
"""Partition-isolated COCO calib replication of analytic soft routing."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

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
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260911


def interval(values: np.ndarray, blocks: list[np.ndarray], generator) -> dict:
    replicates = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_blocks = generator.integers(0, len(blocks), size=len(blocks))
        selected = np.concatenate([blocks[index] for index in selected_blocks])
        replicates[replicate] = values[selected].mean()
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
    _calib_ids, calib_features_np, calib_labels_np = load_feature_shards_with_ids("calib")
    gate_features = torch.from_numpy(gate_features_np)
    calib_features = torch.from_numpy(calib_features_np)
    train_mask, checkpoint_mask = internal_split(gate_ids)
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
        calib_tensors = build_tensors(candidate, calib_features, calib_labels_np)
        gate_gain = candidate_gain(gate_tensors)
        calib_gain = candidate_gain(calib_tensors)
        gate_sets = feature_matrices(candidate, gate_features, gate_tensors)
        calib_sets = feature_matrices(candidate, calib_features, calib_tensors)
        for representation in REPRESENTATIONS:
            for relation, relation_row in enumerate(relation_rows):
                gain_model = tree_regressor(seed)
                gain_model.fit(
                    gate_sets[representation][train_mask, relation],
                    gate_gain[train_mask, relation],
                )
                checkpoint_predicted_gain = gain_model.predict(
                    gate_sets[representation][checkpoint_mask, relation]
                )
                calib_predicted_gain = gain_model.predict(
                    calib_sets[representation][:, relation]
                )
                classifier = tree_classifier(seed)
                classifier.fit(
                    gate_sets[representation][train_mask, relation],
                    gate_tensors.target_labels.numpy()[train_mask, relation],
                )
                checkpoint_probability = classifier.predict_proba(
                    gate_sets[representation][checkpoint_mask, relation]
                )[:, 1]
                calib_probability = classifier.predict_proba(
                    calib_sets[representation][:, relation]
                )[:, 1]
                checkpoint_q = soft_action(
                    checkpoint_probability,
                    gate_tensors.base.numpy()[checkpoint_mask, relation],
                    gate_tensors.residual.numpy()[checkpoint_mask, relation],
                )
                calib_q = soft_action(
                    calib_probability,
                    calib_tensors.base.numpy()[:, relation],
                    calib_tensors.residual.numpy()[:, relation],
                )
                checkpoint_soft_gain = bce_gain(
                    gate_tensors.base.numpy()[checkpoint_mask, relation],
                    gate_tensors.residual.numpy()[checkpoint_mask, relation],
                    gate_tensors.target_labels.numpy()[checkpoint_mask, relation],
                    checkpoint_q,
                )
                calib_soft_gain = bce_gain(
                    calib_tensors.base.numpy()[:, relation],
                    calib_tensors.residual.numpy()[:, relation],
                    calib_tensors.target_labels.numpy()[:, relation],
                    calib_q,
                )
                checkpoint_binary_gain = gate_gain[checkpoint_mask, relation] * (
                    checkpoint_predicted_gain > 0
                )
                calib_binary_gain = calib_gain[:, relation] * (
                    calib_predicted_gain > 0
                )
                calib_constant = max(float(calib_gain[:, relation].mean()), 0.0)
                rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "pair_id": relation_row["pair_id"],
                        "checkpoint_binary_achieved": float(
                            checkpoint_binary_gain.mean()
                        ),
                        "checkpoint_soft_achieved": float(
                            checkpoint_soft_gain.mean()
                        ),
                        "checkpoint_soft_minus_binary": float(
                            checkpoint_soft_gain.mean()
                            - checkpoint_binary_gain.mean()
                        ),
                        "calib_binary_achieved": float(calib_binary_gain.mean()),
                        "calib_soft_achieved": float(calib_soft_gain.mean()),
                        "calib_soft_minus_binary": float(
                            calib_soft_gain.mean() - calib_binary_gain.mean()
                        ),
                        "calib_constant": calib_constant,
                        "calib_soft_above_constant": float(
                            calib_soft_gain.mean() - calib_constant
                        ),
                        "calib_soft_interior_fraction": float(
                            np.mean((calib_q > 0) & (calib_q < 1))
                        ),
                    }
                )

    relations = sorted({str(row["pair_id"]) for row in rows})
    relation_indices = [
        np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] == relation]
        )
        for relation in relations
    ]
    generator = np.random.default_rng(RANDOM_SEED)
    advantage = np.asarray([row["calib_soft_minus_binary"] for row in rows])
    primary = interval(advantage, relation_indices, generator)
    by_representation = {}
    for representation in REPRESENTATIONS:
        selected = np.asarray(
            [index for index, row in enumerate(rows) if row["representation"] == representation]
        )
        blocks = [
            np.asarray(
                [
                    local_index
                    for local_index, index in enumerate(selected)
                    if rows[index]["pair_id"] == relation
                ]
            )
            for relation in relations
        ]
        by_representation[representation] = {
            "soft_minus_binary": interval(advantage[selected], blocks, generator),
            "mean_soft_interior_fraction": float(
                np.mean([rows[index]["calib_soft_interior_fraction"] for index in selected])
            ),
            "mean_soft_above_constant": float(
                np.mean([rows[index]["calib_soft_above_constant"] for index in selected])
            ),
        }
    checkpoint_advantage = np.asarray(
        [row["checkpoint_soft_minus_binary"] for row in rows]
    )
    report = {
        "schema_version": 1,
        "protocol": "v4-C-calib-analytic-soft-internal-replication",
        "status": "historically-label-accessible but V4-isolated internal replication",
        "development_only": True,
        "test_accessed": False,
        "task_count": len(rows),
        "relation_count": len(relations),
        "calib_sample_count": len(calib_features),
        "primary_soft_minus_binary": primary,
        "passes_preregistered_primary": bool(
            primary["mean"] > 0
            and primary["relation_cluster_95_percentile_interval"][0] > 0
        ),
        "checkpoint_to_calib_advantage_spearman": float(
            spearmanr(checkpoint_advantage, advantage).statistic
        ),
        "by_representation": by_representation,
        "by_seed": {
            str(seed): {
                "mean_soft_minus_binary": float(
                    np.mean(
                        [
                            row["calib_soft_minus_binary"]
                            for row in rows
                            if row["seed"] == seed
                        ]
                    )
                )
            }
            for seed in SEEDS
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (REPO_ROOT / "reports/v4c_calib_replication.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
