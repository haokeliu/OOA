#!/usr/bin/env python3
"""Strict-checkpoint three-action observable-opportunity development study."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import feature_matrices
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
)

from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
ACTION_SCALES = (0.0, 0.5, 1.0)
BOOTSTRAP_REPLICATES = 20_000
SIGN_FLIP_REPLICATES = 100_000
RANDOM_SEED = 20260911


def action_gains(tensors) -> np.ndarray:
    base_loss = functional.binary_cross_entropy_with_logits(
        tensors.base, tensors.target_labels, reduction="none"
    )
    gains = []
    for scale in ACTION_SCALES:
        action_loss = functional.binary_cross_entropy_with_logits(
            tensors.base + scale * tensors.residual,
            tensors.target_labels,
            reduction="none",
        )
        gains.append((base_loss - action_loss).numpy())
    return np.stack(gains, axis=-1)


def portfolio_summary(gains: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    action = np.argmax(scores, axis=-1)
    achieved = float(np.take_along_axis(gains, action[:, None], axis=1).mean())
    oracle = float(np.max(gains, axis=1).mean())
    action_means = gains.mean(axis=0)
    constant_action = int(np.argmax(action_means))
    constant = float(action_means[constant_action])
    headroom = oracle - constant
    return {
        "oracle": oracle,
        "constant": constant,
        "constant_action": constant_action,
        "oracle_headroom": headroom,
        "achieved": achieved,
        "achieved_above_constant": achieved - constant,
        "recovered_headroom_fraction": (
            (achieved - constant) / headroom if headroom > 0 else None
        ),
        "action_fractions": [float(np.mean(action == index)) for index in range(scores.shape[1])],
    }


def design_matrix(
    rows: list[dict[str, object]],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    include_observable: bool,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_names = ["checkpoint_oracle_headroom", "checkpoint_constant"]
    if include_observable:
        numeric_names.append("checkpoint_achieved_above_constant")
    train_numeric = np.asarray(
        [[rows[index][name] for name in numeric_names] for index in train_indices]
    )
    test_numeric = np.asarray(
        [[rows[index][name] for name in numeric_names] for index in test_indices]
    )
    mean = train_numeric.mean(axis=0)
    scale = np.maximum(train_numeric.std(axis=0), 1e-12)
    train_numeric = (train_numeric - mean) / scale
    test_numeric = (test_numeric - mean) / scale
    train_categories = np.asarray(
        [
            [float(rows[index]["representation"] == name) for name in REPRESENTATIONS[1:]]
            + [float(rows[index]["seed"] == seed) for seed in SEEDS[1:]]
            for index in train_indices
        ]
    )
    test_categories = np.asarray(
        [
            [float(rows[index]["representation"] == name) for name in REPRESENTATIONS[1:]]
            + [float(rows[index]["seed"] == seed) for seed in SEEDS[1:]]
            for index in test_indices
        ]
    )
    return (
        np.concatenate((train_numeric, train_categories), axis=1),
        np.concatenate((test_numeric, test_categories), axis=1),
    )


def cross_validated_prediction(
    rows: list[dict[str, object]], include_observable: bool
) -> np.ndarray:
    target = np.asarray(
        [row["modelval_achieved_above_constant"] for row in rows], dtype=np.float64
    )
    prediction = np.zeros_like(target)
    for relation in sorted({str(row["pair_id"]) for row in rows}):
        test = np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] == relation]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] != relation]
        )
        train_x, test_x = design_matrix(rows, train, test, include_observable)
        model = Ridge(alpha=1.0).fit(train_x, target[train])
        prediction[test] = model.predict(test_x)
    return prediction


def main() -> int:
    started = time.perf_counter()
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    relation_rows = json.loads(
        (REPO_ROOT / "data/manifests/v3_relation_panel.json").read_text(
            encoding="utf-8"
        )
    )
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    _modelval_ids, modelval_features_np, modelval_labels_np = (
        load_feature_shards_with_ids("modelval")
    )
    gate_features = torch.from_numpy(gate_features_np)
    modelval_features = torch.from_numpy(modelval_features_np)
    train_mask, checkpoint_mask = internal_split(gate_ids)
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
        modelval_tensors = build_tensors(
            candidate, modelval_features, modelval_labels_np
        )
        gate_gains = action_gains(gate_tensors)
        modelval_gains = action_gains(modelval_tensors)
        gate_sets = feature_matrices(candidate, gate_features, gate_tensors)
        modelval_sets = feature_matrices(
            candidate, modelval_features, modelval_tensors
        )
        for representation in REPRESENTATIONS:
            for relation, relation_row in enumerate(relation_rows):
                checkpoint_scores = np.zeros((int(checkpoint_mask.sum()), 3))
                modelval_scores = np.zeros((len(modelval_features), 3))
                for action_index in (1, 2):
                    model = HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.05,
                        max_iter=100,
                        max_leaf_nodes=15,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        random_state=seed,
                    )
                    model.fit(
                        gate_sets[representation][train_mask, relation],
                        gate_gains[train_mask, relation, action_index],
                    )
                    checkpoint_scores[:, action_index] = model.predict(
                        gate_sets[representation][checkpoint_mask, relation]
                    )
                    modelval_scores[:, action_index] = model.predict(
                        modelval_sets[representation][:, relation]
                    )
                checkpoint_three = portfolio_summary(
                    gate_gains[checkpoint_mask, relation], checkpoint_scores
                )
                modelval_three = portfolio_summary(
                    modelval_gains[:, relation], modelval_scores
                )
                binary_indices = (0, 2)
                checkpoint_binary = portfolio_summary(
                    gate_gains[checkpoint_mask, relation][:, binary_indices],
                    checkpoint_scores[:, binary_indices],
                )
                modelval_binary = portfolio_summary(
                    modelval_gains[:, relation][:, binary_indices],
                    modelval_scores[:, binary_indices],
                )
                rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "pair_id": relation_row["pair_id"],
                        "checkpoint_oracle_headroom": checkpoint_three[
                            "oracle_headroom"
                        ],
                        "checkpoint_constant": checkpoint_three["constant"],
                        "checkpoint_achieved_above_constant": checkpoint_three[
                            "achieved_above_constant"
                        ],
                        "checkpoint_recovered_headroom_fraction": checkpoint_three[
                            "recovered_headroom_fraction"
                        ],
                        "checkpoint_action_fractions": checkpoint_three[
                            "action_fractions"
                        ],
                        "modelval_oracle_headroom": modelval_three["oracle_headroom"],
                        "modelval_constant": modelval_three["constant"],
                        "modelval_achieved_above_constant": modelval_three[
                            "achieved_above_constant"
                        ],
                        "modelval_recovered_headroom_fraction": modelval_three[
                            "recovered_headroom_fraction"
                        ],
                        "modelval_action_fractions": modelval_three[
                            "action_fractions"
                        ],
                        "modelval_three_minus_binary_achieved": modelval_three[
                            "achieved"
                        ]
                        - modelval_binary["achieved"],
                        "modelval_three_minus_binary_oracle": modelval_three["oracle"]
                        - modelval_binary["oracle"],
                        "checkpoint_three_minus_binary_achieved": checkpoint_three[
                            "achieved"
                        ]
                        - checkpoint_binary["achieved"],
                    }
                )

    target = np.asarray(
        [row["modelval_achieved_above_constant"] for row in rows], dtype=np.float64
    )
    oracle_prediction = cross_validated_prediction(rows, False)
    observable_prediction = cross_validated_prediction(rows, True)
    oracle_r2 = float(r2_score(target, oracle_prediction))
    observable_r2 = float(r2_score(target, observable_prediction))
    r2_gain_observed = observable_r2 - oracle_r2
    relations = sorted({str(row["pair_id"]) for row in rows})
    relation_indices = [
        np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] == relation]
        )
        for relation in relations
    ]
    generator = np.random.default_rng(RANDOM_SEED)
    r2_gain = np.empty(BOOTSTRAP_REPLICATES)
    three_minus_binary = np.empty(BOOTSTRAP_REPLICATES)
    three_binary_rows = np.asarray(
        [row["modelval_three_minus_binary_achieved"] for row in rows]
    )
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_relations = generator.integers(0, len(relations), size=len(relations))
        selected = np.concatenate([relation_indices[index] for index in selected_relations])
        r2_gain[replicate] = r2_score(
            target[selected], observable_prediction[selected]
        ) - r2_score(target[selected], oracle_prediction[selected])
        three_minus_binary[replicate] = three_binary_rows[selected].mean()
    per_relation_sse_gain = np.asarray(
        [
            np.sum((target[index] - oracle_prediction[index]) ** 2)
            - np.sum((target[index] - observable_prediction[index]) ** 2)
            for index in relation_indices
        ]
    )
    observed_sse_gain = float(per_relation_sse_gain.sum())
    null_sse_gain = (
        generator.choice(
            (-1.0, 1.0), size=(SIGN_FLIP_REPLICATES, len(relations))
        )
        @ per_relation_sse_gain
    )
    report = {
        "schema_version": 1,
        "protocol": "v4-A-strict-three-action-observable-opportunity",
        "development_only": True,
        "test_accessed": False,
        "action_scales": list(ACTION_SCALES),
        "task_count": len(rows),
        "relation_count": len(relations),
        "outer_train_count": int(train_mask.sum()),
        "outer_checkpoint_count": int(checkpoint_mask.sum()),
        "cross_validation": "leave-one-relation-out Ridge alpha=1",
        "oracle_only": {
            "r_squared": oracle_r2,
            "spearman": float(spearmanr(target, oracle_prediction).statistic),
            "mean_absolute_error": float(np.mean(np.abs(target - oracle_prediction))),
        },
        "oracle_plus_observable": {
            "r_squared": observable_r2,
            "spearman": float(spearmanr(target, observable_prediction).statistic),
            "mean_absolute_error": float(
                np.mean(np.abs(target - observable_prediction))
            ),
        },
        "r_squared_gain": r2_gain_observed,
        "passes_preregistered_rule": bool(
            r2_gain_observed >= 0.10 and np.quantile(r2_gain, 0.025) > 0
        ),
        "relation_cluster_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": RANDOM_SEED,
            "r_squared_gain_95_percentile_interval": np.quantile(
                r2_gain, [0.025, 0.975]
            ).tolist(),
            "three_minus_binary_achieved_95_percentile_interval": np.quantile(
                three_minus_binary, [0.025, 0.975]
            ).tolist(),
        },
        "relation_block_sign_flip": {
            "replicates": SIGN_FLIP_REPLICATES,
            "one_sided_p": float(
                (1 + np.sum(null_sse_gain >= observed_sse_gain))
                / (SIGN_FLIP_REPLICATES + 1)
            ),
        },
        "portfolio_summary": {
            "mean_checkpoint_achieved_above_constant": float(
                np.mean([row["checkpoint_achieved_above_constant"] for row in rows])
            ),
            "mean_modelval_achieved_above_constant": float(np.mean(target)),
            "mean_modelval_oracle_headroom": float(
                np.mean([row["modelval_oracle_headroom"] for row in rows])
            ),
            "mean_modelval_recovered_headroom_fraction": float(
                np.mean(
                    [row["modelval_recovered_headroom_fraction"] for row in rows]
                )
            ),
            "mean_modelval_three_minus_binary_achieved": float(
                three_binary_rows.mean()
            ),
            "mean_modelval_three_minus_binary_oracle": float(
                np.mean([row["modelval_three_minus_binary_oracle"] for row in rows])
            ),
            "mean_modelval_intermediate_action_fraction": float(
                np.mean([row["modelval_action_fractions"][1] for row in rows])
            ),
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (REPO_ROOT / "reports/v4a_multicandidate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
