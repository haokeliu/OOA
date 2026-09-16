#!/usr/bin/env python3
"""Five-action and analytic-soft uncertainty-hedging development study."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import feature_matrices
from run_v4a_multicandidate import (
    cross_validated_prediction,
    portfolio_summary,
)
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
)

from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
ACTION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOTSTRAP_REPLICATES = 20_000
SIGN_FLIP_REPLICATES = 100_000
RANDOM_SEED = 20260911


def bce_gain(base: np.ndarray, residual: np.ndarray, labels: np.ndarray, q) -> np.ndarray:
    base_loss = np.logaddexp(0.0, base) - labels * base
    logits = base + q * residual
    action_loss = np.logaddexp(0.0, logits) - labels * logits
    return base_loss - action_loss


def action_gains(tensors) -> np.ndarray:
    base = tensors.base.numpy()
    residual = tensors.residual.numpy()
    labels = tensors.target_labels.numpy()
    return np.stack(
        [bce_gain(base, residual, labels, scale) for scale in ACTION_SCALES],
        axis=-1,
    )


def soft_action(
    probability: np.ndarray, base: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    action = np.zeros_like(clipped)
    stable = np.abs(residual) >= 1e-8
    action[stable] = (logit(clipped[stable]) - base[stable]) / residual[stable]
    return np.clip(action, 0.0, 1.0)


def tree_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def tree_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seed,
    )


def interval(values: np.ndarray, relation_indices: list[np.ndarray], generator) -> dict:
    replicates = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_relations = generator.integers(
            0, len(relation_indices), size=len(relation_indices)
        )
        selected = np.concatenate(
            [relation_indices[index] for index in selected_relations]
        )
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
    _modelval_ids, modelval_features_np, modelval_labels_np = (
        load_feature_shards_with_ids("modelval")
    )
    gate_features = torch.from_numpy(gate_features_np)
    modelval_features = torch.from_numpy(modelval_features_np)
    train_mask, checkpoint_mask = internal_split(gate_ids)
    rows: list[dict[str, object]] = []
    maximum_oracle_difference = 0.0

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
                checkpoint_scores = np.zeros((int(checkpoint_mask.sum()), 5))
                modelval_scores = np.zeros((len(modelval_features), 5))
                for action_index in range(1, len(ACTION_SCALES)):
                    model = tree_regressor(seed)
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
                classifier = tree_classifier(seed)
                classifier.fit(
                    gate_sets[representation][train_mask, relation],
                    gate_tensors.target_labels.numpy()[train_mask, relation],
                )
                checkpoint_probability = classifier.predict_proba(
                    gate_sets[representation][checkpoint_mask, relation]
                )[:, 1]
                modelval_probability = classifier.predict_proba(
                    modelval_sets[representation][:, relation]
                )[:, 1]
                checkpoint_q = soft_action(
                    checkpoint_probability,
                    gate_tensors.base.numpy()[checkpoint_mask, relation],
                    gate_tensors.residual.numpy()[checkpoint_mask, relation],
                )
                modelval_q = soft_action(
                    modelval_probability,
                    modelval_tensors.base.numpy()[:, relation],
                    modelval_tensors.residual.numpy()[:, relation],
                )
                checkpoint_soft_gain = bce_gain(
                    gate_tensors.base.numpy()[checkpoint_mask, relation],
                    gate_tensors.residual.numpy()[checkpoint_mask, relation],
                    gate_tensors.target_labels.numpy()[checkpoint_mask, relation],
                    checkpoint_q,
                )
                modelval_soft_gain = bce_gain(
                    modelval_tensors.base.numpy()[:, relation],
                    modelval_tensors.residual.numpy()[:, relation],
                    modelval_tensors.target_labels.numpy()[:, relation],
                    modelval_q,
                )
                checkpoint_five = portfolio_summary(
                    gate_gains[checkpoint_mask, relation], checkpoint_scores
                )
                modelval_five = portfolio_summary(
                    modelval_gains[:, relation], modelval_scores
                )
                endpoint_indices = (0, 4)
                checkpoint_binary = portfolio_summary(
                    gate_gains[checkpoint_mask, relation][:, endpoint_indices],
                    checkpoint_scores[:, endpoint_indices],
                )
                modelval_binary = portfolio_summary(
                    modelval_gains[:, relation][:, endpoint_indices],
                    modelval_scores[:, endpoint_indices],
                )
                checkpoint_oracle_difference = abs(
                    checkpoint_five["oracle"] - checkpoint_binary["oracle"]
                )
                modelval_oracle_difference = abs(
                    modelval_five["oracle"] - modelval_binary["oracle"]
                )
                maximum_oracle_difference = max(
                    maximum_oracle_difference,
                    checkpoint_oracle_difference,
                    modelval_oracle_difference,
                )
                rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "pair_id": relation_row["pair_id"],
                        "checkpoint_oracle_headroom": checkpoint_five[
                            "oracle_headroom"
                        ],
                        "checkpoint_constant": checkpoint_five["constant"],
                        "checkpoint_achieved_above_constant": checkpoint_five[
                            "achieved_above_constant"
                        ],
                        "checkpoint_five_achieved": checkpoint_five["achieved"],
                        "checkpoint_binary_achieved": checkpoint_binary["achieved"],
                        "checkpoint_soft_achieved": float(
                            checkpoint_soft_gain.mean()
                        ),
                        "modelval_oracle_headroom": modelval_five[
                            "oracle_headroom"
                        ],
                        "modelval_constant": modelval_five["constant"],
                        "modelval_achieved_above_constant": modelval_five[
                            "achieved_above_constant"
                        ],
                        "modelval_five_achieved": modelval_five["achieved"],
                        "modelval_binary_achieved": modelval_binary["achieved"],
                        "modelval_soft_achieved": float(modelval_soft_gain.mean()),
                        "modelval_interior_action_fraction": float(
                            1
                            - modelval_five["action_fractions"][0]
                            - modelval_five["action_fractions"][-1]
                        ),
                        "modelval_mean_soft_q": float(modelval_q.mean()),
                        "modelval_soft_interior_fraction": float(
                            np.mean((modelval_q > 0) & (modelval_q < 1))
                        ),
                    }
                )

    if maximum_oracle_difference > 1e-8:
        raise RuntimeError(
            f"interpolated oracle differs from endpoint oracle: {maximum_oracle_difference}"
        )
    relations = sorted({str(row["pair_id"]) for row in rows})
    relation_indices = [
        np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] == relation]
        )
        for relation in relations
    ]
    generator = np.random.default_rng(RANDOM_SEED)
    five_minus_binary = np.asarray(
        [row["modelval_five_achieved"] - row["modelval_binary_achieved"] for row in rows]
    )
    soft_minus_binary = np.asarray(
        [row["modelval_soft_achieved"] - row["modelval_binary_achieved"] for row in rows]
    )
    primary = interval(five_minus_binary, relation_indices, generator)
    analytic_soft = interval(soft_minus_binary, relation_indices, generator)
    by_representation = {}
    for representation in REPRESENTATIONS:
        selected = np.asarray(
            [index for index, row in enumerate(rows) if row["representation"] == representation]
        )
        representation_relations = [
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
            "five_minus_binary": interval(
                five_minus_binary[selected], representation_relations, generator
            ),
            "soft_minus_binary": interval(
                soft_minus_binary[selected], representation_relations, generator
            ),
            "mean_five_interior_action_fraction": float(
                np.mean([rows[index]["modelval_interior_action_fraction"] for index in selected])
            ),
            "mean_soft_interior_action_fraction": float(
                np.mean([rows[index]["modelval_soft_interior_fraction"] for index in selected])
            ),
        }

    target = np.asarray(
        [row["modelval_achieved_above_constant"] for row in rows], dtype=np.float64
    )
    oracle_prediction = cross_validated_prediction(rows, False)
    observable_prediction = cross_validated_prediction(rows, True)
    oracle_r2 = float(r2_score(target, oracle_prediction))
    observable_r2 = float(r2_score(target, observable_prediction))
    meta_gain = observable_r2 - oracle_r2
    meta_bootstrap = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected_relations = generator.integers(0, len(relations), size=len(relations))
        selected = np.concatenate([relation_indices[index] for index in selected_relations])
        meta_bootstrap[replicate] = r2_score(
            target[selected], observable_prediction[selected]
        ) - r2_score(target[selected], oracle_prediction[selected])
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
        "protocol": "v4-B-action-grid-uncertainty-hedging",
        "development_only": True,
        "test_accessed": False,
        "action_scales": list(ACTION_SCALES),
        "task_count": len(rows),
        "relation_count": len(relations),
        "maximum_binary_vs_grid_oracle_difference": maximum_oracle_difference,
        "primary_five_minus_binary": primary,
        "passes_preregistered_primary": bool(
            primary["mean"] > 0
            and primary["relation_cluster_95_percentile_interval"][0] > 0
        ),
        "analytic_soft_minus_binary": analytic_soft,
        "by_representation": by_representation,
        "observable_meta_prediction": {
            "oracle_only_r_squared": oracle_r2,
            "oracle_plus_observable_r_squared": observable_r2,
            "r_squared_gain": meta_gain,
            "r_squared_gain_95_percentile_interval": np.quantile(
                meta_bootstrap, [0.025, 0.975]
            ).tolist(),
            "oracle_plus_observable_spearman": float(
                spearmanr(target, observable_prediction).statistic
            ),
            "relation_block_sign_flip_p": float(
                (1 + np.sum(null_sse_gain >= observed_sse_gain))
                / (SIGN_FLIP_REPLICATES + 1)
            ),
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (REPO_ROOT / "reports/v4b_action_grid.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
