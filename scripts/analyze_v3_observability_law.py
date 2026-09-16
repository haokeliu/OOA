#!/usr/bin/env python3
"""Test whether held-out observable opportunity predicts realized selector gain."""

from __future__ import annotations

import csv
import json
import sys
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

from audit_observable_opportunity import candidate_gain, feature_matrices
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
    predict_model,
)

from edcr.config import load_config
from edcr.observability import LogitGainMLP, RelationConditionedViewGain

SEEDS = (17, 29, 43)
TREE_REPRESENTATIONS = ("confidence", "summary", "view_logits")
NEURAL_REPRESENTATIONS = ("LogitMLP", "RCVI")


def opportunity(gain: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    oracle = float(np.mean(np.maximum(gain, 0)))
    constant = float(max(np.mean(gain), 0))
    achieved = float(np.mean((prediction > 0) * gain))
    return {
        "oracle_improvement": oracle,
        "best_constant_improvement": constant,
        "achieved_improvement": achieved,
        "recovered_oracle_fraction": achieved / oracle,
        "recovered_beyond_constant_fraction": (achieved - constant) / (oracle - constant),
    }


def neural_predictions(
    seed: int,
    candidate,
    gate_features: torch.Tensor,
    gate_tensors,
    modelval_features: torch.Tensor,
    modelval_tensors,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    checkpoint = torch.load(
        REPO_ROOT / f"runs/v3a_rcvi_seed{seed}/gain_models.pt",
        map_location="cpu",
        weights_only=False,
    )
    relation_count = len(candidate.relations)
    gate_feature_sets = feature_matrices(candidate, gate_features, gate_tensors)
    modelval_feature_sets = feature_matrices(candidate, modelval_features, modelval_tensors)
    gate_logits = (gate_feature_sets["view_logits"] - checkpoint["logit_mean"]) / checkpoint[
        "logit_scale"
    ]
    modelval_logits = (
        modelval_feature_sets["view_logits"] - checkpoint["logit_mean"]
    ) / checkpoint["logit_scale"]
    gate_numeric = (gate_tensors.numeric.numpy() - checkpoint["numeric_mean"]) / checkpoint[
        "numeric_scale"
    ]
    modelval_numeric = (
        modelval_tensors.numeric.numpy() - checkpoint["numeric_mean"]
    ) / checkpoint["numeric_scale"]
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    models = {
        "LogitMLP": LogitGainMLP(18, relation_count),
        "RCVI": RelationConditionedViewGain(
            gate_features.shape[-1], gate_features.shape[1], gate_tensors.numeric.shape[-1]
        ),
    }
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
    shared = {
        "relation_indices": torch.arange(relation_count),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    gate_inputs = {
        **shared,
        "logits": torch.from_numpy(gate_logits).float(),
        "views": gate_features,
        "numeric": torch.from_numpy(gate_numeric).float(),
    }
    modelval_inputs = {
        **shared,
        "logits": torch.from_numpy(modelval_logits).float(),
        "views": modelval_features,
        "numeric": torch.from_numpy(modelval_numeric).float(),
    }
    gate_predictions = {
        name: predict_model(
            name, model, gate_inputs, len(gate_features), relation_count
        )
        for name, model in models.items()
    }
    modelval_predictions = {
        name: predict_model(
            name, model, modelval_inputs, len(modelval_features), relation_count
        )
        for name, model in models.items()
    }
    return gate_predictions, modelval_predictions


def design_matrix(
    rows: list[dict[str, object]],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    include_observable: bool,
) -> tuple[np.ndarray, np.ndarray]:
    numeric_names = ["checkpoint_oracle", "checkpoint_constant"]
    if include_observable:
        numeric_names.append("checkpoint_observable")
    train_numeric = np.asarray(
        [[rows[index][name] for name in numeric_names] for index in train_indices],
        dtype=np.float64,
    )
    test_numeric = np.asarray(
        [[rows[index][name] for name in numeric_names] for index in test_indices],
        dtype=np.float64,
    )
    mean = train_numeric.mean(axis=0)
    scale = np.maximum(train_numeric.std(axis=0), 1e-12)
    train_numeric = (train_numeric - mean) / scale
    test_numeric = (test_numeric - mean) / scale
    representations = (*TREE_REPRESENTATIONS, *NEURAL_REPRESENTATIONS)
    train_categories = np.asarray(
        [
            [float(rows[index]["representation"] == name) for name in representations[1:]]
            + [float(rows[index]["seed"] == seed) for seed in SEEDS[1:]]
            for index in train_indices
        ]
    )
    test_categories = np.asarray(
        [
            [float(rows[index]["representation"] == name) for name in representations[1:]]
            + [float(rows[index]["seed"] == seed) for seed in SEEDS[1:]]
            for index in test_indices
        ]
    )
    return (
        np.concatenate((train_numeric, train_categories), axis=1),
        np.concatenate((test_numeric, test_categories), axis=1),
    )


def leave_one_relation_out(
    rows: list[dict[str, object]], include_observable: bool
) -> dict[str, float]:
    target = np.asarray([row["modelval_realized"] for row in rows], dtype=np.float64)
    relation_ids = sorted({str(row["pair_id"]) for row in rows})
    prediction = np.zeros_like(target)
    for relation_id in relation_ids:
        test = np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] == relation_id]
        )
        train = np.asarray(
            [index for index, row in enumerate(rows) if row["pair_id"] != relation_id]
        )
        train_x, test_x = design_matrix(rows, train, test, include_observable)
        model = Ridge(alpha=1.0)
        model.fit(train_x, target[train])
        prediction[test] = model.predict(test_x)
    return {
        "r_squared": float(r2_score(target, prediction)),
        "spearman": float(spearmanr(target, prediction).statistic),
        "mean_absolute_error": float(np.mean(np.abs(target - prediction))),
    }


def main() -> int:
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    relation_rows = json.loads(
        (REPO_ROOT / "data/manifests/v3_relation_panel.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        candidate = load_candidate(REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config)
        gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
        _modelval_ids, modelval_features_np, modelval_labels_np = load_feature_shards_with_ids(
            "modelval"
        )
        gate_features = torch.from_numpy(gate_features_np)
        modelval_features = torch.from_numpy(modelval_features_np)
        gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
        modelval_tensors = build_tensors(candidate, modelval_features, modelval_labels_np)
        gate_gain = candidate_gain(gate_tensors)
        modelval_gain = candidate_gain(modelval_tensors)
        train_mask, checkpoint_mask = internal_split(gate_ids)
        gate_feature_sets = feature_matrices(candidate, gate_features, gate_tensors)
        modelval_feature_sets = feature_matrices(
            candidate, modelval_features, modelval_tensors
        )
        gate_predictions: dict[str, np.ndarray] = {}
        modelval_predictions: dict[str, np.ndarray] = {}
        for representation in TREE_REPRESENTATIONS:
            gate_predictions[representation] = np.zeros_like(gate_gain)
            modelval_predictions[representation] = np.zeros_like(modelval_gain)
            for relation in range(len(relation_rows)):
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
                    gate_feature_sets[representation][train_mask, relation],
                    gate_gain[train_mask, relation],
                )
                gate_predictions[representation][:, relation] = model.predict(
                    gate_feature_sets[representation][:, relation]
                )
                modelval_predictions[representation][:, relation] = model.predict(
                    modelval_feature_sets[representation][:, relation]
                )
        neural_gate, neural_modelval = neural_predictions(
            seed,
            candidate,
            gate_features,
            gate_tensors,
            modelval_features,
            modelval_tensors,
        )
        gate_predictions.update(neural_gate)
        modelval_predictions.update(neural_modelval)
        for representation in (*TREE_REPRESENTATIONS, *NEURAL_REPRESENTATIONS):
            for relation, relation_row in enumerate(relation_rows):
                checkpoint = opportunity(
                    gate_gain[checkpoint_mask, relation],
                    gate_predictions[representation][checkpoint_mask, relation],
                )
                modelval = opportunity(
                    modelval_gain[:, relation],
                    modelval_predictions[representation][:, relation],
                )
                rows.append(
                    {
                        "seed": seed,
                        "pair_id": relation_row["pair_id"],
                        "representation": representation,
                        "smoothed_lift": relation_row["smoothed_lift"],
                        "checkpoint_oracle": checkpoint["oracle_improvement"],
                        "checkpoint_constant": checkpoint["best_constant_improvement"],
                        "checkpoint_observable": checkpoint["achieved_improvement"],
                        "checkpoint_recovered_fraction": checkpoint[
                            "recovered_oracle_fraction"
                        ],
                        "modelval_oracle": modelval["oracle_improvement"],
                        "modelval_realized": modelval["achieved_improvement"],
                        "modelval_recovered_fraction": modelval[
                            "recovered_oracle_fraction"
                        ],
                    }
                )
    baseline = leave_one_relation_out(rows, include_observable=False)
    observable = leave_one_relation_out(rows, include_observable=True)
    per_seed = {}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        per_seed[str(seed)] = {
            "oracle_to_realized_spearman": float(
                spearmanr(
                    [row["checkpoint_oracle"] for row in selected],
                    [row["modelval_realized"] for row in selected],
                ).statistic
            ),
            "observable_to_realized_spearman": float(
                spearmanr(
                    [row["checkpoint_observable"] for row in selected],
                    [row["modelval_realized"] for row in selected],
                ).statistic
            ),
        }
    report = {
        "schema_version": 1,
        "protocol": "v3-A-observability-law",
        "development_only": True,
        "test_accessed": False,
        "task_count": len(rows),
        "relation_count": len(relation_rows),
        "seed_count": len(SEEDS),
        "representation_count": len(TREE_REPRESENTATIONS) + len(NEURAL_REPRESENTATIONS),
        "cross_validation": "leave-one-relation-out; fixed Ridge alpha=1",
        "oracle_only": baseline,
        "oracle_plus_observable": observable,
        "r_squared_gain": observable["r_squared"] - baseline["r_squared"],
        "per_seed_correlations": per_seed,
        "passes_preregistered_rule": (
            observable["r_squared"] - baseline["r_squared"] >= 0.10
            and all(
                values["observable_to_realized_spearman"] > 0
                for values in per_seed.values()
            )
        ),
        "rows": rows,
    }
    output_path = REPO_ROOT / "reports/v3a_observability_law.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (REPO_ROOT / "reports/v3a_observability_tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
