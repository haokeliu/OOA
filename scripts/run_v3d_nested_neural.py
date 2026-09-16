#!/usr/bin/env python3
"""Nested-early-stopping audit for V3 neural selectors on COCO development data."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_v3_observability_law import opportunity
from audit_observable_opportunity import candidate_gain, feature_matrices
from train_gate import load_candidate
from train_v2b_gain_models import (
    PAIR_BATCH_SIZE,
    TARGET_SCALE,
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
    predict_model,
    standardizer,
    train_model,
)

from edcr.config import load_config
from edcr.observability import LogitGainMLP, RelationConditionedViewGain

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("LogitMLP", "RCVI")


def inner_split(sample_ids: np.ndarray, outer_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residues = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"v3d-inner:{sample_id}".encode()).digest()[:8]
            )
            % 5
            for sample_id in sample_ids.astype(str)
        ]
    )
    return outer_train & (residues != 0), outer_train & (residues == 0)


def make_model(
    representation: str,
    seed: int,
    relation_count: int,
    feature_dim: int,
    view_count: int,
    numeric_dim: int,
) -> nn.Module:
    offset = REPRESENTATIONS.index(representation) * 10_000
    torch.manual_seed(seed + offset)
    if representation == "LogitMLP":
        return LogitGainMLP(18, relation_count)
    return RelationConditionedViewGain(feature_dim, view_count, numeric_dim)


def model_output(
    representation: str,
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    indices: torch.Tensor,
) -> torch.Tensor:
    if representation == "LogitMLP":
        return model(inputs["logits"][indices], inputs["relation_indices"])
    return model(
        inputs["views"][indices],
        inputs["numeric"][indices],
        inputs["source_queries"],
        inputs["target_queries"],
    )


def refit_fixed_epochs(
    representation: str,
    model: nn.Module,
    train_mask: np.ndarray,
    inputs: dict[str, torch.Tensor],
    gain: torch.Tensor,
    seed: int,
    epochs: int,
) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed + 20_000)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask))
    image_batch_size = max(1, PAIR_BATCH_SIZE // gain.shape[1])
    for _epoch in range(epochs):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for offset in range(0, len(order), image_batch_size):
            indices = order[offset : offset + image_batch_size]
            prediction = model_output(representation, model, inputs, indices)
            loss = functional.mse_loss(prediction, gain[indices] * TARGET_SCALE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def build_inputs(candidate, features: torch.Tensor, tensors, means: dict[str, np.ndarray]):
    feature_sets = feature_matrices(candidate, features, tensors)
    normalized_logits = (feature_sets["view_logits"] - means["logit_mean"]) / means[
        "logit_scale"
    ]
    normalized_numeric = (tensors.numeric.numpy() - means["numeric_mean"]) / means[
        "numeric_scale"
    ]
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    return {
        "logits": torch.from_numpy(normalized_logits).float(),
        "views": features,
        "numeric": torch.from_numpy(normalized_numeric).float(),
        "relation_indices": torch.arange(len(candidate.relations)),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }


def correlation(first: list[float], second: list[float]) -> float:
    return float(spearmanr(first, second).statistic)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    checkpoint = [float(row["checkpoint_observable"]) for row in rows]
    realized = [float(row["modelval_realized"]) for row in rows]
    legacy_checkpoint = [float(row["legacy_checkpoint_observable"]) for row in rows]
    return {
        "task_count": len(rows),
        "nested_checkpoint_to_modelval_spearman": correlation(checkpoint, realized),
        "legacy_checkpoint_to_modelval_spearman": correlation(
            legacy_checkpoint, realized
        ),
        "nested_checkpoint_mean_absolute_error": float(
            np.mean(np.abs(np.asarray(checkpoint) - np.asarray(realized)))
        ),
        "mean_modelval_realized": float(np.mean(realized)),
        "mean_modelval_recovered_oracle_fraction": float(
            np.mean([row["modelval_recovered_fraction"] for row in rows])
        ),
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
    outer_train, outer_checkpoint = internal_split(gate_ids)
    tuning_train, tuning_validation = inner_split(gate_ids, outer_train)
    rows: list[dict[str, object]] = []
    training_records: list[dict[str, object]] = []

    legacy_source = json.loads(
        (REPO_ROOT / "reports/v3a_observability_law.json").read_text(
            encoding="utf-8"
        )
    )
    legacy = {
        (row["seed"], row["representation"], row["pair_id"]): row
        for row in legacy_source["rows"]
    }

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
        modelval_tensors = build_tensors(
            candidate, modelval_features, modelval_labels_np
        )
        gate_gain_np = candidate_gain(gate_tensors)
        modelval_gain_np = candidate_gain(modelval_tensors)
        gate_gain = torch.from_numpy(gate_gain_np)
        gate_feature_sets = feature_matrices(candidate, gate_features, gate_tensors)
        logit_mean, logit_scale = standardizer(
            gate_feature_sets["view_logits"], outer_train
        )
        numeric_mean, numeric_scale = standardizer(
            gate_tensors.numeric.numpy(), outer_train
        )
        means = {
            "logit_mean": logit_mean,
            "logit_scale": logit_scale,
            "numeric_mean": numeric_mean,
            "numeric_scale": numeric_scale,
        }
        gate_inputs = build_inputs(candidate, gate_features, gate_tensors, means)
        modelval_inputs = build_inputs(
            candidate, modelval_features, modelval_tensors, means
        )
        output_dir = REPO_ROOT / "runs" / f"v3d_nested_neural_seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=False)
        final_states = {}
        with (output_dir / "tuning_log.jsonl").open("w", encoding="utf-8") as log_handle:
            for representation in REPRESENTATIONS:
                tuning_model = make_model(
                    representation,
                    seed,
                    len(candidate.relations),
                    gate_features.shape[-1],
                    gate_features.shape[1],
                    gate_tensors.numeric.shape[-1],
                )
                _tuned_model, tuning = train_model(
                    representation,
                    tuning_model,
                    tuning_train,
                    tuning_validation,
                    gate_inputs,
                    gate_gain,
                    seed,
                    log_handle,
                )
                selected_epochs = int(tuning["best_epoch"]) + 1
                final_model = make_model(
                    representation,
                    seed,
                    len(candidate.relations),
                    gate_features.shape[-1],
                    gate_features.shape[1],
                    gate_tensors.numeric.shape[-1],
                )
                final_model = refit_fixed_epochs(
                    representation,
                    final_model,
                    outer_train,
                    gate_inputs,
                    gate_gain,
                    seed,
                    selected_epochs,
                )
                gate_prediction = predict_model(
                    representation,
                    final_model,
                    gate_inputs,
                    len(gate_features),
                    len(candidate.relations),
                )
                modelval_prediction = predict_model(
                    representation,
                    final_model,
                    modelval_inputs,
                    len(modelval_features),
                    len(candidate.relations),
                )
                final_states[representation] = final_model.state_dict()
                training_records.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "selected_epochs": selected_epochs,
                        "inner_validation_mse": tuning["internal_validation_mse"],
                    }
                )
                for relation, relation_row in enumerate(relation_rows):
                    checkpoint = opportunity(
                        gate_gain_np[outer_checkpoint, relation],
                        gate_prediction[outer_checkpoint, relation],
                    )
                    modelval = opportunity(
                        modelval_gain_np[:, relation],
                        modelval_prediction[:, relation],
                    )
                    legacy_row = legacy[(seed, representation, relation_row["pair_id"])]
                    rows.append(
                        {
                            "seed": seed,
                            "representation": representation,
                            "pair_id": relation_row["pair_id"],
                            "checkpoint_oracle": checkpoint["oracle_improvement"],
                            "checkpoint_observable": checkpoint["achieved_improvement"],
                            "checkpoint_recovered_fraction": checkpoint[
                                "recovered_oracle_fraction"
                            ],
                            "modelval_oracle": modelval["oracle_improvement"],
                            "modelval_realized": modelval["achieved_improvement"],
                            "modelval_recovered_fraction": modelval[
                                "recovered_oracle_fraction"
                            ],
                            "legacy_checkpoint_observable": legacy_row[
                                "checkpoint_observable"
                            ],
                            "legacy_modelval_realized": legacy_row["modelval_realized"],
                        }
                    )
        torch.save(
            {
                "models": final_states,
                **means,
                "outer_checkpoint_labels_used": False,
            },
            output_dir / "gain_models.pt",
        )
        (output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "protocol": "v3-D-nested-neural-checkpoint",
                    "development_only": True,
                    "test_accessed": False,
                    "seed": seed,
                    "outer_train_count": int(outer_train.sum()),
                    "outer_checkpoint_count": int(outer_checkpoint.sum()),
                    "inner_train_count": int(tuning_train.sum()),
                    "inner_validation_count": int(tuning_validation.sum()),
                    "training": [
                        record for record in training_records if record["seed"] == seed
                    ],
                    "rows": [row for row in rows if row["seed"] == seed],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    report = {
        "schema_version": 1,
        "protocol": "v3-D-nested-neural-checkpoint",
        "development_only": True,
        "test_accessed": False,
        "outer_checkpoint_labels_used_for_training_or_selection": False,
        "split_counts": {
            "outer_train": int(outer_train.sum()),
            "outer_checkpoint": int(outer_checkpoint.sum()),
            "inner_train": int(tuning_train.sum()),
            "inner_validation": int(tuning_validation.sum()),
        },
        "training": training_records,
        "overall": summarize(rows),
        "by_representation": {
            representation: summarize(
                [row for row in rows if row["representation"] == representation]
            )
            for representation in REPRESENTATIONS
        },
        "by_seed": {
            str(seed): summarize([row for row in rows if row["seed"] == seed])
            for seed in SEEDS
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (REPO_ROOT / "reports/v3d_nested_neural_checkpoint.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
