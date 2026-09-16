#!/usr/bin/env python3
"""Train the preregistered balanced risk-aligned RCVI gain model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import aggregate, candidate_gain, summarize_relation
from train_gate import load_candidate
from train_v2b_gain_models import (
    MAX_EPOCHS,
    PAIR_BATCH_SIZE,
    PATIENCE,
    TARGET_SCALE,
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
    predict_model,
    standardizer,
)

from edcr.config import load_config
from edcr.observability import RelationConditionedViewGain


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_groups(path: Path, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as groups:
        if not np.array_equal(groups["sample_id"].astype(str), sample_ids.astype(str)):
            raise ValueError(f"group sample order differs: {path}")
        return groups["negative"].astype(bool), groups["weak"].astype(bool)


def balanced_weights(
    negative: np.ndarray,
    weak: np.ndarray,
    training_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    negative_prevalence = negative[training_mask].mean(axis=0)
    weak_prevalence = weak[training_mask].mean(axis=0)
    if np.any(negative_prevalence <= 0) or np.any(weak_prevalence <= 0):
        raise ValueError("every relation must have negative and weak training support")
    weights = (
        1.0
        + negative / negative_prevalence[None, :]
        + weak / weak_prevalence[None, :]
    )
    return weights.astype(np.float32), {
        "negative": negative_prevalence.tolist(),
        "weak": weak_prevalence.tolist(),
    }


def train_model(
    model: nn.Module,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    inputs: dict[str, torch.Tensor],
    gain: torch.Tensor,
    weights: torch.Tensor,
    seed: int,
    log_handle,
) -> tuple[nn.Module, dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask))
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask))
    image_batch_size = max(1, PAIR_BATCH_SIZE // gain.shape[1])
    best_state: dict[str, torch.Tensor] | None = None
    best_mse = float("inf")
    best_epoch = -1
    stale = 0

    def predict(indices: torch.Tensor) -> torch.Tensor:
        return model(
            inputs["views"][indices],
            inputs["numeric"][indices],
            inputs["source_queries"],
            inputs["target_queries"],
        )

    def weighted_mse(predictions: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        squared = (predictions - gain[indices] * TARGET_SCALE).square()
        selected_weights = weights[indices]
        return (squared * selected_weights).sum() / selected_weights.sum()

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for offset in range(0, len(order), image_batch_size):
            indices = order[offset : offset + image_batch_size]
            loss = weighted_mse(predict(indices), indices)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat(
                [
                    predict(validation_indices[offset : offset + image_batch_size])
                    for offset in range(0, len(validation_indices), image_batch_size)
                ]
            )
            validation_mse = float(weighted_mse(predictions, validation_indices))
        log_handle.write(
            json.dumps(
                {"epoch": epoch, "internal_validation_weighted_mse": validation_mse},
                sort_keys=True,
            )
            + "\n"
        )
        log_handle.flush()
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("Balanced-RCVI produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "internal_validation_weighted_mse": best_mse}


def stratum_summary(
    gain: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    achieved = []
    oracle = []
    for relation in range(gain.shape[1]):
        selected = mask[:, relation]
        action = predictions[selected, relation] > 0
        values = gain[selected, relation]
        achieved.append(float(np.mean(action * values)))
        oracle.append(float(np.mean(np.maximum(values, 0))))
    return {
        "achieved_improvement": float(np.mean(achieved)),
        "oracle_improvement": float(np.mean(oracle)),
        "recovered_oracle_fraction": float(np.mean(achieved) / np.mean(oracle)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")
    output_dir = REPO_ROOT / "runs" / (args.run_id or f"v2d_balanced_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    candidate_dir = REPO_ROOT / "runs" / f"p2e_seed{args.seed}"
    candidate = load_candidate(candidate_dir, config)
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    modelval_ids, modelval_features_np, modelval_labels_np = load_feature_shards_with_ids(
        "modelval"
    )
    gate_negative, gate_weak = load_groups(
        REPO_ROOT / "data/groups/natural_gate.npz", gate_ids
    )
    modelval_negative, modelval_weak = load_groups(
        REPO_ROOT / "data/groups/natural_modelval.npz", modelval_ids
    )
    gate_features = torch.from_numpy(gate_features_np)
    modelval_features = torch.from_numpy(modelval_features_np)
    gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
    modelval_tensors = build_tensors(candidate, modelval_features, modelval_labels_np)
    gate_gain = torch.from_numpy(candidate_gain(gate_tensors))
    modelval_gain = candidate_gain(modelval_tensors)
    train_mask, validation_mask = internal_split(gate_ids)
    weights_np, prevalence = balanced_weights(
        gate_negative, gate_weak, train_mask
    )
    numeric_mean, numeric_scale = standardizer(gate_tensors.numeric.numpy(), train_mask)
    gate_numeric = (gate_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    modelval_numeric = (modelval_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    inputs = {
        "views": gate_features,
        "numeric": torch.from_numpy(gate_numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    torch.manual_seed(args.seed)
    model = RelationConditionedViewGain(
        gate_features.shape[-1], gate_features.shape[1], gate_tensors.numeric.shape[-1]
    )
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
        model, training = train_model(
            model,
            train_mask,
            validation_mask,
            inputs,
            gate_gain,
            torch.from_numpy(weights_np),
            args.seed,
            log_handle,
        )
    validation_inputs = {
        "views": modelval_features,
        "numeric": torch.from_numpy(modelval_numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    predictions = predict_model(
        "RCVI", model, validation_inputs, len(modelval_features), len(candidate.relations)
    )
    relation_rows = json.loads(
        (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
    )
    per_relation = [
        {
            "pair_id": relation_rows[relation]["pair_id"],
            **summarize_relation(modelval_gain[:, relation], predictions[:, relation]),
        }
        for relation in range(len(candidate.relations))
    ]
    report = {
        "schema_version": 1,
        "protocol": "v2-D",
        "development_only": True,
        "test_accessed": False,
        "seed": args.seed,
        "candidate_run": candidate_dir.name,
        "training_split": "gate_internal_4_of_5_hash_residues",
        "checkpoint_split": "gate_internal_1_of_5_hash_residues",
        "evaluation_split": "modelval",
        "weight_definition": "1 + I_negative/p_negative + I_weak/p_weak",
        "training_prevalence": prevalence,
        "training": training,
        "aggregate": aggregate(per_relation),
        "per_relation": per_relation,
        "strata": {
            "context_negative": stratum_summary(
                modelval_gain, predictions, modelval_negative
            ),
            "weak_target": stratum_summary(modelval_gain, predictions, modelval_weak),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "numeric_mean": numeric_mean,
            "numeric_scale": numeric_scale,
        },
        output_dir / "balanced_gain_model.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "candidate_checkpoint": candidate_dir / "candidates.pt",
        "gate_cache_index": REPO_ROOT / "data/features/gate/index.json",
        "modelval_cache_index": REPO_ROOT / "data/features/modelval/index.json",
        "gate_groups": REPO_ROOT / "data/groups/natural_gate.npz",
        "modelval_groups": REPO_ROOT / "data/groups/natural_modelval.npz",
        "preregistration": REPO_ROOT / "docs/v2_preregistration.md",
        "training_code": Path(__file__).resolve(),
    }
    (output_dir / "input_hashes.json").write_text(
        json.dumps(
            {name: sha256(path) for name, path in provenance.items()}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
