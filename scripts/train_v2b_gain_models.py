#!/usr/bin/env python3
"""Train preregistered neural observable-gain models on frozen multiview features."""

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

from audit_observable_opportunity import (
    aggregate,
    candidate_gain,
    feature_matrices,
    summarize_relation,
)
from train_gate import load_candidate

from edcr.config import load_config
from edcr.observability import LogitGainMLP, RelationConditionedViewGain

TARGET_SCALE = 100.0
PAIR_BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 5


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def internal_split(sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    residues = np.asarray(
        [
            int.from_bytes(hashlib.sha256(f"v2b:{sample_id}".encode()).digest()[:8]) % 5
            for sample_id in sample_ids.astype(str)
        ]
    )
    return residues != 0, residues == 0


def standardizer(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = values[train_mask].reshape(-1, values.shape[-1])
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    return mean.astype(np.float32), np.maximum(scale, 1e-5).astype(np.float32)


def train_model(
    kind: str,
    model: nn.Module,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    inputs: dict[str, torch.Tensor],
    gain: torch.Tensor,
    seed: int,
    log_handle,
) -> tuple[nn.Module, dict[str, object]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask))
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask))
    relation_count = gain.shape[1]
    image_batch_size = max(1, PAIR_BATCH_SIZE // relation_count)
    best_state: dict[str, torch.Tensor] | None = None
    best_mse = float("inf")
    best_epoch = -1
    stale = 0

    def predict(indices: torch.Tensor) -> torch.Tensor:
        if kind == "LogitMLP":
            return model(inputs["logits"][indices], inputs["relation_indices"])
        return model(
            inputs["views"][indices],
            inputs["numeric"][indices],
            inputs["source_queries"],
            inputs["target_queries"],
        )

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for offset in range(0, len(order), image_batch_size):
            indices = order[offset : offset + image_batch_size]
            predictions = predict(indices)
            loss = functional.mse_loss(predictions, gain[indices] * TARGET_SCALE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_predictions = torch.cat(
                [
                    predict(validation_indices[offset : offset + image_batch_size])
                    for offset in range(0, len(validation_indices), image_batch_size)
                ]
            )
            validation_mse = float(
                functional.mse_loss(
                    validation_predictions, gain[validation_indices] * TARGET_SCALE
                )
            )
        log_handle.write(
            json.dumps(
                {"model": kind, "epoch": epoch, "internal_validation_mse": validation_mse},
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
        raise RuntimeError(f"{kind} produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "internal_validation_mse": best_mse}


def predict_model(
    kind: str,
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    sample_count: int,
    relation_count: int,
) -> np.ndarray:
    image_batch_size = max(1, PAIR_BATCH_SIZE // relation_count)
    relation_indices = torch.arange(relation_count)
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, sample_count, image_batch_size):
            selected = slice(offset, min(offset + image_batch_size, sample_count))
            if kind == "LogitMLP":
                output = model(inputs["logits"][selected], relation_indices)
            else:
                output = model(
                    inputs["views"][selected],
                    inputs["numeric"][selected],
                    inputs["source_queries"],
                    inputs["target_queries"],
                )
            chunks.append(output / TARGET_SCALE)
    return torch.cat(chunks).numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-run")
    parser.add_argument("--run-id")
    parser.add_argument("--reference-run")
    parser.add_argument(
        "--relation-manifest", default="data/manifests/selected_relations.json"
    )
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")
    candidate_run = args.candidate_run or f"p2e_seed{args.seed}"
    run_id = args.run_id or f"v2b_gain_seed{args.seed}"
    output_dir = REPO_ROOT / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidate_dir = REPO_ROOT / "runs" / candidate_run
    candidate = load_candidate(candidate_dir, config)

    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    modelval_ids, modelval_features_np, modelval_labels_np = load_feature_shards_with_ids("modelval")
    gate_features = torch.from_numpy(gate_features_np)
    modelval_features = torch.from_numpy(modelval_features_np)
    gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
    modelval_tensors = build_tensors(candidate, modelval_features, modelval_labels_np)
    del modelval_ids
    gate_gain = torch.from_numpy(candidate_gain(gate_tensors))
    modelval_gain = candidate_gain(modelval_tensors)
    gate_feature_sets = feature_matrices(candidate, gate_features, gate_tensors)
    modelval_feature_sets = feature_matrices(candidate, modelval_features, modelval_tensors)
    train_mask, internal_validation_mask = internal_split(gate_ids)
    logit_mean, logit_scale = standardizer(gate_feature_sets["view_logits"], train_mask)
    numeric_mean, numeric_scale = standardizer(gate_tensors.numeric.numpy(), train_mask)
    gate_logits = (gate_feature_sets["view_logits"] - logit_mean) / logit_scale
    modelval_logits = (modelval_feature_sets["view_logits"] - logit_mean) / logit_scale
    gate_numeric = (gate_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    modelval_numeric = (modelval_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    relation_count = len(candidate.relations)
    train_inputs = {
        "logits": torch.from_numpy(gate_logits).float(),
        "views": gate_features,
        "numeric": torch.from_numpy(gate_numeric).float(),
        "relation_indices": torch.arange(relation_count),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    validation_inputs = {
        "logits": torch.from_numpy(modelval_logits).float(),
        "views": modelval_features,
        "numeric": torch.from_numpy(modelval_numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    torch.manual_seed(args.seed)
    models: dict[str, nn.Module] = {
        "LogitMLP": LogitGainMLP(18, relation_count),
        "RCVI": RelationConditionedViewGain(
            gate_features.shape[-1], gate_features.shape[1], 6
        ),
    }
    checkpoints: dict[str, object] = {}
    results: dict[str, object] = {}
    relation_rows = json.loads(
        (REPO_ROOT / args.relation_manifest).read_text(encoding="utf-8")
    )
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
        for kind, model in models.items():
            model, training = train_model(
                kind,
                model,
                train_mask,
                internal_validation_mask,
                train_inputs,
                gate_gain,
                args.seed,
                log_handle,
            )
            predictions = predict_model(
                kind, model, validation_inputs, len(modelval_features), relation_count
            )
            per_relation = [
                {
                    "pair_id": relation_rows[relation]["pair_id"],
                    **summarize_relation(modelval_gain[:, relation], predictions[:, relation]),
                }
                for relation in range(relation_count)
            ]
            results[kind] = {"training": training, "aggregate": aggregate(per_relation), "per_relation": per_relation}
            checkpoints[kind] = model.state_dict()
    reference_run = args.reference_run or f"v2a_observable_seed{args.seed}"
    reference_report = json.loads(
        (REPO_ROOT / "runs" / reference_run / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    reference = reference_report["feature_sets"]["view_logits"]["aggregate"]
    rcvi = results["RCVI"]["aggregate"]
    report = {
        "schema_version": 1,
        "protocol": "v2-B",
        "development_only": True,
        "test_accessed": False,
        "training_split": "gate_internal_4_of_5_hash_residues",
        "checkpoint_split": "gate_internal_1_of_5_hash_residues",
        "evaluation_split": "modelval",
        "seed": args.seed,
        "candidate_run": candidate_run,
        "target_scale": TARGET_SCALE,
        "pair_batch_size": PAIR_BATCH_SIZE,
        "models": results,
        "v2a_view_logits_reference": reference,
        "rcvi_beyond_constant_gain_over_v2a": (
            rcvi["recovered_beyond_constant_fraction"]
            - reference["recovered_beyond_constant_fraction"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "models": checkpoints,
            "logit_mean": logit_mean,
            "logit_scale": logit_scale,
            "numeric_mean": numeric_mean,
            "numeric_scale": numeric_scale,
        },
        output_dir / "gain_models.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "candidate_checkpoint": candidate_dir / "candidates.pt",
        "gate_cache_index": REPO_ROOT / "data/features/gate/index.json",
        "modelval_cache_index": REPO_ROOT / "data/features/modelval/index.json",
        "relation_manifest": REPO_ROOT / args.relation_manifest,
        "preregistration": REPO_ROOT / "docs/v2_preregistration.md",
        "model_code": REPO_ROOT / "src/edcr/observability.py",
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


def load_feature_shards_with_ids(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from edcr.cache import load_feature_shards

    return load_feature_shards(
        REPO_ROOT / "data/features" / split,
        REPO_ROOT / "data/manifests" / f"{split}.jsonl",
        80,
    )


def build_tensors(
    candidate: nn.Module, features: torch.Tensor, labels: np.ndarray
):
    from edcr.gating import build_gate_tensors

    return build_gate_tensors(candidate, features, torch.from_numpy(labels))


if __name__ == "__main__":
    raise SystemExit(main())
