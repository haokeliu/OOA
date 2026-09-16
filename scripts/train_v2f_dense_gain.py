#!/usr/bin/env python3
"""Train the preregistered DenseMapGain observable-opportunity model."""

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
    standardizer,
)

from edcr.config import load_config
from edcr.observability import DenseMapGain


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_dense_maps(split: str, expected_ids: np.ndarray) -> np.ndarray:
    directory = REPO_ROOT / "data/features_dense" / split
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if not index["complete"] or index["sample_count"] != len(expected_ids):
        raise ValueError(f"dense map cache incomplete for {split}")
    maps: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as shard:
            maps.append(shard["maps"].astype(np.float32))
            ids.append(shard["sample_id"].astype(str))
    loaded_ids = np.concatenate(ids)
    if not np.array_equal(loaded_ids, expected_ids.astype(str)):
        raise ValueError(f"dense map sample order differs for {split}")
    output = np.concatenate(maps)
    if output.shape[1:] != (5, 2, 14, 14):
        raise ValueError(f"unexpected dense map shape for {split}: {output.shape}")
    return output


def train_model(
    model: nn.Module,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    maps: torch.Tensor,
    numeric: torch.Tensor,
    gain: torch.Tensor,
    seed: int,
    log_handle,
) -> tuple[nn.Module, dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(seed)
    train_indices = torch.from_numpy(np.flatnonzero(train_mask))
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask))
    relation_indices = torch.arange(gain.shape[1])
    image_batch_size = max(1, PAIR_BATCH_SIZE // gain.shape[1])
    best_state: dict[str, torch.Tensor] | None = None
    best_mse = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        for offset in range(0, len(order), image_batch_size):
            indices = order[offset : offset + image_batch_size]
            prediction = model(maps[indices], numeric[indices], relation_indices)
            loss = functional.mse_loss(prediction, gain[indices] * TARGET_SCALE)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat(
                [
                    model(
                        maps[validation_indices[offset : offset + image_batch_size]],
                        numeric[validation_indices[offset : offset + image_batch_size]],
                        relation_indices,
                    )
                    for offset in range(0, len(validation_indices), image_batch_size)
                ]
            )
            validation_mse = float(
                functional.mse_loss(
                    predictions, gain[validation_indices] * TARGET_SCALE
                )
            )
        log_handle.write(
            json.dumps(
                {"epoch": epoch, "internal_validation_mse": validation_mse},
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
        raise RuntimeError("DenseMapGain produced no checkpoint")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "internal_validation_mse": best_mse}


def predict(
    model: nn.Module, maps: torch.Tensor, numeric: torch.Tensor, relation_count: int
) -> np.ndarray:
    image_batch_size = max(1, PAIR_BATCH_SIZE // relation_count)
    relation_indices = torch.arange(relation_count)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(maps), image_batch_size):
            chunks.append(
                model(
                    maps[offset : offset + image_batch_size],
                    numeric[offset : offset + image_batch_size],
                    relation_indices,
                )
                / TARGET_SCALE
            )
    return torch.cat(chunks).numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")
    output_dir = REPO_ROOT / "runs" / f"v2f_dense_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidate_dir = REPO_ROOT / "runs" / f"p2e_seed{args.seed}"
    candidate = load_candidate(candidate_dir, config)
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    modelval_ids, modelval_features_np, modelval_labels_np = load_feature_shards_with_ids(
        "modelval"
    )
    gate_maps = torch.from_numpy(load_dense_maps("gate", gate_ids))
    modelval_maps = torch.from_numpy(load_dense_maps("modelval", modelval_ids))
    gate_tensors = build_tensors(
        candidate, torch.from_numpy(gate_features_np), gate_labels_np
    )
    modelval_tensors = build_tensors(
        candidate, torch.from_numpy(modelval_features_np), modelval_labels_np
    )
    gate_gain = torch.from_numpy(candidate_gain(gate_tensors))
    modelval_gain = candidate_gain(modelval_tensors)
    train_mask, validation_mask = internal_split(gate_ids)
    numeric_mean, numeric_scale = standardizer(gate_tensors.numeric.numpy(), train_mask)
    gate_numeric = torch.from_numpy(
        (gate_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    ).float()
    modelval_numeric = torch.from_numpy(
        (modelval_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    ).float()
    relation_count = len(candidate.relations)
    torch.manual_seed(args.seed)
    model = DenseMapGain(relation_count, gate_tensors.numeric.shape[-1])
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
        model, training = train_model(
            model,
            train_mask,
            validation_mask,
            gate_maps,
            gate_numeric,
            gate_gain,
            args.seed,
            log_handle,
        )
    predictions = predict(model, modelval_maps, modelval_numeric, relation_count)
    relation_rows = json.loads(
        (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
    )
    per_relation = [
        {
            "pair_id": relation_rows[relation]["pair_id"],
            **summarize_relation(modelval_gain[:, relation], predictions[:, relation]),
        }
        for relation in range(relation_count)
    ]
    rcvi_reference = json.loads(
        (REPO_ROOT / f"runs/v2b_gain_seed{args.seed}/metrics.json").read_text(
            encoding="utf-8"
        )
    )["models"]["RCVI"]["aggregate"]
    dense_aggregate = aggregate(per_relation)
    report = {
        "schema_version": 1,
        "protocol": "v2-F",
        "development_only": True,
        "test_accessed": False,
        "seed": args.seed,
        "candidate_run": candidate_dir.name,
        "training_split": "gate_internal_4_of_5_hash_residues",
        "checkpoint_split": "gate_internal_1_of_5_hash_residues",
        "evaluation_split": "modelval",
        "training": training,
        "aggregate": dense_aggregate,
        "per_relation": per_relation,
        "rcvi_reference": rcvi_reference,
        "beyond_constant_gain_over_rcvi": (
            dense_aggregate["recovered_beyond_constant_fraction"]
            - rcvi_reference["recovered_beyond_constant_fraction"]
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "model": model.state_dict(),
            "numeric_mean": numeric_mean,
            "numeric_scale": numeric_scale,
        },
        output_dir / "dense_gain_model.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "candidate_checkpoint": candidate_dir / "candidates.pt",
        "gate_dense_index": REPO_ROOT / "data/features_dense/gate/index.json",
        "modelval_dense_index": REPO_ROOT / "data/features_dense/modelval/index.json",
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


if __name__ == "__main__":
    raise SystemExit(main())
