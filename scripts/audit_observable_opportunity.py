#!/usr/bin/env python3
"""Estimate representation-measurable candidate opportunity on v2 development data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_gate import load_candidate

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.gating import GateTensors, build_gate_tensors

FEATURE_SETS = ("confidence", "summary", "view_logits")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_split(
    split: str, candidate: torch.nn.Module
) -> tuple[torch.Tensor, GateTensors]:
    _ids, features, labels = load_feature_shards(
        REPO_ROOT / "data/features" / split,
        REPO_ROOT / "data/manifests" / f"{split}.jsonl",
        80,
    )
    feature_tensor = torch.from_numpy(features)
    tensors = build_gate_tensors(candidate, feature_tensor, torch.from_numpy(labels))
    return feature_tensor, tensors


def candidate_gain(tensors: GateTensors) -> np.ndarray:
    base_loss = functional.binary_cross_entropy_with_logits(
        tensors.base, tensors.target_labels, reduction="none"
    )
    context_loss = functional.binary_cross_entropy_with_logits(
        tensors.base + tensors.residual, tensors.target_labels, reduction="none"
    )
    return (base_loss - context_loss).numpy()


def feature_matrices(
    candidate: torch.nn.Module, features: torch.Tensor, tensors: GateTensors
) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        view_logits = candidate.base_head.view_logits(features)
        full_logits = candidate.full_head(features[:, 0])
    relation_features: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_SETS}
    for relation_index, relation in enumerate(candidate.relations):
        summary = tensors.numeric[:, relation_index].numpy()
        relation_features["confidence"].append(
            torch.sigmoid(tensors.base[:, relation_index]).numpy()[:, None]
        )
        relation_features["summary"].append(summary)
        relation_features["view_logits"].append(
            np.concatenate(
                (
                    summary,
                    view_logits[:, :, relation.target].numpy(),
                    view_logits[:, :, relation.source].numpy(),
                    full_logits[:, relation.target].numpy()[:, None],
                    full_logits[:, relation.source].numpy()[:, None],
                ),
                axis=1,
            )
        )
    return {name: np.stack(rows, axis=1) for name, rows in relation_features.items()}


def safe_weighted_auc(delta: np.ndarray, predictions: np.ndarray) -> float | None:
    keep = np.abs(delta) >= 1e-4
    targets = delta[keep] > 0
    if len(np.unique(targets)) != 2:
        return None
    return float(
        roc_auc_score(targets, predictions[keep], sample_weight=np.abs(delta[keep]))
    )


def summarize_relation(delta: np.ndarray, predicted_gain: np.ndarray) -> dict[str, float | None]:
    oracle = float(np.maximum(delta, 0).mean())
    constant = max(float(delta.mean()), 0.0)
    achieved = float((delta * (predicted_gain > 0)).mean())
    denominator = oracle - constant
    return {
        "oracle_improvement": oracle,
        "best_constant_improvement": constant,
        "achieved_improvement": achieved,
        "recovered_oracle_fraction": achieved / oracle if oracle > 0 else None,
        "recovered_beyond_constant_fraction": (
            (achieved - constant) / denominator if denominator > 0 else None
        ),
        "weighted_action_auroc": safe_weighted_auc(delta, predicted_gain),
        "gain_regression_mse": float(np.mean((predicted_gain - delta) ** 2)),
        "context_action_fraction": float(np.mean(predicted_gain > 0)),
    }


def aggregate(per_relation: list[dict[str, float | None]]) -> dict[str, float]:
    oracle = float(np.mean([row["oracle_improvement"] for row in per_relation]))
    constant = float(np.mean([row["best_constant_improvement"] for row in per_relation]))
    achieved = float(np.mean([row["achieved_improvement"] for row in per_relation]))
    return {
        "oracle_improvement": oracle,
        "best_constant_improvement": constant,
        "achieved_improvement": achieved,
        "recovered_oracle_fraction": achieved / oracle,
        "recovered_beyond_constant_fraction": (achieved - constant) / (oracle - constant),
        "macro_weighted_action_auroc": float(
            np.mean([row["weighted_action_auroc"] for row in per_relation])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-run")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--relation-manifest", default="data/manifests/selected_relations.json"
    )
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")
    candidate_run = args.candidate_run or f"p2e_seed{args.seed}"
    run_id = args.run_id or f"v2a_observable_seed{args.seed}"
    output_dir = REPO_ROOT / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidate_dir = REPO_ROOT / "runs" / candidate_run
    candidate = load_candidate(candidate_dir, config)
    gate_features, gate_tensors = load_split("gate", candidate)
    modelval_features, modelval_tensors = load_split("modelval", candidate)
    gate_gain = candidate_gain(gate_tensors)
    modelval_gain = candidate_gain(modelval_tensors)
    train_features = feature_matrices(candidate, gate_features, gate_tensors)
    validation_features = feature_matrices(candidate, modelval_features, modelval_tensors)
    relation_rows = json.loads(
        (REPO_ROOT / args.relation_manifest).read_text(encoding="utf-8")
    )
    results: dict[str, object] = {}
    models: dict[str, list[HistGradientBoostingRegressor]] = {}
    for feature_set in FEATURE_SETS:
        per_relation: list[dict[str, object]] = []
        models[feature_set] = []
        for relation, relation_row in enumerate(relation_rows):
            regressor = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=args.seed,
            )
            regressor.fit(train_features[feature_set][:, relation], gate_gain[:, relation])
            predicted_gain = regressor.predict(
                validation_features[feature_set][:, relation]
            )
            models[feature_set].append(regressor)
            per_relation.append(
                {
                    "pair_id": relation_row["pair_id"],
                    **summarize_relation(modelval_gain[:, relation], predicted_gain),
                }
            )
        results[feature_set] = {
            "feature_dimension": int(train_features[feature_set].shape[-1]),
            "aggregate": aggregate(per_relation),
            "per_relation": per_relation,
        }
    summary_gain = results["summary"]["aggregate"]["achieved_improvement"]
    view_gain = results["view_logits"]["aggregate"]["achieved_improvement"]
    report = {
        "schema_version": 1,
        "protocol": "v2-A",
        "development_only": True,
        "test_accessed": False,
        "training_split": "gate",
        "evaluation_split": "modelval",
        "seed": args.seed,
        "candidate_run": candidate_run,
        "regressor": {
            "type": "HistGradientBoostingRegressor",
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
        },
        "feature_sets": results,
        "view_logits_minus_summary_achieved_improvement": view_gain - summary_gain,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "candidate_checkpoint": candidate_dir / "candidates.pt",
        "gate_cache_index": REPO_ROOT / "data/features/gate/index.json",
        "modelval_cache_index": REPO_ROOT / "data/features/modelval/index.json",
        "gate_manifest": REPO_ROOT / "data/manifests/gate.jsonl",
        "modelval_manifest": REPO_ROOT / "data/manifests/modelval.jsonl",
        "relation_manifest": REPO_ROOT / args.relation_manifest,
        "preregistration": REPO_ROOT / "docs/v2_preregistration.md",
        "audit_code": Path(__file__).resolve(),
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
