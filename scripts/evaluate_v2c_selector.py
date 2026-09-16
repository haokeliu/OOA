#!/usr/bin/env python3
"""Evaluate frozen V2-B gain predictors as hard downstream selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import feature_matrices
from evaluate_gate_modelval import load_split, method_scores, summarize_method
from train_gate import load_candidate
from train_v2b_gain_models import predict_model

from edcr.calibration import calibrate_classes
from edcr.config import load_config
from edcr.observability import LogitGainMLP, RelationConditionedViewGain


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def gain_predictions(
    checkpoint_path: Path,
    candidate,
    features: torch.Tensor,
    tensors,
) -> dict[str, np.ndarray]:
    # This local checkpoint also contains NumPy standardization arrays written by V2-B.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    relation_count = len(candidate.relations)
    feature_sets = feature_matrices(candidate, features, tensors)
    logits = (feature_sets["view_logits"] - checkpoint["logit_mean"]) / checkpoint[
        "logit_scale"
    ]
    numeric = (tensors.numeric.numpy() - checkpoint["numeric_mean"]) / checkpoint[
        "numeric_scale"
    ]
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    inputs = {
        "logits": torch.from_numpy(logits).float(),
        "views": features,
        "numeric": torch.from_numpy(numeric).float(),
        "relation_indices": torch.arange(relation_count),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    models = {
        "LogitMLP-H": LogitGainMLP(18, relation_count),
        "RCVI-H": RelationConditionedViewGain(
            features.shape[-1], features.shape[1], tensors.numeric.shape[-1]
        ),
    }
    state_names = {"LogitMLP-H": "LogitMLP", "RCVI-H": "RCVI"}
    output: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][state_names[name]])
        output[name] = predict_model(
            state_names[name], model, inputs, len(features), relation_count
        )
    return output


def selected_scores(tensors, predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    base = tensors.base.numpy()
    residual = tensors.residual.numpy()
    return {
        method: base + (gain > 0).astype(np.float32) * residual
        for method, gain in predictions.items()
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

    candidate_dir = REPO_ROOT / "runs" / f"p2e_seed{args.seed}"
    gain_dir = REPO_ROOT / "runs" / f"v2b_gain_seed{args.seed}"
    p3_dir = REPO_ROOT / "runs" / f"p3e_groupsup_seed{args.seed}"
    output_dir = REPO_ROOT / "runs" / (args.run_id or f"v2c_selector_seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=False)

    candidate = load_candidate(candidate_dir, config)
    _calib_ids, calib_features, calib_tensors = load_split("calib", candidate)
    modelval_ids, modelval_features, modelval_tensors = load_split("modelval", candidate)
    gain_checkpoint = gain_dir / "gain_models.pt"
    calib_gain = gain_predictions(gain_checkpoint, candidate, calib_features, calib_tensors)
    modelval_gain = gain_predictions(
        gain_checkpoint, candidate, modelval_features, modelval_tensors
    )
    calib_scores = {
        "B1": calib_tensors.base.numpy(),
        "B2": (calib_tensors.base + calib_tensors.residual).numpy(),
        **selected_scores(calib_tensors, calib_gain),
    }
    modelval_scores = {
        "B1": modelval_tensors.base.numpy(),
        "B2": (modelval_tensors.base + modelval_tensors.residual).numpy(),
        **selected_scores(modelval_tensors, modelval_gain),
    }

    p3_states = torch.load(p3_dir / "gates.pt", map_location="cpu", weights_only=True)
    hidden = int(config["training"]["gate_hidden"])
    batch_size = int(config["training"]["cached_batch_size"])
    old_calib, _ = method_scores(
        candidate, calib_features, calib_tensors, p3_states, hidden, batch_size
    )
    old_modelval, _ = method_scores(
        candidate, modelval_features, modelval_tensors, p3_states, hidden, batch_size
    )
    for method in ("B5", "M2"):
        calib_scores[method] = old_calib[method]
        modelval_scores[method] = old_modelval[method]

    target_recall = float(config["evaluation"]["calibration_recall"])
    min_positives = int(config["evaluation"]["min_calibration_positives"])
    thresholds = {
        method: calibrate_classes(
            calib_tensors.target_labels.numpy(), scores, target_recall, min_positives
        )
        for method, scores in calib_scores.items()
    }
    groups_path = REPO_ROOT / "data/groups/natural_modelval.npz"
    with np.load(groups_path) as groups:
        if not np.array_equal(groups["sample_id"].astype(str), modelval_ids.astype(str)):
            raise ValueError("modelval natural-group sample order differs")
        negative = groups["negative"].astype(bool)
        weak = groups["weak"].astype(bool)
        weak_coco_small = groups["weak_coco_small"].astype(bool)
    labels = modelval_tensors.target_labels.numpy()
    metrics = {
        method: summarize_method(
            labels, scores, thresholds[method], negative, weak, weak_coco_small
        )
        for method, scores in modelval_scores.items()
    }
    action_fractions = {
        method: {
            "calib": float(np.mean(prediction > 0)),
            "modelval": float(np.mean(modelval_gain[method] > 0)),
        }
        for method, prediction in calib_gain.items()
    }
    report = {
        "schema_version": 1,
        "protocol": "v2-C",
        "development_only": True,
        "test_accessed": False,
        "seed": args.seed,
        "candidate_run": candidate_dir.name,
        "gain_run": gain_dir.name,
        "v1_baseline_run": p3_dir.name,
        "calibration_split": "calib",
        "evaluation_split": "modelval",
        "target_recall": target_recall,
        "selection_rule": "context iff predicted conditional BCE gain > 0",
        "methods": metrics,
        "thresholds": thresholds,
        "context_action_fractions": action_fractions,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "selector_outputs.npz",
        labels=labels,
        negative=negative,
        weak=weak,
        weak_coco_small=weak_coco_small,
        **{f"scores_{method}": scores for method, scores in modelval_scores.items()},
        **{
            f"thresholds_{method}": np.asarray(thresholds[method], dtype=np.float32)
            for method in modelval_scores
        },
    )
    provenance = {
        "candidate_checkpoint": candidate_dir / "candidates.pt",
        "gain_checkpoint": gain_checkpoint,
        "v1_gate_checkpoint": p3_dir / "gates.pt",
        "modelval_groups": groups_path,
        "preregistration": REPO_ROOT / "docs/v2_preregistration.md",
        "evaluation_code": Path(__file__).resolve(),
    }
    (output_dir / "input_hashes.json").write_text(
        json.dumps(
            {name: sha256(path) for name, path in provenance.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
