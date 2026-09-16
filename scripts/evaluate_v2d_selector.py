#!/usr/bin/env python3
"""Evaluate the frozen balanced risk-aligned selector on downstream metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_gate_modelval import load_split, method_scores, summarize_method
from evaluate_v2c_selector import gain_predictions, selected_scores
from train_gate import load_candidate
from train_v2b_gain_models import predict_model

from edcr.calibration import calibrate_classes
from edcr.config import load_config
from edcr.observability import RelationConditionedViewGain


def balanced_gain_predictions(
    checkpoint_path: Path,
    candidate,
    features: torch.Tensor,
    tensors,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    model = RelationConditionedViewGain(
        features.shape[-1], features.shape[1], tensors.numeric.shape[-1]
    )
    model.load_state_dict(checkpoint["model"])
    inputs = {
        "views": features,
        "numeric": torch.from_numpy(numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    return predict_model(
        "RCVI", model, inputs, len(features), len(candidate.relations)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")
    candidate_dir = REPO_ROOT / "runs" / f"p2e_seed{args.seed}"
    v2b_dir = REPO_ROOT / "runs" / f"v2b_gain_seed{args.seed}"
    v2d_dir = REPO_ROOT / "runs" / f"v2d_balanced_seed{args.seed}"
    p3_dir = REPO_ROOT / "runs" / f"p3e_groupsup_seed{args.seed}"
    candidate = load_candidate(candidate_dir, config)
    _calib_ids, calib_features, calib_tensors = load_split("calib", candidate)
    modelval_ids, modelval_features, modelval_tensors = load_split("modelval", candidate)

    calib_rcvi = gain_predictions(
        v2b_dir / "gain_models.pt", candidate, calib_features, calib_tensors
    )["RCVI-H"]
    modelval_rcvi = gain_predictions(
        v2b_dir / "gain_models.pt", candidate, modelval_features, modelval_tensors
    )["RCVI-H"]
    calib_balanced = balanced_gain_predictions(
        v2d_dir / "balanced_gain_model.pt", candidate, calib_features, calib_tensors
    )
    modelval_balanced = balanced_gain_predictions(
        v2d_dir / "balanced_gain_model.pt", candidate, modelval_features, modelval_tensors
    )
    calib_scores = {
        "B1": calib_tensors.base.numpy(),
        "RCVI-H": selected_scores(calib_tensors, {"RCVI-H": calib_rcvi})["RCVI-H"],
        "Balanced-RCVI-H": selected_scores(
            calib_tensors, {"Balanced-RCVI-H": calib_balanced}
        )["Balanced-RCVI-H"],
    }
    modelval_scores = {
        "B1": modelval_tensors.base.numpy(),
        "RCVI-H": selected_scores(modelval_tensors, {"RCVI-H": modelval_rcvi})[
            "RCVI-H"
        ],
        "Balanced-RCVI-H": selected_scores(
            modelval_tensors, {"Balanced-RCVI-H": modelval_balanced}
        )["Balanced-RCVI-H"],
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
    with np.load(REPO_ROOT / "data/groups/natural_modelval.npz") as groups:
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
    report = {
        "schema_version": 1,
        "protocol": "v2-D-downstream",
        "development_only": True,
        "test_accessed": False,
        "seed": args.seed,
        "calibration_split": "calib",
        "evaluation_split": "modelval",
        "target_recall": target_recall,
        "methods": metrics,
        "thresholds": thresholds,
        "context_action_fractions": {
            "RCVI-H": {
                "calib": float(np.mean(calib_rcvi > 0)),
                "modelval": float(np.mean(modelval_rcvi > 0)),
            },
            "Balanced-RCVI-H": {
                "calib": float(np.mean(calib_balanced > 0)),
                "modelval": float(np.mean(modelval_balanced > 0)),
            },
        },
    }
    (v2d_dir / "downstream_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
