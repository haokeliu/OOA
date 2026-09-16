#!/usr/bin/env python3
"""Evaluate frozen DenseMapGain hard selectors on downstream development metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_gate_modelval import load_split, method_scores, summarize_method
from evaluate_v2c_selector import gain_predictions, selected_scores
from train_gate import load_candidate
from train_v2f_dense_gain import load_dense_maps, predict

from edcr.calibration import calibrate_classes
from edcr.config import load_config
from edcr.observability import DenseMapGain


def dense_predictions(checkpoint_path: Path, maps: np.ndarray, tensors) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    numeric = torch.from_numpy(
        (tensors.numeric.numpy() - checkpoint["numeric_mean"])
        / checkpoint["numeric_scale"]
    ).float()
    model = DenseMapGain(maps.shape[1], tensors.numeric.shape[-1])
    model.load_state_dict(checkpoint["model"])
    return predict(model, torch.from_numpy(maps), numeric, maps.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    candidate = load_candidate(REPO_ROOT / "runs" / f"p2e_seed{args.seed}", config)
    calib_ids, calib_features, calib_tensors = load_split("calib", candidate)
    modelval_ids, modelval_features, modelval_tensors = load_split("modelval", candidate)
    calib_maps = load_dense_maps("calib", calib_ids)
    modelval_maps = load_dense_maps("modelval", modelval_ids)
    dense_dir = REPO_ROOT / "runs" / f"v2f_dense_seed{args.seed}"
    dense_checkpoint = dense_dir / "dense_gain_model.pt"
    calib_dense = dense_predictions(dense_checkpoint, calib_maps, calib_tensors)
    modelval_dense = dense_predictions(dense_checkpoint, modelval_maps, modelval_tensors)
    calib_rcvi = gain_predictions(
        REPO_ROOT / "runs" / f"v2b_gain_seed{args.seed}/gain_models.pt",
        candidate,
        calib_features,
        calib_tensors,
    )["RCVI-H"]
    modelval_rcvi = gain_predictions(
        REPO_ROOT / "runs" / f"v2b_gain_seed{args.seed}/gain_models.pt",
        candidate,
        modelval_features,
        modelval_tensors,
    )["RCVI-H"]
    calib_scores = {
        "B1": calib_tensors.base.numpy(),
        "RCVI-H": selected_scores(calib_tensors, {"RCVI-H": calib_rcvi})["RCVI-H"],
        "DenseMapGain-H": selected_scores(
            calib_tensors, {"DenseMapGain-H": calib_dense}
        )["DenseMapGain-H"],
    }
    modelval_scores = {
        "B1": modelval_tensors.base.numpy(),
        "RCVI-H": selected_scores(modelval_tensors, {"RCVI-H": modelval_rcvi})[
            "RCVI-H"
        ],
        "DenseMapGain-H": selected_scores(
            modelval_tensors, {"DenseMapGain-H": modelval_dense}
        )["DenseMapGain-H"],
    }
    p3_dir = REPO_ROOT / "runs" / f"p3e_groupsup_seed{args.seed}"
    p3_states = torch.load(p3_dir / "gates.pt", map_location="cpu", weights_only=True)
    old_calib, _ = method_scores(
        candidate,
        calib_features,
        calib_tensors,
        p3_states,
        int(config["training"]["gate_hidden"]),
        int(config["training"]["cached_batch_size"]),
    )
    old_modelval, _ = method_scores(
        candidate,
        modelval_features,
        modelval_tensors,
        p3_states,
        int(config["training"]["gate_hidden"]),
        int(config["training"]["cached_batch_size"]),
    )
    for method in ("B5", "M2"):
        calib_scores[method] = old_calib[method]
        modelval_scores[method] = old_modelval[method]
    target_recall = float(config["evaluation"]["calibration_recall"])
    min_positives = int(config["evaluation"]["min_calibration_positives"])
    thresholds = {
        method: calibrate_classes(
            calib_tensors.target_labels.numpy(), score, target_recall, min_positives
        )
        for method, score in calib_scores.items()
    }
    with np.load(REPO_ROOT / "data/groups/natural_modelval.npz") as groups:
        if not np.array_equal(groups["sample_id"].astype(str), modelval_ids.astype(str)):
            raise ValueError("modelval natural-group sample order differs")
        negative = groups["negative"].astype(bool)
        weak = groups["weak"].astype(bool)
        weak_coco_small = groups["weak_coco_small"].astype(bool)
    labels = modelval_tensors.target_labels.numpy()
    report = {
        "schema_version": 1,
        "protocol": "v2-F-downstream",
        "development_only": True,
        "test_accessed": False,
        "seed": args.seed,
        "calibration_split": "calib",
        "evaluation_split": "modelval",
        "target_recall": target_recall,
        "methods": {
            method: summarize_method(
                labels, score, thresholds[method], negative, weak, weak_coco_small
            )
            for method, score in modelval_scores.items()
        },
        "thresholds": thresholds,
        "context_action_fractions": {
            "RCVI-H": float(np.mean(modelval_rcvi > 0)),
            "DenseMapGain-H": float(np.mean(modelval_dense > 0)),
        },
    }
    (dense_dir / "downstream_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
