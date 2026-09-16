#!/usr/bin/env python3
"""Trace the preregistered RCVI multi-risk opportunity frontier."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import candidate_gain
from evaluate_gate_modelval import load_split, method_scores, summarize_method
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
    predict_model,
    standardizer,
)
from train_v2d_balanced_gain import load_groups, train_model

from edcr.calibration import calibrate_classes
from edcr.config import load_config
from edcr.observability import RelationConditionedViewGain

NEGATIVE_COEFFICIENTS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
WEAK_COEFFICIENTS = (0.0, 0.25, 1.0, 4.0)


def risk_weights(
    negative: np.ndarray,
    weak: np.ndarray,
    train_mask: np.ndarray,
    negative_coefficient: float,
    weak_coefficient: float,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    negative_prevalence = negative[train_mask].mean(axis=0)
    weak_prevalence = weak[train_mask].mean(axis=0)
    weights = (
        1.0
        + negative_coefficient * negative / negative_prevalence[None, :]
        + weak_coefficient * weak / weak_prevalence[None, :]
    )
    return weights.astype(np.float32), {
        "negative": negative_prevalence.tolist(),
        "weak": weak_prevalence.tolist(),
    }


def hard_scores(tensors, predictions: np.ndarray) -> np.ndarray:
    return tensors.base.numpy() + (predictions > 0).astype(np.float32) * tensors.residual.numpy()


def point_id(negative_coefficient: float, weak_coefficient: float) -> str:
    def token(value: float) -> str:
        return str(value).replace(".", "p")

    return f"neg{token(negative_coefficient)}_weak{token(weak_coefficient)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")
    output_dir = REPO_ROOT / "runs" / f"v2e_frontier_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    candidate_dir = REPO_ROOT / "runs" / f"p2e_seed{args.seed}"
    p3_dir = REPO_ROOT / "runs" / f"p3e_groupsup_seed{args.seed}"
    candidate = load_candidate(candidate_dir, config)
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    gate_negative, gate_weak = load_groups(
        REPO_ROOT / "data/groups/natural_gate.npz", gate_ids
    )
    gate_features = torch.from_numpy(gate_features_np)
    gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
    gate_gain = torch.from_numpy(candidate_gain(gate_tensors))
    train_mask, validation_mask = internal_split(gate_ids)
    numeric_mean, numeric_scale = standardizer(gate_tensors.numeric.numpy(), train_mask)
    gate_numeric = (gate_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    train_inputs = {
        "views": gate_features,
        "numeric": torch.from_numpy(gate_numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }

    _calib_ids, calib_features, calib_tensors = load_split("calib", candidate)
    modelval_ids, modelval_features, modelval_tensors = load_split("modelval", candidate)
    calib_numeric = (calib_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    modelval_numeric = (modelval_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    calib_inputs = {
        "views": calib_features,
        "numeric": torch.from_numpy(calib_numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    modelval_inputs = {
        "views": modelval_features,
        "numeric": torch.from_numpy(modelval_numeric).float(),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    with np.load(REPO_ROOT / "data/groups/natural_modelval.npz") as groups:
        if not np.array_equal(groups["sample_id"].astype(str), modelval_ids.astype(str)):
            raise ValueError("modelval natural-group sample order differs")
        negative = groups["negative"].astype(bool)
        weak = groups["weak"].astype(bool)
        weak_coco_small = groups["weak_coco_small"].astype(bool)
    labels = modelval_tensors.target_labels.numpy()
    target_recall = float(config["evaluation"]["calibration_recall"])
    min_positives = int(config["evaluation"]["min_calibration_positives"])

    base_calib = calib_tensors.base.numpy()
    base_modelval = modelval_tensors.base.numpy()
    base_thresholds = calibrate_classes(
        calib_tensors.target_labels.numpy(), base_calib, target_recall, min_positives
    )
    baselines = {
        "B1": summarize_method(
            labels, base_modelval, base_thresholds, negative, weak, weak_coco_small
        )
    }
    p3_states = torch.load(p3_dir / "gates.pt", map_location="cpu", weights_only=True)
    hidden = int(config["training"]["gate_hidden"])
    batch_size = int(config["training"]["cached_batch_size"])
    p3_calib, _ = method_scores(
        candidate, calib_features, calib_tensors, p3_states, hidden, batch_size
    )
    p3_modelval, _ = method_scores(
        candidate, modelval_features, modelval_tensors, p3_states, hidden, batch_size
    )
    for method in ("B5", "M2"):
        thresholds = calibrate_classes(
            calib_tensors.target_labels.numpy(),
            p3_calib[method],
            target_recall,
            min_positives,
        )
        baselines[method] = summarize_method(
            labels, p3_modelval[method], thresholds, negative, weak, weak_coco_small
        )

    points: list[dict[str, object]] = []
    checkpoints: dict[str, object] = {}
    prevalence: dict[str, list[float]] | None = None
    for negative_coefficient in NEGATIVE_COEFFICIENTS:
        for weak_coefficient in WEAK_COEFFICIENTS:
            weights_np, prevalence = risk_weights(
                gate_negative,
                gate_weak,
                train_mask,
                negative_coefficient,
                weak_coefficient,
            )
            torch.manual_seed(args.seed)
            model = RelationConditionedViewGain(
                gate_features.shape[-1],
                gate_features.shape[1],
                gate_tensors.numeric.shape[-1],
            )
            identifier = point_id(negative_coefficient, weak_coefficient)
            with (output_dir / f"train_{identifier}.jsonl").open(
                "w", encoding="utf-8"
            ) as log_handle:
                model, training = train_model(
                    model,
                    train_mask,
                    validation_mask,
                    train_inputs,
                    gate_gain,
                    torch.from_numpy(weights_np),
                    args.seed,
                    log_handle,
                )
            calib_prediction = predict_model(
                "RCVI", model, calib_inputs, len(calib_features), len(candidate.relations)
            )
            modelval_prediction = predict_model(
                "RCVI",
                model,
                modelval_inputs,
                len(modelval_features),
                len(candidate.relations),
            )
            calib_score = hard_scores(calib_tensors, calib_prediction)
            modelval_score = hard_scores(modelval_tensors, modelval_prediction)
            thresholds = calibrate_classes(
                calib_tensors.target_labels.numpy(),
                calib_score,
                target_recall,
                min_positives,
            )
            points.append(
                {
                    "point_id": identifier,
                    "lambda_negative": negative_coefficient,
                    "lambda_weak": weak_coefficient,
                    "training": training,
                    "context_action_fraction": float(np.mean(modelval_prediction > 0)),
                    "metrics": summarize_method(
                        labels,
                        modelval_score,
                        thresholds,
                        negative,
                        weak,
                        weak_coco_small,
                    ),
                }
            )
            checkpoints[identifier] = model.state_dict()
    report = {
        "schema_version": 1,
        "protocol": "v2-E",
        "development_only": True,
        "test_accessed": False,
        "seed": args.seed,
        "coefficient_grid": {
            "lambda_negative": list(NEGATIVE_COEFFICIENTS),
            "lambda_weak": list(WEAK_COEFFICIENTS),
        },
        "training_prevalence": prevalence,
        "baselines": baselines,
        "points": points,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "models": checkpoints,
            "numeric_mean": numeric_mean,
            "numeric_scale": numeric_scale,
        },
        output_dir / "frontier_models.pt",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
