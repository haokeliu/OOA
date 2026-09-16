#!/usr/bin/env python3
"""Calibrate P3 gates on calib and evaluate the frozen operating point on modelval."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.cache import load_feature_shards
from edcr.calibration import calibrate_classes
from edcr.config import load_config
from edcr.gating import (
    ConfidenceGate,
    ConstantGate,
    GateTensors,
    build_gate_tensors,
    predict_gate,
)
from edcr.models import CandidateModel, GateNetwork, MultiViewHead, RelationIndex


def load_candidate(run_dir: Path, config: dict[str, object]) -> CandidateModel:
    checkpoint = torch.load(run_dir / "candidates.pt", map_location="cpu", weights_only=True)
    first_weight = checkpoint["base_head"]["linear.weight"]
    feature_dim, class_count = first_weight.shape[1], first_weight.shape[0]
    base_head = MultiViewHead(feature_dim, class_count, config["features"]["pooling_temperature"])
    full_head = nn.Linear(feature_dim, class_count)
    relations = [RelationIndex(**row) for row in checkpoint["relations"]]
    model = CandidateModel(
        base_head,
        full_head,
        relations,
        config["training"]["residual_bound"],
        tuple(checkpoint["base_view_indices"]),
        torch.tensor(checkpoint["context_centers"]),
        torch.tensor(checkpoint["context_scales"]),
        checkpoint["context_signal"],
    )
    model.load_state_dict(checkpoint["candidate"])
    model.requires_grad_(False)
    model.eval()
    return model


def load_split(
    split: str, candidate: CandidateModel
) -> tuple[np.ndarray, torch.Tensor, GateTensors]:
    sample_ids, features_np, labels_np = load_feature_shards(
        REPO_ROOT / "data" / "features" / split,
        REPO_ROOT / "data" / "manifests" / f"{split}.jsonl",
        80,
    )
    features = torch.from_numpy(features_np)
    tensors = build_gate_tensors(candidate, features, torch.from_numpy(labels_np))
    return sample_ids, features, tensors


def make_gate(method: str, tensors: GateTensors, hidden: int) -> nn.Module:
    if method == "B3":
        return ConstantGate(tensors.numeric.shape[1])
    if method in {"B4", "B4C", "B4G"}:
        return ConfidenceGate(hidden)
    return GateNetwork(tensors.numeric.shape[-1], tensors.source_embeddings.shape[-1], hidden)


def method_scores(
    candidate: CandidateModel,
    features: torch.Tensor,
    tensors: GateTensors,
    gate_states: dict[str, dict[str, torch.Tensor]],
    hidden: int,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target_indices = [relation.target for relation in candidate.relations]
    with torch.inference_mode():
        full = candidate.full_head(features[:, 0])[:, target_indices]
    scores = {
        "B0": full.numpy(),
        "B1": tensors.base.numpy(),
        "B2": (tensors.base + tensors.residual).numpy(),
    }
    gate_values_by_method: dict[str, np.ndarray] = {}
    context = tensors.base + tensors.residual
    scores["O1"] = torch.where(
        tensors.target_labels > 0.5,
        torch.maximum(tensors.base, context),
        torch.minimum(tensors.base, context),
    ).numpy()
    for method, state in gate_states.items():
        gate = make_gate(method, tensors, hidden)
        gate.load_state_dict(state)
        values = predict_gate(method, gate, tensors, batch_size)
        scores[method] = (tensors.base + values * tensors.residual).numpy()
        gate_values_by_method[method] = values.numpy()
    return scores, gate_values_by_method


def gate_diagnostics(
    tensors: GateTensors,
    values: np.ndarray,
    weak: np.ndarray,
    epsilon: float,
) -> dict[str, object]:
    labels = tensors.target_labels.numpy().astype(bool)
    source = tensors.source_labels.numpy().astype(bool)
    base = tensors.base.numpy()
    context = (tensors.base + tensors.residual).numpy()
    labels_float = labels.astype(np.float64)
    difference = (
        np.logaddexp(0.0, base)
        - labels_float * base
        - np.logaddexp(0.0, context)
        + labels_float * context
    )
    gain_target = difference > 0
    keep = np.abs(difference) >= epsilon

    def gain_auc(
        target: np.ndarray, predictions: np.ndarray, weights: np.ndarray
    ) -> float | None:
        return (
            float(roc_auc_score(target, predictions, sample_weight=weights))
            if len(np.unique(target)) == 2
            else None
        )

    per_relation: list[dict[str, float | None]] = []
    for relation in range(values.shape[1]):
        groups: dict[str, float | None] = {}
        for source_value, target_value in ((0, 0), (0, 1), (1, 0), (1, 1)):
            mask = (source[:, relation] == source_value) & (
                labels[:, relation] == target_value
            )
            groups[f"mean_q_n{source_value}{target_value}"] = (
                float(values[mask, relation].mean()) if mask.any() else None
            )
        groups["mean_q_weak"] = (
            float(values[weak[:, relation], relation].mean())
            if weak[:, relation].any()
            else None
        )
        retained = keep[:, relation]
        groups["gain_target_weighted_auroc"] = gain_auc(
            gain_target[retained, relation],
            values[retained, relation],
            np.abs(difference[retained, relation]),
        )
        per_relation.append(groups)
    return {
        "gain_target_weighted_auroc": gain_auc(
            gain_target[keep], values[keep], np.abs(difference[keep])
        ),
        "per_relation": per_relation,
    }


def safe_rate(predictions: np.ndarray, mask: np.ndarray) -> float | None:
    return float(predictions[mask].mean()) if mask.any() else None


def summarize_method(
    labels: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float | None],
    negative: np.ndarray,
    weak: np.ndarray,
    weak_coco_small: np.ndarray,
) -> dict[str, object]:
    if any(value is None for value in thresholds):
        raise ValueError("all selected relations must have a calibrated threshold")
    threshold_array = np.asarray(thresholds, dtype=np.float64)
    predictions = scores >= threshold_array[None, :]
    per_relation: list[dict[str, float | int | None]] = []
    for relation in range(labels.shape[1]):
        positive = labels[:, relation] == 1
        per_relation.append(
            {
                "average_precision": float(
                    average_precision_score(labels[:, relation], scores[:, relation])
                ),
                "cfpr": safe_rate(predictions[:, relation], negative[:, relation]),
                "recall": safe_rate(predictions[:, relation], positive),
                "weak_recall": safe_rate(predictions[:, relation], weak[:, relation]),
                "coco_small_recall": safe_rate(
                    predictions[:, relation], weak_coco_small[:, relation]
                ),
                "negative_support": int(negative[:, relation].sum()),
                "positive_support": int(positive.sum()),
                "weak_support": int(weak[:, relation].sum()),
                "coco_small_support": int(weak_coco_small[:, relation].sum()),
            }
        )

    def macro(key: str) -> float:
        values = [row[key] for row in per_relation if row[key] is not None]
        return float(np.mean(values))

    return {
        "selected_macro_ap": macro("average_precision"),
        "macro_cfpr": macro("cfpr"),
        "macro_recall": macro("recall"),
        "macro_weak_recall": macro("weak_recall"),
        "macro_coco_small_recall": macro("coco_small_recall"),
        "per_relation": per_relation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--p3-run", default="p3_natural_seed17")
    parser.add_argument("--candidate-run")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    output_dir = REPO_ROOT / "runs" / args.p3_run
    training_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    candidate_run = args.candidate_run or training_metrics["candidate_run"]
    candidate = load_candidate(REPO_ROOT / "runs" / candidate_run, config)
    calib_ids, calib_features, calib_tensors = load_split("calib", candidate)
    modelval_ids, modelval_features, modelval_tensors = load_split("modelval", candidate)
    del calib_ids

    gate_states = torch.load(output_dir / "gates.pt", map_location="cpu", weights_only=True)
    hidden = int(config["training"]["gate_hidden"])
    batch_size = int(config["training"]["cached_batch_size"])
    calib_scores, _calib_gates = method_scores(
        candidate, calib_features, calib_tensors, gate_states, hidden, batch_size
    )
    modelval_scores, modelval_gates = method_scores(
        candidate, modelval_features, modelval_tensors, gate_states, hidden, batch_size
    )
    target_recall = float(config["evaluation"]["calibration_recall"])
    min_positives = int(config["evaluation"]["min_calibration_positives"])
    thresholds = {
        method: calibrate_classes(
            calib_tensors.target_labels.numpy(), scores, target_recall, min_positives
        )
        for method, scores in calib_scores.items()
    }
    with np.load(REPO_ROOT / "data" / "groups" / "natural_modelval.npz") as groups:
        if not np.array_equal(groups["sample_id"].astype(str), modelval_ids.astype(str)):
            raise ValueError("modelval natural-group sample order differs")
        negative = groups["negative"].astype(bool)
        weak = groups["weak"].astype(bool)
        weak_coco_small = groups["weak_coco_small"].astype(bool)
    labels = modelval_tensors.target_labels.numpy()
    relation_rows = json.loads(
        (REPO_ROOT / "data" / "manifests" / "selected_relations.json").read_text(
            encoding="utf-8"
        )
    )
    method_metrics = {
        method: summarize_method(
            labels, scores, thresholds[method], negative, weak, weak_coco_small
        )
        for method, scores in modelval_scores.items()
    }
    gate_behavior = {
        method: gate_diagnostics(
            modelval_tensors,
            values,
            weak,
            float(config["training"]["weak_delta_epsilon"]),
        )
        for method, values in modelval_gates.items()
    }
    recall_grid = (0.5, 0.6, 0.7, 0.8, 0.9)
    operating_curves: dict[str, list[dict[str, float]]] = {}
    for method, scores in modelval_scores.items():
        points: list[dict[str, float]] = []
        for calibration_recall in recall_grid:
            curve_thresholds = calibrate_classes(
                calib_tensors.target_labels.numpy(),
                calib_scores[method],
                calibration_recall,
                min_positives,
            )
            curve_metrics = summarize_method(
                labels, scores, curve_thresholds, negative, weak, weak_coco_small
            )
            points.append(
                {
                    "calibration_recall": calibration_recall,
                    "modelval_macro_cfpr": curve_metrics["macro_cfpr"],
                    "modelval_macro_recall": curve_metrics["macro_recall"],
                    "modelval_macro_weak_recall": curve_metrics["macro_weak_recall"],
                }
            )
        operating_curves[method] = points
    primary_method = "M2" if "M2" in method_metrics else "M1"
    primary_ap = method_metrics[primary_method]["selected_macro_ap"]
    p3_ap_pass = bool(
        training_metrics["methods"][primary_method]["feasible"]
        and primary_ap > method_metrics["B5"]["selected_macro_ap"]
        and primary_ap
        > method_metrics["B4G" if primary_method == "M2" else "B4"]["selected_macro_ap"]
    )
    report = {
        "schema_version": 1,
        "p3_run": args.p3_run,
        "candidate_run": candidate_run,
        "calibration_split": "calib",
        "evaluation_split": "modelval",
        "test_accessed": False,
        "target_recall": target_recall,
        "relations": relation_rows,
        "thresholds": thresholds,
        "methods": method_metrics,
        "gate_behavior_diagnostics": gate_behavior,
        "calib_to_modelval_operating_curves": operating_curves,
        "diagnostic_only_methods": ["O1"],
        "p3_ap_pass": p3_ap_pass,
        "primary_method": primary_method,
        "p3_ap_rule": (
            "M2 feasible and selected macro AP strictly exceeds B4G and B5"
            if primary_method == "M2"
            else "M1 feasible and selected macro AP strictly exceeds B4 and B5"
        ),
    }
    report_path = output_dir / "modelval_operating_metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "modelval_operating_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "selected_macro_ap",
                "macro_cfpr",
                "macro_recall",
                "macro_weak_recall",
                "macro_coco_small_recall",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for method, row in method_metrics.items():
            writer.writerow({"method": method, **{key: row[key] for key in writer.fieldnames[1:]}})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
