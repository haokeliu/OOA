#!/usr/bin/env python3
"""Nested calibration and baseline study on Open Images validation only."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_v7_cross_candidate_development import (
    ANNOTATIONS,
    CACHE_PATH,
    FOLD_COUNT,
    MANIFEST_PATH,
    PANEL_PATH,
    REPRESENTATIONS,
    SEEDS,
    bce_gain,
    classifier,
    expected_endpoint_gain,
    feature_sets,
    read_labels,
    soft_action,
    stratified_folds,
)
from train_gate import load_candidate

from edcr.cache import load_feature_shards
from edcr.config import load_config

CALIBRATION_METHODS = ("raw", "temperature", "platt", "isotonic")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260912
PROBABILITY_EPSILON = 1e-6
PLAN_PATH = REPO_ROOT / "docs/v8_calibration_baseline_plan.md"
OUTPUT_PATH = REPO_ROOT / "reports/v8_calibration_baselines.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_probability(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_EPSILON, 1 - PROBABILITY_EPSILON)


def probability_nll(probability: np.ndarray, labels: np.ndarray) -> float:
    probability = clip_probability(probability)
    return float(
        np.mean(-labels * np.log(probability) - (1 - labels) * np.log1p(-probability))
    )


def probability_brier(probability: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probability - labels) ** 2))


def fixed_bin_ece(probability: np.ndarray, labels: np.ndarray, bin_count: int = 10) -> float:
    probability = np.asarray(probability, dtype=np.float64)
    bins = np.minimum((probability * bin_count).astype(np.int64), bin_count - 1)
    result = 0.0
    for bin_index in range(bin_count):
        selected = bins == bin_index
        if selected.any():
            result += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(labels[selected].mean())
            )
    return result


def fit_calibrator(
    method: str,
    calibration_probability: np.ndarray,
    calibration_labels: np.ndarray,
):
    probability = clip_probability(calibration_probability)
    scores = logit(probability)
    if method == "raw":
        return lambda values: clip_probability(values)
    if method == "temperature":
        objective = lambda log_temperature: probability_nll(
            expit(scores / np.exp(log_temperature)), calibration_labels
        )
        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if not result.success:
            raise RuntimeError("temperature optimization failed")
        temperature = float(np.exp(result.x))
        return lambda values: clip_probability(
            expit(logit(clip_probability(values)) / temperature)
        )
    if method == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", random_state=0)
        model.fit(scores[:, None], calibration_labels)
        return lambda values: clip_probability(
            model.predict_proba(logit(clip_probability(values))[:, None])[:, 1]
        )
    if method == "isotonic":
        model = IsotonicRegression(
            y_min=PROBABILITY_EPSILON,
            y_max=1 - PROBABILITY_EPSILON,
            out_of_bounds="clip",
        )
        model.fit(probability, calibration_labels)
        return lambda values: clip_probability(model.predict(clip_probability(values)))
    raise ValueError(f"unknown calibration method: {method}")


def gain_regressor(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=seed,
    )


def probability_gain(
    base: np.ndarray, probability: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    probability = clip_probability(probability)
    base_loss = np.logaddexp(0.0, base) - labels * base
    forecast_loss = -labels * np.log(probability) - (1 - labels) * np.log1p(-probability)
    return base_loss - forecast_loss


def bootstrap_metric(
    rows: list[dict[str, object]], key: str, generator: np.random.Generator
) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(float(row[key]))
    blocks = [np.asarray(grouped[pair_id], dtype=np.float64) for pair_id in sorted(grouped)]
    values = np.concatenate(blocks)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        replicates[replicate] = np.concatenate([blocks[index] for index in chosen]).mean()
    return {
        "task_count": len(values),
        "relation_count": len(blocks),
        "mean": float(values.mean()),
        "relation_cluster_95_percentile_interval": np.quantile(
            replicates, (0.025, 0.975)
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
    }


def summarize_calibration(
    rows: list[dict[str, object]], generator: np.random.Generator
) -> dict[str, object]:
    return {
        "task_count": len(rows),
        "mean_nll": float(np.mean([row["nll"] for row in rows])),
        "mean_brier": float(np.mean([row["brier"] for row in rows])),
        "mean_ece_10_bin": float(np.mean([row["ece_10_bin"] for row in rows])),
        "mean_probability_hard_gain": float(
            np.mean([row["probability_hard_gain"] for row in rows])
        ),
        "mean_analytic_soft_gain": float(
            np.mean([row["analytic_soft_gain"] for row in rows])
        ),
        "mean_soft_interior_fraction": float(
            np.mean([row["soft_interior_fraction"] for row in rows])
        ),
        "soft_minus_probability_hard": bootstrap_metric(
            rows, "soft_minus_probability_hard", generator
        ),
    }


def summarize_baseline(
    rows: list[dict[str, object]], generator: np.random.Generator
) -> dict[str, object]:
    return {
        "scope": rows[0]["scope"],
        "task_count": len(rows),
        "gain": bootstrap_metric(rows, "gain", generator),
        "gain_beyond_nested_selected_constant": bootstrap_metric(
            rows, "gain_beyond_nested_selected_constant", generator
        ),
    }


def baseline_row(
    method: str,
    scope: str,
    seed: int,
    pair_id: str,
    gain: np.ndarray,
    selected_constant_mean: float,
    representation: str | None = None,
) -> dict[str, object]:
    mean_gain = float(np.mean(gain))
    return {
        "method": method,
        "scope": scope,
        "seed": seed,
        "pair_id": pair_id,
        "representation": representation,
        "gain": mean_gain,
        "gain_beyond_nested_selected_constant": mean_gain - selected_constant_mean,
    }


def main() -> int:
    started = time.perf_counter()
    panel = json.loads(PANEL_PATH.read_text())
    labels_by_image_mid = read_labels({row["target_openimages_mid"] for row in panel})
    sample_ids, features_np, sealed_labels = load_feature_shards(
        CACHE_PATH, MANIFEST_PATH, 80
    )
    if np.any(sealed_labels):
        raise RuntimeError("feature manifest unexpectedly exposes labels")
    image_ids = np.asarray(
        [sample_id.removeprefix("oi_v7_val_") for sample_id in sample_ids.astype(str)]
    )
    features = torch.from_numpy(features_np)
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    calibration_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    maximum_soft_oracle_excess = -np.inf

    for seed in SEEDS:
        candidate = load_candidate(REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config)
        if tuple(candidate.base_view_indices) != (1, 2, 3, 4):
            raise RuntimeError("V8 requires the frozen four-corner candidate head")
        with torch.inference_mode():
            global_logits = candidate.full_head(features[:, 0]).numpy().astype(np.float64)
            crop_view_logits = (
                candidate.base_head.view_logits(features[:, candidate.base_view_indices])
                .numpy()
                .astype(np.float64)
            )
            crop_logits = (
                candidate.base_head(features[:, candidate.base_view_indices])
                .numpy()
                .astype(np.float64)
            )
        relation_index = {
            (relation.source, relation.target): relation for relation in candidate.relations
        }
        for relation_row in panel:
            pair_id = relation_row["pair_id"]
            mid = relation_row["target_openimages_mid"]
            known = np.asarray([(image_id, mid) in labels_by_image_mid for image_id in image_ids])
            known_image_ids = image_ids[known]
            labels = np.asarray(
                [labels_by_image_mid[(image_id, mid)] for image_id in known_image_ids],
                dtype=np.float64,
            )
            folds = stratified_folds(known_image_ids, labels, pair_id)
            relation = relation_index[
                (relation_row["source_coco_id"], relation_row["target_coco_id"])
            ]
            base = global_logits[known, relation.target]
            endpoint = crop_logits[known, relation.target]
            residual = endpoint - base
            endpoint_gain = bce_gain(base, residual, labels, 1.0)
            endpoint_oracle = np.maximum(endpoint_gain, 0.0)
            entropy_hard_q = (np.abs(endpoint) > np.abs(base)).astype(np.float64)
            selected_constant_q = np.empty(len(labels), dtype=np.float64)
            selected_static_q = np.empty(len(labels), dtype=np.float64)
            stacked_probability = np.empty(len(labels), dtype=np.float64)
            q_grid = np.linspace(0.0, 1.0, 21)
            endpoint_features = np.column_stack((base, endpoint))
            for outer_fold in range(FOLD_COUNT):
                calibration_fold = (outer_fold + 1) % FOLD_COUNT
                evaluate = folds == outer_fold
                calibration = folds == calibration_fold
                train = ~(evaluate | calibration)
                if len(np.unique(labels[train])) != 2 or len(np.unique(labels[calibration])) != 2:
                    raise RuntimeError(f"nested split lacks both labels for {pair_id}")
                selected_constant_q[evaluate] = float(endpoint_gain[calibration].mean() > 0)
                calibration_grid_gain = np.asarray(
                    [bce_gain(base[calibration], residual[calibration], labels[calibration], q).mean() for q in q_grid]
                )
                selected_static_q[evaluate] = q_grid[int(np.argmax(calibration_grid_gain))]
                stacker = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
                stacker.fit(endpoint_features[train], labels[train])
                stacked_probability[evaluate] = stacker.predict_proba(
                    endpoint_features[evaluate]
                )[:, 1]

            selected_constant_gain = bce_gain(
                base, residual, labels, selected_constant_q
            )
            selected_constant_mean = float(selected_constant_gain.mean())
            candidate_only = {
                "full_image_endpoint": np.zeros(len(labels), dtype=np.float64),
                "four_corner_endpoint": endpoint_gain,
                "nested_selected_constant_endpoint": selected_constant_gain,
                "fixed_midpoint_interpolation": bce_gain(base, residual, labels, 0.5),
                "nested_selected_static_interpolation": bce_gain(
                    base, residual, labels, selected_static_q
                ),
                "endpoint_confidence_hard": bce_gain(
                    base, residual, labels, entropy_hard_q
                ),
                "logistic_endpoint_stacking": probability_gain(
                    base, stacked_probability, labels
                ),
                "revealed_label_endpoint_oracle": endpoint_oracle,
            }
            for method, gains in candidate_only.items():
                if "oracle" in method:
                    scope = "non_deployable_oracle"
                elif method == "logistic_endpoint_stacking":
                    scope = "unconstrained_scope_control"
                else:
                    scope = "candidate_constrained_or_fixed"
                baseline_rows.append(
                    baseline_row(
                        method,
                        scope,
                        seed,
                        pair_id,
                        gains,
                        selected_constant_mean,
                    )
                )

            matrices = feature_sets(
                global_logits[known],
                crop_logits[known],
                crop_view_logits[known],
                relation.source,
                relation.target,
            )
            for representation in REPRESENTATIONS:
                calibrated_probability = {
                    method: np.empty(len(labels), dtype=np.float64)
                    for method in CALIBRATION_METHODS
                }
                predicted_gain = np.empty(len(labels), dtype=np.float64)
                for outer_fold in range(FOLD_COUNT):
                    calibration_fold = (outer_fold + 1) % FOLD_COUNT
                    evaluate = folds == outer_fold
                    calibration = folds == calibration_fold
                    train = ~(evaluate | calibration)
                    probability_model = classifier(seed)
                    probability_model.fit(matrices[representation][train], labels[train])
                    raw_calibration = probability_model.predict_proba(
                        matrices[representation][calibration]
                    )[:, 1]
                    raw_evaluation = probability_model.predict_proba(
                        matrices[representation][evaluate]
                    )[:, 1]
                    for method in CALIBRATION_METHODS:
                        calibrator = fit_calibrator(
                            method, raw_calibration, labels[calibration]
                        )
                        calibrated_probability[method][evaluate] = calibrator(raw_evaluation)
                    gain_model = gain_regressor(seed)
                    gain_model.fit(matrices[representation][train], endpoint_gain[train])
                    predicted_gain[evaluate] = gain_model.predict(
                        matrices[representation][evaluate]
                    )

                direct_gain_q = (predicted_gain > 0).astype(np.float64)
                baseline_rows.append(
                    baseline_row(
                        "direct_gain_hard",
                        "learned_router",
                        seed,
                        pair_id,
                        bce_gain(base, residual, labels, direct_gain_q),
                        selected_constant_mean,
                        representation,
                    )
                )
                for method in CALIBRATION_METHODS:
                    probability = calibrated_probability[method]
                    q_hard = (
                        expected_endpoint_gain(probability, base, residual) > 0
                    ).astype(np.float64)
                    q_soft = soft_action(probability, base, residual)
                    hard_gain = bce_gain(base, residual, labels, q_hard)
                    soft_gain = bce_gain(base, residual, labels, q_soft)
                    soft_oracle_excess = float(np.max(soft_gain - endpoint_oracle))
                    maximum_soft_oracle_excess = max(
                        maximum_soft_oracle_excess, soft_oracle_excess
                    )
                    calibration_rows.append(
                        {
                            "calibration": method,
                            "seed": seed,
                            "representation": representation,
                            "pair_id": pair_id,
                            "evaluation_count": len(labels),
                            "nll": probability_nll(probability, labels),
                            "brier": probability_brier(probability, labels),
                            "ece_10_bin": fixed_bin_ece(probability, labels),
                            "probability_hard_gain": float(hard_gain.mean()),
                            "analytic_soft_gain": float(soft_gain.mean()),
                            "soft_minus_probability_hard": float(
                                soft_gain.mean() - hard_gain.mean()
                            ),
                            "soft_interior_fraction": float(
                                np.mean((q_soft > 0) & (q_soft < 1))
                            ),
                        }
                    )
                    for method_name, gains, scope in (
                        (
                            f"probability_hard_{method}",
                            hard_gain,
                            "candidate_constrained_router",
                        ),
                        (
                            f"analytic_soft_{method}",
                            soft_gain,
                            "candidate_constrained_router",
                        ),
                        (
                            f"unconstrained_probability_forecast_{method}",
                            probability_gain(base, probability, labels),
                            "unconstrained_scope_control",
                        ),
                    ):
                        baseline_rows.append(
                            baseline_row(
                                method_name,
                                scope,
                                seed,
                                pair_id,
                                gains,
                                selected_constant_mean,
                                representation,
                            )
                        )

    if maximum_soft_oracle_excess > 1e-10:
        raise RuntimeError(
            f"candidate-constrained soft action exceeded endpoint oracle: {maximum_soft_oracle_excess}"
        )
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    calibration_summary = {
        method: summarize_calibration(
            [row for row in calibration_rows if row["calibration"] == method],
            generator,
        )
        for method in CALIBRATION_METHODS
    }
    calibration_by_representation = {
        method: {
            representation: summarize_calibration(
                [
                    row
                    for row in calibration_rows
                    if row["calibration"] == method
                    and row["representation"] == representation
                ],
                generator,
            )
            for representation in REPRESENTATIONS
        }
        for method in CALIBRATION_METHODS
    }
    baseline_summary = {
        method: summarize_baseline(
            [row for row in baseline_rows if row["method"] == method], generator
        )
        for method in sorted({str(row["method"]) for row in baseline_rows})
    }
    report = {
        "schema_version": 1,
        "protocol": "v8-nested-calibration-and-baselines",
        "development_only": True,
        "official_test_accessed_by_v8": False,
        "relation_count": len(panel),
        "candidate_seed_count": len(SEEDS),
        "representations": list(REPRESENTATIONS),
        "outer_fold_count": FOLD_COUNT,
        "router_training_fold_count": 3,
        "calibration_fold_count": 1,
        "calibration_methods": list(CALIBRATION_METHODS),
        "calibration_summary": calibration_summary,
        "calibration_by_representation": calibration_by_representation,
        "baseline_summary": baseline_summary,
        "maximum_pointwise_soft_excess_over_endpoint_oracle": maximum_soft_oracle_excess,
        "input_sha256": {
            "annotations": sha256(ANNOTATIONS),
            "cache_index": sha256(CACHE_PATH / "index.json"),
            "manifest": sha256(MANIFEST_PATH),
            "panel": sha256(PANEL_PATH),
            "plan": sha256(PLAN_PATH),
            "v7_implementation": sha256(
                REPO_ROOT / "scripts/run_v7_cross_candidate_development.py"
            ),
            "script": sha256(Path(__file__)),
            **{
                f"candidate_seed_{seed}": sha256(
                    REPO_ROOT / f"runs/v3a_candidates_seed{seed}/candidates.pt"
                )
                for seed in SEEDS
            },
        },
        "calibration_rows": calibration_rows,
        "baseline_rows": baseline_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "protocol": report["protocol"],
                "development_only": True,
                "calibration_summary": calibration_summary,
                "baseline_summary": baseline_summary,
                "maximum_pointwise_soft_excess_over_endpoint_oracle": maximum_soft_oracle_excess,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
