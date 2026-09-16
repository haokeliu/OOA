#!/usr/bin/env python3
"""Evaluate a directly optimized candidate-constrained soft gate on V8 folds."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit

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
from run_v8_calibration_baselines import fit_calibrator, gain_regressor
from train_gate import load_candidate

from edcr.cache import load_feature_shards
from edcr.config import load_config

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
MAX_ITERATIONS = 200
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260914
PLAN_PATH = REPO_ROOT / "docs/v9_independent_candidates_and_direct_soft_plan.md"
OUTPUT_PATH = REPO_ROOT / "reports/v9a_direct_soft_gate.json"


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def standardize(
    training_features: np.ndarray, evaluation_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (training_features - mean) / scale, (evaluation_features - mean) / scale


def direct_soft_objective(
    parameters: np.ndarray,
    features: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    weights = parameters[:-1]
    bias = parameters[-1]
    linear = features @ weights + bias
    action = expit(linear)
    selected = base + action * residual
    loss = np.logaddexp(0.0, selected) - labels * selected
    value = float(loss.mean() + l2 * np.dot(weights, weights))
    selected_probability = expit(selected)
    chain = (selected_probability - labels) * residual * action * (1.0 - action)
    gradient = np.empty_like(parameters)
    gradient[:-1] = features.T @ chain / len(labels) + 2.0 * l2 * weights
    gradient[-1] = chain.mean()
    return value, gradient


def fit_direct_soft_gate(
    features: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> np.ndarray:
    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    result = minimize(
        direct_soft_objective,
        initial,
        args=(features, base, residual, labels, l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_ITERATIONS, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success and result.status != 1:
        raise RuntimeError(f"direct soft optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def predict_direct_soft(parameters: np.ndarray, features: np.ndarray) -> np.ndarray:
    return expit(features @ parameters[:-1] + parameters[-1])


def mean_bce(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def cluster_summary(rows: list[dict[str, object]], key: str, generator) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(float(row[key]))
    blocks = [np.asarray(grouped[name], dtype=np.float64) for name in sorted(grouped)]
    values = np.concatenate(blocks)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = generator.integers(0, len(blocks), size=len(blocks))
        replicates[replicate] = np.concatenate([blocks[index] for index in selected]).mean()
    return {
        "mean": float(values.mean()),
        "relation_cluster_95_percentile_interval": np.quantile(
            replicates, (0.025, 0.975)
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
        "task_count": len(values),
        "relation_count": len(blocks),
    }


def main() -> int:
    started = time.perf_counter()
    panel = json.loads(PANEL_PATH.read_text())
    labels_by_image_mid = read_labels({row["target_openimages_mid"] for row in panel})
    sample_ids, features_np, sealed_labels = load_feature_shards(CACHE_PATH, MANIFEST_PATH, 80)
    if np.any(sealed_labels):
        raise RuntimeError("feature manifest unexpectedly exposes labels")
    image_ids = np.asarray(
        [sample_id.removeprefix("oi_v7_val_") for sample_id in sample_ids.astype(str)]
    )
    frozen_features = torch.from_numpy(features_np)
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        candidate = load_candidate(REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config)
        with torch.inference_mode():
            global_logits = candidate.full_head(frozen_features[:, 0]).numpy().astype(np.float64)
            crop_view_logits = (
                candidate.base_head.view_logits(
                    frozen_features[:, candidate.base_view_indices]
                )
                .numpy()
                .astype(np.float64)
            )
            crop_logits = (
                candidate.base_head(frozen_features[:, candidate.base_view_indices])
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
            matrices = feature_sets(
                global_logits[known],
                crop_logits[known],
                crop_view_logits[known],
                relation.source,
                relation.target,
            )
            base = global_logits[known, relation.target]
            residual = crop_logits[known, relation.target] - base
            endpoint_gain = bce_gain(base, residual, labels, 1.0)
            endpoint_oracle = np.maximum(endpoint_gain, 0.0)

            for representation in REPRESENTATIONS:
                matrix = matrices[representation]
                direct_soft_q = np.empty(len(labels), dtype=np.float64)
                direct_hard_q = np.empty(len(labels), dtype=np.float64)
                analytic_soft_q = np.empty(len(labels), dtype=np.float64)
                probability_hard_q = np.empty(len(labels), dtype=np.float64)
                static_q = np.empty(len(labels), dtype=np.float64)
                constant_q = np.empty(len(labels), dtype=np.float64)
                selected_l2: list[float] = []
                for outer_fold in range(FOLD_COUNT):
                    calibration_fold = (outer_fold + 1) % FOLD_COUNT
                    evaluate = folds == outer_fold
                    calibration = folds == calibration_fold
                    train = ~(evaluate | calibration)
                    train_x, calibration_x = standardize(matrix[train], matrix[calibration])
                    _, evaluation_x = standardize(matrix[train], matrix[evaluate])

                    candidates = []
                    for l2 in L2_GRID:
                        parameters = fit_direct_soft_gate(
                            train_x,
                            base[train],
                            residual[train],
                            labels[train],
                            l2,
                        )
                        calibration_q = predict_direct_soft(parameters, calibration_x)
                        calibration_loss = mean_bce(
                            base[calibration] + calibration_q * residual[calibration],
                            labels[calibration],
                        )
                        candidates.append((calibration_loss, l2, parameters))
                    _, chosen_l2, chosen_parameters = min(candidates, key=lambda item: item[0])
                    selected_l2.append(chosen_l2)
                    direct_soft_q[evaluate] = predict_direct_soft(
                        chosen_parameters, evaluation_x
                    )

                    gain_model = gain_regressor(seed)
                    gain_model.fit(matrix[train], endpoint_gain[train])
                    direct_hard_q[evaluate] = (
                        gain_model.predict(matrix[evaluate]) > 0
                    ).astype(np.float64)

                    probability_model = classifier(seed)
                    probability_model.fit(matrix[train], labels[train])
                    raw_calibration = probability_model.predict_proba(matrix[calibration])[:, 1]
                    raw_evaluation = probability_model.predict_proba(matrix[evaluate])[:, 1]
                    platt = fit_calibrator("platt", raw_calibration, labels[calibration])
                    probability = platt(raw_evaluation)
                    probability_hard_q[evaluate] = (
                        expected_endpoint_gain(
                            probability, base[evaluate], residual[evaluate]
                        )
                        > 0
                    ).astype(np.float64)
                    analytic_soft_q[evaluate] = soft_action(
                        probability, base[evaluate], residual[evaluate]
                    )

                    q_grid = np.linspace(0.0, 1.0, 21)
                    grid_gain = np.asarray(
                        [
                            bce_gain(
                                base[calibration],
                                residual[calibration],
                                labels[calibration],
                                q,
                            ).mean()
                            for q in q_grid
                        ]
                    )
                    static_q[evaluate] = q_grid[int(np.argmax(grid_gain))]
                    constant_q[evaluate] = float(endpoint_gain[calibration].mean() > 0)

                gains = {
                    "selected_constant": bce_gain(base, residual, labels, constant_q),
                    "selected_static": bce_gain(base, residual, labels, static_q),
                    "direct_gain_hard": bce_gain(base, residual, labels, direct_hard_q),
                    "probability_hard_platt": bce_gain(
                        base, residual, labels, probability_hard_q
                    ),
                    "analytic_soft_platt": bce_gain(
                        base, residual, labels, analytic_soft_q
                    ),
                    "direct_soft": bce_gain(base, residual, labels, direct_soft_q),
                    "endpoint_oracle": endpoint_oracle,
                }
                row: dict[str, object] = {
                    "seed": seed,
                    "pair_id": pair_id,
                    "representation": representation,
                    "evaluation_count": len(labels),
                    "selected_l2_by_outer_fold": selected_l2,
                    "direct_soft_interior_fraction": float(
                        np.mean((direct_soft_q > 0) & (direct_soft_q < 1))
                    ),
                }
                row.update({f"{name}_gain": float(value.mean()) for name, value in gains.items()})
                row.update(
                    {
                        "direct_soft_minus_direct_gain_hard": float(
                            gains["direct_soft"].mean() - gains["direct_gain_hard"].mean()
                        ),
                        "direct_soft_minus_probability_hard_platt": float(
                            gains["direct_soft"].mean()
                            - gains["probability_hard_platt"].mean()
                        ),
                        "direct_soft_minus_analytic_soft_platt": float(
                            gains["direct_soft"].mean() - gains["analytic_soft_platt"].mean()
                        ),
                        "direct_soft_minus_selected_static": float(
                            gains["direct_soft"].mean() - gains["selected_static"].mean()
                        ),
                    }
                )
                rows.append(row)

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    metric_keys = (
        "selected_constant_gain",
        "selected_static_gain",
        "direct_gain_hard_gain",
        "probability_hard_platt_gain",
        "analytic_soft_platt_gain",
        "direct_soft_gain",
        "endpoint_oracle_gain",
        "direct_soft_minus_direct_gain_hard",
        "direct_soft_minus_probability_hard_platt",
        "direct_soft_minus_analytic_soft_platt",
        "direct_soft_minus_selected_static",
    )
    report = {
        "schema_version": 1,
        "protocol": "v9a-direct-candidate-constrained-soft-gate",
        "development_only": True,
        "official_test_accessed": False,
        "relation_count": len(panel),
        "task_count": len(rows),
        "l2_grid": list(L2_GRID),
        "maximum_iterations": MAX_ITERATIONS,
        "summary": {key: cluster_summary(rows, key, generator) for key in metric_keys},
        "by_representation": {
            representation: {
                key: cluster_summary(
                    [row for row in rows if row["representation"] == representation],
                    key,
                    generator,
                )
                for key in metric_keys
            }
            for representation in REPRESENTATIONS
        },
        "rows": rows,
        "input_sha256": {
            "plan": sha256(PLAN_PATH),
            "script": sha256(Path(__file__)),
            "annotations": sha256(ANNOTATIONS),
            "manifest": sha256(MANIFEST_PATH),
            "cache_index": sha256(CACHE_PATH / "index.json"),
            "panel": sha256(PANEL_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": report["summary"], "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
