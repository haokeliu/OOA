#!/usr/bin/env python3
"""Training-only feasibility study for independent CLIP and ResNet candidates."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = REPO_ROOT / "data/v9_cifar100_features/train.npz"
PLAN_PATH = REPO_ROOT / "docs/v9_independent_candidates_and_direct_soft_plan.md"
OUTPUT_PATH = REPO_ROOT / "reports/v9b0_cifar100_feasibility.json"
MODEL_PATH = REPO_ROOT / "runs/v9_cifar100/candidate_heads.npz"
ROUTER_PATH = REPO_ROOT / "runs/v9_cifar100/final_routers.pkl"

RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)
DIRECT_SOFT_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
ROUTER_FOLDS = 5
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260914


def sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def multiclass_nll(logits: np.ndarray, labels: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    return float(np.mean(log_normalizer - shifted[np.arange(len(labels)), labels]))


def per_example_nll(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return np.log(np.exp(shifted).sum(axis=1)) - shifted[
        np.arange(len(labels)), labels
    ]


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    objective = lambda log_temperature: multiclass_nll(
        logits / np.exp(log_temperature), labels
    )
    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    if not result.success:
        raise RuntimeError("multiclass temperature optimization failed")
    return float(np.exp(result.x))


def fit_candidate(
    features: np.ndarray,
    labels: np.ndarray,
    fit_mask: np.ndarray,
    calibration_mask: np.ndarray,
) -> tuple[Ridge, float, dict[str, float]]:
    targets = np.eye(100, dtype=np.float32)[labels[fit_mask]]
    candidates: list[tuple[float, float, float, Ridge]] = []
    for alpha in RIDGE_GRID:
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        model.fit(features[fit_mask], targets)
        calibration_logits = model.predict(features[calibration_mask])
        temperature = fit_temperature(calibration_logits, labels[calibration_mask])
        nll = multiclass_nll(
            calibration_logits / temperature, labels[calibration_mask]
        )
        candidates.append((nll, alpha, temperature, model))
    nll, alpha, temperature, model = min(candidates, key=lambda item: item[0])
    return model, temperature, {
        "selected_alpha": alpha,
        "temperature": temperature,
        "calibration_nll": nll,
    }


def router_features(logits_a: np.ndarray, logits_b: np.ndarray) -> np.ndarray:
    probability_a = softmax(logits_a, axis=1)
    probability_b = softmax(logits_b, axis=1)
    sorted_a = np.partition(probability_a, -2, axis=1)[:, -2:]
    sorted_b = np.partition(probability_b, -2, axis=1)[:, -2:]
    top_a = probability_a.argmax(axis=1)
    top_b = probability_b.argmax(axis=1)
    entropy_a = -(probability_a * np.log(np.clip(probability_a, 1e-12, 1.0))).sum(axis=1)
    entropy_b = -(probability_b * np.log(np.clip(probability_b, 1e-12, 1.0))).sum(axis=1)
    return np.column_stack(
        (
            probability_a.max(axis=1),
            probability_b.max(axis=1),
            sorted_a[:, 1] - sorted_a[:, 0],
            sorted_b[:, 1] - sorted_b[:, 0],
            entropy_a / np.log(100.0),
            entropy_b / np.log(100.0),
            (top_a == top_b).astype(np.float64),
            probability_a[np.arange(len(top_b)), top_b],
            probability_b[np.arange(len(top_a)), top_a],
            np.linalg.norm(logits_a - logits_b, axis=1) / np.sqrt(100.0),
        )
    )


def standardize(
    training_features: np.ndarray, evaluation_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (
        (training_features - mean) / scale,
        (evaluation_features - mean) / scale,
        mean,
        scale,
    )


def direct_soft_objective(
    parameters: np.ndarray,
    features: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    weights = parameters[:-1]
    action = expit(features @ weights + parameters[-1])
    residual = logits_b - logits_a
    selected = logits_a + action[:, None] * residual
    probability = softmax(selected, axis=1)
    loss = per_example_nll(selected, labels)
    value = float(loss.mean() + l2 * np.dot(weights, weights))
    probability[np.arange(len(labels)), labels] -= 1.0
    derivative_q = np.sum(probability * residual, axis=1)
    chain = derivative_q * action * (1.0 - action)
    gradient = np.empty_like(parameters)
    gradient[:-1] = features.T @ chain / len(labels) + 2.0 * l2 * weights
    gradient[-1] = chain.mean()
    return value, gradient


def fit_direct_soft(
    features: np.ndarray,
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    l2: float,
) -> np.ndarray:
    result = minimize(
        direct_soft_objective,
        np.zeros(features.shape[1] + 1, dtype=np.float64),
        args=(features, logits_a, logits_b, labels, l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success and result.status != 1:
        raise RuntimeError(f"multiclass direct-soft optimization failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def predict_direct_soft(parameters: np.ndarray, features: np.ndarray) -> np.ndarray:
    return expit(features @ parameters[:-1] + parameters[-1])


def deterministic_router_folds(indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    folds = np.empty(len(indices), dtype=np.int64)
    for class_index in range(100):
        class_positions = np.flatnonzero(labels == class_index)
        ranked = sorted(
            class_positions,
            key=lambda position: hashlib.sha256(
                f"v9-router:{class_index}:{indices[position]}".encode()
            ).digest(),
        )
        for rank, position in enumerate(ranked):
            folds[position] = rank % ROUTER_FOLDS
    return folds


def gain_regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=17,
    )


def cluster_summary(values: np.ndarray, clusters: np.ndarray, generator) -> dict[str, object]:
    grouped = [values[clusters == cluster] for cluster in sorted(np.unique(clusters))]
    if len({len(block) for block in grouped}) != 1:
        raise RuntimeError("V9 requires equal support across coarse-class clusters")
    block_means = np.asarray([block.mean() for block in grouped])
    selected = generator.integers(
        0, len(grouped), size=(BOOTSTRAP_REPLICATES, len(grouped))
    )
    replicates = block_means[selected].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "coarse_class_cluster_95_percentile_interval": np.quantile(
            replicates, (0.025, 0.975)
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
        "sample_count": len(values),
        "cluster_count": len(grouped),
    }


def main() -> int:
    started = time.perf_counter()
    with np.load(FEATURE_PATH) as cache:
        sample_indices = cache["sample_index"]
        clip_features = cache["clip"].astype(np.float64)
        resnet_features = cache["resnet18"].astype(np.float64)
        labels = cache["fine_labels"]
        coarse_labels = cache["coarse_labels"]
        partition = cache["partition"]
    fit_mask = partition == "candidate_fit"
    calibration_mask = partition == "candidate_calib"
    router_mask = partition == "router"

    clip_model, clip_temperature, clip_selection = fit_candidate(
        clip_features, labels, fit_mask, calibration_mask
    )
    resnet_model, resnet_temperature, resnet_selection = fit_candidate(
        resnet_features, labels, fit_mask, calibration_mask
    )
    clip_logits = clip_model.predict(clip_features) / clip_temperature
    resnet_logits = resnet_model.predict(resnet_features) / resnet_temperature

    calibration_a = per_example_nll(clip_logits[calibration_mask], labels[calibration_mask])
    calibration_b = per_example_nll(
        resnet_logits[calibration_mask], labels[calibration_mask]
    )
    calibration_gain = calibration_a - calibration_b
    selected_constant_q = float(calibration_gain.mean() > 0)
    q_grid = np.linspace(0.0, 1.0, 21)
    selected_static_q = float(
        q_grid[
            int(
                np.argmin(
                    [
                        multiclass_nll(
                            clip_logits[calibration_mask]
                            + q
                            * (
                                resnet_logits[calibration_mask]
                                - clip_logits[calibration_mask]
                            ),
                            labels[calibration_mask],
                        )
                        for q in q_grid
                    ]
                )
            )
        ]
    )

    router_indices = sample_indices[router_mask]
    router_labels = labels[router_mask]
    router_coarse = coarse_labels[router_mask]
    logits_a = clip_logits[router_mask]
    logits_b = resnet_logits[router_mask]
    matrix = router_features(logits_a, logits_b)
    folds = deterministic_router_folds(router_indices, router_labels)
    hard_q = np.empty(len(router_labels), dtype=np.float64)
    direct_soft_q = np.empty(len(router_labels), dtype=np.float64)
    chosen_l2: list[float] = []
    for outer_fold in range(ROUTER_FOLDS):
        calibration_fold = (outer_fold + 1) % ROUTER_FOLDS
        evaluate = folds == outer_fold
        calibrate = folds == calibration_fold
        train = ~(evaluate | calibrate)
        train_x, calibration_x, mean, scale = standardize(matrix[train], matrix[calibrate])
        evaluation_x = (matrix[evaluate] - mean) / scale

        regressor = gain_regressor()
        training_gain = per_example_nll(logits_a[train], router_labels[train]) - per_example_nll(
            logits_b[train], router_labels[train]
        )
        regressor.fit(matrix[train], training_gain)
        hard_q[evaluate] = (regressor.predict(matrix[evaluate]) > 0).astype(np.float64)

        soft_candidates = []
        for l2 in DIRECT_SOFT_L2_GRID:
            parameters = fit_direct_soft(
                train_x, logits_a[train], logits_b[train], router_labels[train], l2
            )
            calibration_q = predict_direct_soft(parameters, calibration_x)
            calibration_loss = multiclass_nll(
                logits_a[calibrate]
                + calibration_q[:, None]
                * (logits_b[calibrate] - logits_a[calibrate]),
                router_labels[calibrate],
            )
            soft_candidates.append((calibration_loss, l2, parameters))
        _, l2, parameters = min(soft_candidates, key=lambda item: item[0])
        chosen_l2.append(l2)
        direct_soft_q[evaluate] = predict_direct_soft(parameters, evaluation_x)

    loss_a = per_example_nll(logits_a, router_labels)
    loss_b = per_example_nll(logits_b, router_labels)
    base_gain = loss_a - per_example_nll(
        logits_a + selected_constant_q * (logits_b - logits_a), router_labels
    )
    endpoint_oracle_gain = loss_a - np.minimum(loss_a, loss_b)
    static_gain = loss_a - per_example_nll(
        logits_a + selected_static_q * (logits_b - logits_a), router_labels
    )
    hard_gain = loss_a - per_example_nll(
        logits_a + hard_q[:, None] * (logits_b - logits_a), router_labels
    )
    soft_gain = loss_a - per_example_nll(
        logits_a + direct_soft_q[:, None] * (logits_b - logits_a), router_labels
    )

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {
        "selected_constant_gain": base_gain,
        "selected_static_gain": static_gain,
        "direct_gain_hard_gain": hard_gain,
        "direct_soft_gain": soft_gain,
        "endpoint_oracle_gain": endpoint_oracle_gain,
        "hard_beyond_selected_constant": hard_gain - base_gain,
        "direct_soft_beyond_selected_constant": soft_gain - base_gain,
        "direct_soft_minus_hard": soft_gain - hard_gain,
    }

    # Select final soft regularization by complete OOF loss, then fit final test-facing routers.
    selected_final_l2 = float(
        max(
            DIRECT_SOFT_L2_GRID,
            key=lambda l2: (chosen_l2.count(l2), -DIRECT_SOFT_L2_GRID.index(l2)),
        )
    )
    all_x, _, router_mean, router_scale = standardize(matrix, matrix)
    final_soft_parameters = fit_direct_soft(
        all_x, logits_a, logits_b, router_labels, selected_final_l2
    )
    final_hard_model = gain_regressor()
    final_hard_model.fit(matrix, loss_a - loss_b)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        MODEL_PATH,
        clip_coef=clip_model.coef_,
        clip_intercept=clip_model.intercept_,
        clip_temperature=np.asarray(clip_temperature),
        resnet_coef=resnet_model.coef_,
        resnet_intercept=resnet_model.intercept_,
        resnet_temperature=np.asarray(resnet_temperature),
        selected_constant_q=np.asarray(selected_constant_q),
        selected_static_q=np.asarray(selected_static_q),
        router_mean=router_mean,
        router_scale=router_scale,
        direct_soft_parameters=final_soft_parameters,
        direct_soft_l2=np.asarray(selected_final_l2),
    )
    with ROUTER_PATH.open("wb") as handle:
        pickle.dump(final_hard_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    report = {
        "schema_version": 1,
        "protocol": "v9b0-cifar100-independent-candidate-feasibility",
        "development_only": True,
        "official_test_accessed": False,
        "candidate_fit_count": int(fit_mask.sum()),
        "candidate_calibration_count": int(calibration_mask.sum()),
        "router_count": int(router_mask.sum()),
        "candidate_selection": {"clip": clip_selection, "resnet18": resnet_selection},
        "candidate_calibration_accuracy": {
            "clip": float(
                np.mean(clip_logits[calibration_mask].argmax(axis=1) == labels[calibration_mask])
            ),
            "resnet18": float(
                np.mean(
                    resnet_logits[calibration_mask].argmax(axis=1)
                    == labels[calibration_mask]
                )
            ),
        },
        "selected_constant_q": selected_constant_q,
        "selected_static_q": selected_static_q,
        "selected_l2_by_outer_fold": chosen_l2,
        "selected_final_l2": selected_final_l2,
        "direct_soft_interior_fraction": float(
            np.mean((direct_soft_q > 0) & (direct_soft_q < 1))
        ),
        "summary": {
            name: cluster_summary(values, router_coarse, generator)
            for name, values in metrics.items()
        },
        "feasibility": {
            "oracle_beyond_selected_constant_positive": bool(
                (endpoint_oracle_gain - base_gain).mean() > 0
            ),
            "hard_beyond_selected_constant_interval_positive": bool(
                cluster_summary(
                    hard_gain - base_gain,
                    router_coarse,
                    np.random.default_rng(BOOTSTRAP_SEED + 1),
                )["coarse_class_cluster_95_percentile_interval"][0]
                > 0
            ),
            "direct_soft_minus_hard_interval_positive": bool(
                cluster_summary(
                    soft_gain - hard_gain,
                    router_coarse,
                    np.random.default_rng(BOOTSTRAP_SEED + 2),
                )["coarse_class_cluster_95_percentile_interval"][0]
                > 0
            ),
        },
        "artifact_sha256": {
            "plan": sha256(PLAN_PATH),
            "feature_cache": sha256(FEATURE_PATH),
            "script": sha256(Path(__file__)),
            "candidate_heads": sha256(MODEL_PATH),
            "final_routers": sha256(ROUTER_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
