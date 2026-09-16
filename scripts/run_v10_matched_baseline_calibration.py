#!/usr/bin/env python3
"""Run the V10 post-confirmation matched-baseline and calibration audit."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from evaluate_v9b_cifar100_sealed import (
    candidate_logits,
    cluster_summary,
    read_test_labels,
    verify_freeze,
)
from run_v9b0_cifar100_feasibility import (
    BOOTSTRAP_SEED,
    DIRECT_SOFT_L2_GRID,
    RIDGE_GRID,
    deterministic_router_folds,
    fit_direct_soft,
    gain_regressor,
    multiclass_nll,
    per_example_nll,
    predict_direct_soft,
    router_features,
    standardize,
)
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_FEATURE_PATH = REPO_ROOT / "data/v9_cifar100_features/train.npz"
TEST_FEATURE_PATH = REPO_ROOT / "data/v9_cifar100_features/test-unlabeled.npz"
MODEL_PATH = REPO_ROOT / "runs/v9_cifar100/candidate_heads.npz"
V9_REPORT_PATH = REPO_ROOT / "reports/v9b_cifar100_test_confirmation.json"
V10_FREEZE_PATH = REPO_ROOT / "reports/v10_matched_baseline_calibration_freeze.json"
PLAN_PATH = REPO_ROOT / "docs/v10_matched_baseline_calibration_plan.md"
OUTPUT_PATH = REPO_ROOT / "reports/v10_matched_baseline_calibration.json"

TEMPERATURE_VARIANTS = ("frozen_bound", "wide_bound", "unscaled")
TEMPERATURE_PROFILE_LOG_VALUES = tuple(float(value) for value in range(-8, 4))


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def fit_temperature_with_bounds(
    logits: np.ndarray, labels: np.ndarray, bounds: tuple[float, float]
) -> tuple[float, float]:
    result = minimize_scalar(
        lambda log_temperature: multiclass_nll(logits / np.exp(log_temperature), labels),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("wide-bound temperature optimization failed")
    return float(np.exp(result.x)), float(result.fun)


def fit_static_action(logits_a: np.ndarray, logits_b: np.ndarray, labels: np.ndarray) -> float:
    residual = logits_b - logits_a
    result = minimize_scalar(
        lambda q: multiclass_nll(logits_a + q * residual, labels),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("matched static action optimization failed")
    return float(result.x)


def select_calibration_static(
    logits_a: np.ndarray, logits_b: np.ndarray, labels: np.ndarray
) -> float:
    grid = np.linspace(0.0, 1.0, 21)
    losses = [multiclass_nll(logits_a + q * (logits_b - logits_a), labels) for q in grid]
    return float(grid[int(np.argmin(losses))])


def modal_grid_value(values: list[float], grid: tuple[float, ...]) -> float:
    return float(max(grid, key=lambda value: (values.count(value), -grid.index(value))))


def summarize_gains(
    gains: dict[str, np.ndarray], clusters: np.ndarray, seed: int
) -> dict[str, dict[str, object]]:
    generator = np.random.default_rng(seed)
    return {name: cluster_summary(values, clusters, generator) for name, values in gains.items()}


def compute_gains(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    actions: dict[str, np.ndarray],
    selected_constant_q: float,
    selected_calibration_static_q: float,
) -> dict[str, np.ndarray]:
    loss_a = per_example_nll(logits_a, labels)
    loss_b = per_example_nll(logits_b, labels)
    residual = logits_b - logits_a

    def action_gain(q: np.ndarray | float) -> np.ndarray:
        return loss_a - per_example_nll(logits_a + q * residual, labels)

    constant = action_gain(selected_constant_q)
    gains = {
        "selected_constant_gain": constant,
        "calibration_selected_static_gain": action_gain(selected_calibration_static_q),
        "label_budget_matched_static_gain": action_gain(
            actions["label_budget_matched_static"][:, None]
        ),
        "tree_hard_gain": action_gain(actions["tree_hard"][:, None]),
        "linear_gain_hard_gain": action_gain(actions["linear_gain_hard"][:, None]),
        "same_score_projected_hard_gain": action_gain(
            actions["same_score_projected_hard"][:, None]
        ),
        "direct_soft_gain": action_gain(actions["direct_soft"][:, None]),
        "endpoint_oracle_gain": loss_a - np.minimum(loss_a, loss_b),
    }
    gains.update(
        {
            "tree_hard_beyond_selected_constant": gains["tree_hard_gain"] - constant,
            "linear_gain_hard_beyond_selected_constant": gains["linear_gain_hard_gain"] - constant,
            "same_score_projected_hard_beyond_selected_constant": gains[
                "same_score_projected_hard_gain"
            ]
            - constant,
            "direct_soft_beyond_selected_constant": gains["direct_soft_gain"] - constant,
            "direct_soft_minus_label_budget_matched_static": gains["direct_soft_gain"]
            - gains["label_budget_matched_static_gain"],
            "direct_soft_minus_tree_hard": gains["direct_soft_gain"] - gains["tree_hard_gain"],
            "direct_soft_minus_linear_gain_hard": gains["direct_soft_gain"]
            - gains["linear_gain_hard_gain"],
            "direct_soft_minus_same_score_projected_hard": gains["direct_soft_gain"]
            - gains["same_score_projected_hard_gain"],
            "oracle_beyond_selected_constant": gains["endpoint_oracle_gain"] - constant,
        }
    )
    if not all(np.isfinite(values).all() for values in gains.values()):
        raise RuntimeError("non-finite V10 gains")
    return gains


def fit_variant(
    logits_a: np.ndarray,
    logits_b: np.ndarray,
    labels: np.ndarray,
    coarse_labels: np.ndarray,
    sample_indices: np.ndarray,
    router_mask: np.ndarray,
    calibration_mask: np.ndarray,
    test_logits_a: np.ndarray,
    test_logits_b: np.ndarray,
    bootstrap_seed: int,
) -> dict[str, object]:
    calibration_a = per_example_nll(logits_a[calibration_mask], labels[calibration_mask])
    calibration_b = per_example_nll(logits_b[calibration_mask], labels[calibration_mask])
    selected_constant_q = float(calibration_b.mean() < calibration_a.mean())
    selected_calibration_static_q = select_calibration_static(
        logits_a[calibration_mask], logits_b[calibration_mask], labels[calibration_mask]
    )

    router_indices = sample_indices[router_mask]
    router_labels = labels[router_mask]
    router_coarse = coarse_labels[router_mask]
    router_logits_a = logits_a[router_mask]
    router_logits_b = logits_b[router_mask]
    matrix = router_features(router_logits_a, router_logits_b)
    folds = deterministic_router_folds(router_indices, router_labels)

    actions = {
        name: np.empty(len(router_labels), dtype=np.float64)
        for name in (
            "label_budget_matched_static",
            "tree_hard",
            "linear_gain_hard",
            "same_score_projected_hard",
            "direct_soft",
        )
    }
    selected_l2: list[float] = []
    selected_linear_alpha: list[float] = []
    selected_static_by_fold: list[float] = []

    for outer_fold in range(5):
        calibration_fold = (outer_fold + 1) % 5
        evaluate = folds == outer_fold
        calibrate = folds == calibration_fold
        train = ~(evaluate | calibrate)
        train_x, calibration_x, mean, scale = standardize(matrix[train], matrix[calibrate])
        evaluation_x = (matrix[evaluate] - mean) / scale

        static_q = fit_static_action(
            router_logits_a[train], router_logits_b[train], router_labels[train]
        )
        selected_static_by_fold.append(static_q)
        actions["label_budget_matched_static"][evaluate] = static_q

        tree = gain_regressor()
        training_gain = per_example_nll(
            router_logits_a[train], router_labels[train]
        ) - per_example_nll(router_logits_b[train], router_labels[train])
        tree.fit(matrix[train], training_gain)
        actions["tree_hard"][evaluate] = (tree.predict(matrix[evaluate]) > 0).astype(np.float64)

        soft_candidates: list[tuple[float, float, np.ndarray]] = []
        for l2 in DIRECT_SOFT_L2_GRID:
            parameters = fit_direct_soft(
                train_x,
                router_logits_a[train],
                router_logits_b[train],
                router_labels[train],
                l2,
            )
            calibration_q = predict_direct_soft(parameters, calibration_x)
            loss = multiclass_nll(
                router_logits_a[calibrate]
                + calibration_q[:, None]
                * (router_logits_b[calibrate] - router_logits_a[calibrate]),
                router_labels[calibrate],
            )
            soft_candidates.append((loss, l2, parameters))
        _, chosen_l2, parameters = min(soft_candidates, key=lambda item: item[0])
        selected_l2.append(chosen_l2)
        soft_q = predict_direct_soft(parameters, evaluation_x)
        actions["direct_soft"][evaluate] = soft_q
        actions["same_score_projected_hard"][evaluate] = (soft_q >= 0.5).astype(np.float64)

        linear_candidates: list[tuple[float, float, Ridge]] = []
        for alpha in RIDGE_GRID:
            linear = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
            linear.fit(train_x, training_gain)
            calibration_q = (linear.predict(calibration_x) > 0).astype(np.float64)
            loss = multiclass_nll(
                router_logits_a[calibrate]
                + calibration_q[:, None]
                * (router_logits_b[calibrate] - router_logits_a[calibrate]),
                router_labels[calibrate],
            )
            linear_candidates.append((loss, alpha, linear))
        _, chosen_alpha, linear = min(linear_candidates, key=lambda item: item[0])
        selected_linear_alpha.append(chosen_alpha)
        actions["linear_gain_hard"][evaluate] = (linear.predict(evaluation_x) > 0).astype(
            np.float64
        )

    oof_gains = compute_gains(
        router_logits_a,
        router_logits_b,
        router_labels,
        actions,
        selected_constant_q,
        selected_calibration_static_q,
    )

    final_static_q = fit_static_action(router_logits_a, router_logits_b, router_labels)
    all_x, _, mean, scale = standardize(matrix, matrix)
    final_l2 = modal_grid_value(selected_l2, DIRECT_SOFT_L2_GRID)
    final_soft_parameters = fit_direct_soft(
        all_x, router_logits_a, router_logits_b, router_labels, final_l2
    )
    final_alpha = modal_grid_value(selected_linear_alpha, RIDGE_GRID)
    final_gain = per_example_nll(router_logits_a, router_labels) - per_example_nll(
        router_logits_b, router_labels
    )
    final_linear = Ridge(alpha=final_alpha, fit_intercept=True, solver="cholesky")
    final_linear.fit(all_x, final_gain)
    final_tree = gain_regressor()
    final_tree.fit(matrix, final_gain)

    test_matrix = router_features(test_logits_a, test_logits_b)
    test_x = (test_matrix - mean) / scale
    test_soft_q = predict_direct_soft(final_soft_parameters, test_x)
    test_actions = {
        "label_budget_matched_static": np.full(
            len(test_logits_a), final_static_q, dtype=np.float64
        ),
        "tree_hard": (final_tree.predict(test_matrix) > 0).astype(np.float64),
        "linear_gain_hard": (final_linear.predict(test_x) > 0).astype(np.float64),
        "same_score_projected_hard": (test_soft_q >= 0.5).astype(np.float64),
        "direct_soft": test_soft_q,
    }
    return {
        "candidate_calibration": {
            "clip_nll": float(calibration_a.mean()),
            "resnet18_nll": float(calibration_b.mean()),
            "clip_accuracy": float(
                np.mean(logits_a[calibration_mask].argmax(axis=1) == labels[calibration_mask])
            ),
            "resnet18_accuracy": float(
                np.mean(logits_b[calibration_mask].argmax(axis=1) == labels[calibration_mask])
            ),
        },
        "selected_constant_q": selected_constant_q,
        "calibration_selected_static_q": selected_calibration_static_q,
        "label_budget_matched_static_q_by_fold": selected_static_by_fold,
        "final_label_budget_matched_static_q": final_static_q,
        "selected_direct_soft_l2_by_fold": selected_l2,
        "final_direct_soft_l2": final_l2,
        "selected_linear_gain_alpha_by_fold": selected_linear_alpha,
        "final_linear_gain_alpha": final_alpha,
        "oof_summary": summarize_gains(oof_gains, router_coarse, bootstrap_seed),
        "test_bundle": {
            "logits_a": test_logits_a,
            "logits_b": test_logits_b,
            "actions": test_actions,
        },
    }


def main() -> int:
    if OUTPUT_PATH.exists():
        raise RuntimeError("refusing to overwrite V10 results")
    started = time.perf_counter()
    verify_freeze()
    freeze = json.loads(V10_FREEZE_PATH.read_text())
    if freeze.get("frozen") is not True or freeze.get("v9_test_already_accessed") is not True:
        raise RuntimeError("invalid V10 post-confirmation freeze")
    for relative_path, expected in freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"post-freeze V10 artifact change: {relative_path}")

    with np.load(TRAIN_FEATURE_PATH) as cache:
        sample_indices = cache["sample_index"]
        clip_features = cache["clip"].astype(np.float64)
        resnet_features = cache["resnet18"].astype(np.float64)
        labels = cache["fine_labels"]
        coarse_labels = cache["coarse_labels"]
        partition = cache["partition"]
    with np.load(TEST_FEATURE_PATH) as cache:
        test_clip_features = cache["clip"].astype(np.float64)
        test_resnet_features = cache["resnet18"].astype(np.float64)
    with np.load(MODEL_PATH) as models:
        clip_coef = models["clip_coef"]
        clip_intercept = models["clip_intercept"]
        resnet_coef = models["resnet_coef"]
        resnet_intercept = models["resnet_intercept"]
        frozen_temperatures = {
            "clip": float(models["clip_temperature"]),
            "resnet18": float(models["resnet_temperature"]),
        }

    raw_clip = candidate_logits(clip_features, clip_coef, clip_intercept, 1.0)
    raw_resnet = candidate_logits(resnet_features, resnet_coef, resnet_intercept, 1.0)
    raw_test_clip = candidate_logits(test_clip_features, clip_coef, clip_intercept, 1.0)
    raw_test_resnet = candidate_logits(test_resnet_features, resnet_coef, resnet_intercept, 1.0)
    calibration_mask = partition == "candidate_calib"
    router_mask = partition == "router"
    wide_clip, wide_clip_nll = fit_temperature_with_bounds(
        raw_clip[calibration_mask], labels[calibration_mask], (-8.0, 3.0)
    )
    wide_resnet, wide_resnet_nll = fit_temperature_with_bounds(
        raw_resnet[calibration_mask], labels[calibration_mask], (-8.0, 3.0)
    )
    temperatures = {
        "frozen_bound": frozen_temperatures,
        "wide_bound": {"clip": wide_clip, "resnet18": wide_resnet},
        "unscaled": {"clip": 1.0, "resnet18": 1.0},
    }
    profile = {
        candidate: [
            {
                "log_temperature": log_temperature,
                "temperature": float(np.exp(log_temperature)),
                "candidate_calibration_nll": multiclass_nll(
                    logits / np.exp(log_temperature), labels[calibration_mask]
                ),
            }
            for log_temperature in TEMPERATURE_PROFILE_LOG_VALUES
        ]
        for candidate, logits in {
            "clip": raw_clip[calibration_mask],
            "resnet18": raw_resnet[calibration_mask],
        }.items()
    }

    fitted: dict[str, dict[str, object]] = {}
    for variant_index, variant in enumerate(TEMPERATURE_VARIANTS):
        temperature = temperatures[variant]
        fitted[variant] = fit_variant(
            raw_clip / temperature["clip"],
            raw_resnet / temperature["resnet18"],
            labels,
            coarse_labels,
            sample_indices,
            router_mask,
            calibration_mask,
            raw_test_clip / temperature["clip"],
            raw_test_resnet / temperature["resnet18"],
            BOOTSTRAP_SEED + 100 * variant_index,
        )

    # All temperatures, models, hyperparameters, and test actions are fixed before this read.
    test_labels, test_coarse = read_test_labels()
    variants: dict[str, dict[str, object]] = {}
    for variant_index, variant in enumerate(TEMPERATURE_VARIANTS):
        result = fitted[variant]
        bundle = result.pop("test_bundle")
        test_gains = compute_gains(
            bundle["logits_a"],
            bundle["logits_b"],
            test_labels,
            bundle["actions"],
            result["selected_constant_q"],
            result["calibration_selected_static_q"],
        )
        loss_a = per_example_nll(bundle["logits_a"], test_labels)
        loss_b = per_example_nll(bundle["logits_b"], test_labels)
        result["temperatures"] = temperatures[variant]
        result["test_candidate"] = {
            "clip_nll": float(loss_a.mean()),
            "resnet18_nll": float(loss_b.mean()),
            "clip_accuracy": float(np.mean(bundle["logits_a"].argmax(axis=1) == test_labels)),
            "resnet18_accuracy": float(np.mean(bundle["logits_b"].argmax(axis=1) == test_labels)),
        }
        result["test_action_statistics"] = {
            name: {
                "mean": float(q.mean()),
                "standard_deviation": float(q.std()),
                "minimum": float(q.min()),
                "maximum": float(q.max()),
                "fraction_choose_resnet": float(np.mean(q >= 0.5)),
            }
            for name, q in bundle["actions"].items()
        }
        result["test_summary"] = summarize_gains(
            test_gains, test_coarse, BOOTSTRAP_SEED + 1000 + 100 * variant_index
        )
        variants[variant] = result

    v9 = json.loads(V9_REPORT_PATH.read_text())
    frozen_test = variants["frozen_bound"]["test_summary"]
    reproduction_errors = {
        "direct_soft_gain": abs(
            frozen_test["direct_soft_gain"]["mean"] - v9["summary"]["direct_soft_gain"]["mean"]
        ),
        "tree_hard_gain": abs(
            frozen_test["tree_hard_gain"]["mean"] - v9["summary"]["direct_gain_hard_gain"]["mean"]
        ),
    }
    if max(reproduction_errors.values()) > 1e-10:
        raise RuntimeError(f"V9 reproduction drift: {reproduction_errors}")

    def positive_interval(variant: str, endpoint: str) -> bool:
        item = variants[variant]["test_summary"][endpoint]
        return bool(item["mean"] > 0 and item["coarse_class_cluster_95_percentile_interval"][0] > 0)

    report = {
        "schema_version": 1,
        "protocol": "v10-post-confirmation-matched-baseline-calibration-audit",
        "posthoc_after_v9_test_access": True,
        "changes_v9_registered_decision": False,
        "temperature_profile": profile,
        "wide_bound_optimum": {
            "clip": {"temperature": wide_clip, "candidate_calibration_nll": wide_clip_nll},
            "resnet18": {
                "temperature": wide_resnet,
                "candidate_calibration_nll": wide_resnet_nll,
            },
        },
        "variants": variants,
        "wide_bound_robustness_checks": {
            "tree_hard_beyond_constant_interval_positive": positive_interval(
                "wide_bound", "tree_hard_beyond_selected_constant"
            ),
            "direct_soft_beyond_matched_static_interval_positive": positive_interval(
                "wide_bound", "direct_soft_minus_label_budget_matched_static"
            ),
            "direct_soft_beyond_same_score_hard_interval_positive": positive_interval(
                "wide_bound", "direct_soft_minus_same_score_projected_hard"
            ),
        },
        "v9_reproduction_absolute_errors": reproduction_errors,
        "artifact_sha256": {
            "plan": sha256(PLAN_PATH),
            "script": sha256(Path(__file__)),
            "v10_freeze": sha256(V10_FREEZE_PATH),
            "v9_freeze": sha256(REPO_ROOT / "reports/v9b_cifar100_test_freeze.json"),
            "v9_confirmation": sha256(V9_REPORT_PATH),
            "candidate_heads": sha256(MODEL_PATH),
            "train_features": sha256(TRAIN_FEATURE_PATH),
            "test_features": sha256(TEST_FEATURE_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    compact = {
        "posthoc_after_v9_test_access": True,
        "wide_bound_optimum": report["wide_bound_optimum"],
        "wide_bound_robustness_checks": report["wide_bound_robustness_checks"],
        "v9_reproduction_absolute_errors": reproduction_errors,
        "test_key_results": {
            variant: {
                endpoint: variants[variant]["test_summary"][endpoint]
                for endpoint in (
                    "label_budget_matched_static_gain",
                    "tree_hard_gain",
                    "linear_gain_hard_gain",
                    "same_score_projected_hard_gain",
                    "direct_soft_gain",
                    "direct_soft_minus_label_budget_matched_static",
                    "direct_soft_minus_tree_hard",
                    "direct_soft_minus_linear_gain_hard",
                    "direct_soft_minus_same_score_projected_hard",
                )
            }
            for variant in TEMPERATURE_VARIANTS
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
