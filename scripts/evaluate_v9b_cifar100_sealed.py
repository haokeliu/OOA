#!/usr/bin/env python3
"""Run the one-shot sealed CIFAR-100 V9-B evaluation."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

import numpy as np
from run_v9b0_cifar100_feasibility import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    per_example_nll,
    predict_direct_soft,
    router_features,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "reports/v9b_cifar100_test_freeze.json"
FEATURE_PATH = REPO_ROOT / "data/v9_cifar100_features/test-unlabeled.npz"
TEST_BATCH_PATH = REPO_ROOT / "data/v9_cifar100/cifar-100-python/test"
MODEL_PATH = REPO_ROOT / "runs/v9_cifar100/candidate_heads.npz"
ROUTER_PATH = REPO_ROOT / "runs/v9_cifar100/final_routers.pkl"
OUTPUT_PATH = REPO_ROOT / "reports/v9b_cifar100_test_confirmation.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("frozen") is not True or freeze.get("test_labels_accessed") is not False:
        raise RuntimeError("invalid V9-B freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        path = REPO_ROOT / relative_path
        if sha256(path) != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def read_test_labels() -> tuple[np.ndarray, np.ndarray]:
    with TEST_BATCH_PATH.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    fine = np.asarray(payload[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(payload[b"coarse_labels"], dtype=np.int64)
    return fine, coarse


def cluster_summary(
    values: np.ndarray, clusters: np.ndarray, generator: np.random.Generator
) -> dict[str, object]:
    unique = np.unique(clusters)
    if not np.array_equal(unique, np.arange(20)):
        raise RuntimeError("unexpected CIFAR-100 coarse-class indexes")
    blocks = [values[clusters == cluster] for cluster in unique]
    if any(len(block) != 500 for block in blocks):
        raise RuntimeError("CIFAR-100 test coarse classes must each contain 500 examples")
    block_means = np.asarray([block.mean() for block in blocks])
    sampled = generator.integers(0, 20, size=(BOOTSTRAP_REPLICATES, 20))
    replicates = block_means[sampled].mean(axis=1)
    interval = np.quantile(replicates, (0.025, 0.975)).tolist()
    return {
        "mean": float(values.mean()),
        "coarse_class_cluster_95_percentile_interval": interval,
        "fraction_positive": float(np.mean(replicates > 0)),
        "sample_count": len(values),
        "cluster_count": 20,
    }


def candidate_logits(
    features: np.ndarray, coef: np.ndarray, intercept: np.ndarray, temperature: float
) -> np.ndarray:
    return (features @ coef.T + intercept) / temperature


def main() -> int:
    if OUTPUT_PATH.exists():
        raise RuntimeError("refusing to overwrite the one-shot V9-B confirmation")
    started = time.perf_counter()
    freeze = verify_freeze()

    with np.load(FEATURE_PATH) as features:
        sample_index = features["sample_index"]
        clip_features = features["clip"].astype(np.float64)
        resnet_features = features["resnet18"].astype(np.float64)
        if set(features.files) != {"sample_index", "clip", "resnet18"}:
            raise RuntimeError("label-blind test cache exposes unexpected arrays")
    if not np.array_equal(sample_index, np.arange(10_000)):
        raise RuntimeError("test feature indexes are not the complete ordered test split")

    with np.load(MODEL_PATH) as models:
        logits_a = candidate_logits(
            clip_features,
            models["clip_coef"],
            models["clip_intercept"],
            float(models["clip_temperature"]),
        )
        logits_b = candidate_logits(
            resnet_features,
            models["resnet_coef"],
            models["resnet_intercept"],
            float(models["resnet_temperature"]),
        )
        selected_constant_q = float(models["selected_constant_q"])
        selected_static_q = float(models["selected_static_q"])
        router_mean = models["router_mean"]
        router_scale = models["router_scale"]
        direct_soft_parameters = models["direct_soft_parameters"]
    if not np.isfinite(logits_a).all() or not np.isfinite(logits_b).all():
        raise RuntimeError("non-finite candidate logits")
    matrix = router_features(logits_a, logits_b)
    with ROUTER_PATH.open("rb") as handle:
        hard_model = pickle.load(handle)
    hard_q = (hard_model.predict(matrix) > 0).astype(np.float64)
    standardized = (matrix - router_mean) / router_scale
    soft_q = predict_direct_soft(direct_soft_parameters, standardized)
    if not np.all((soft_q > 0.0) & (soft_q < 1.0)):
        raise RuntimeError("direct-soft actions must be strictly interior")

    # This is the only point at which the V9 experiment indexes test labels.
    fine_labels, coarse_labels = read_test_labels()
    if len(fine_labels) != 10_000 or len(coarse_labels) != 10_000:
        raise RuntimeError("unexpected CIFAR-100 test support")
    if not np.array_equal(np.bincount(fine_labels, minlength=100), np.full(100, 100)):
        raise RuntimeError("CIFAR-100 test fine classes must each contain 100 examples")

    loss_a = per_example_nll(logits_a, fine_labels)
    loss_b = per_example_nll(logits_b, fine_labels)
    residual = logits_b - logits_a
    gains = {
        "selected_constant_gain": loss_a
        - per_example_nll(logits_a + selected_constant_q * residual, fine_labels),
        "selected_static_gain": loss_a
        - per_example_nll(logits_a + selected_static_q * residual, fine_labels),
        "direct_gain_hard_gain": loss_a
        - per_example_nll(logits_a + hard_q[:, None] * residual, fine_labels),
        "direct_soft_gain": loss_a
        - per_example_nll(logits_a + soft_q[:, None] * residual, fine_labels),
        "endpoint_oracle_gain": loss_a - np.minimum(loss_a, loss_b),
    }
    gains["hard_beyond_selected_constant"] = (
        gains["direct_gain_hard_gain"] - gains["selected_constant_gain"]
    )
    gains["direct_soft_beyond_selected_constant"] = (
        gains["direct_soft_gain"] - gains["selected_constant_gain"]
    )
    gains["direct_soft_minus_hard"] = gains["direct_soft_gain"] - gains["direct_gain_hard_gain"]
    gains["oracle_beyond_selected_constant"] = (
        gains["endpoint_oracle_gain"] - gains["selected_constant_gain"]
    )
    if not all(np.isfinite(values).all() for values in gains.values()):
        raise RuntimeError("non-finite V9-B test loss")

    generator = np.random.default_rng(BOOTSTRAP_SEED)
    summary = {
        name: cluster_summary(values, coarse_labels, generator) for name, values in gains.items()
    }

    def passes(name: str) -> bool:
        item = summary[name]
        return bool(item["mean"] > 0 and item["coarse_class_cluster_95_percentile_interval"][0] > 0)

    decisions = {
        "independent_family_opportunity": passes("oracle_beyond_selected_constant"),
        "recoverable_hard_routing_gain": passes("hard_beyond_selected_constant"),
        "continuous_action_advantage": passes("direct_soft_minus_hard"),
    }
    decisions["independent_family_routing_confirmation"] = bool(
        decisions["independent_family_opportunity"] and decisions["recoverable_hard_routing_gain"]
    )
    report = {
        "schema_version": 1,
        "protocol": "v9b-cifar100-prospective-local-seal",
        "freeze_created_utc": freeze["created_utc"],
        "test_labels_accessed": True,
        "test_pixels_accessed": True,
        "sample_count": 10_000,
        "fine_class_count": 100,
        "coarse_class_count": 20,
        "selected_constant_q": selected_constant_q,
        "selected_static_q": selected_static_q,
        "candidate_test": {
            "clip_accuracy": float(np.mean(logits_a.argmax(axis=1) == fine_labels)),
            "resnet18_accuracy": float(np.mean(logits_b.argmax(axis=1) == fine_labels)),
            "clip_nll": float(loss_a.mean()),
            "resnet18_nll": float(loss_b.mean()),
        },
        "direct_soft_action": {
            "mean": float(soft_q.mean()),
            "standard_deviation": float(soft_q.std()),
            "minimum": float(soft_q.min()),
            "maximum": float(soft_q.max()),
            "interior_fraction": float(np.mean((soft_q > 0.0) & (soft_q < 1.0))),
        },
        "summary": summary,
        "registered_decisions": decisions,
        "artifact_sha256": {
            "test_batch": sha256(TEST_BATCH_PATH),
            "test_feature_cache": sha256(FEATURE_PATH),
            "candidate_heads": sha256(MODEL_PATH),
            "final_routers": sha256(ROUTER_PATH),
            "freeze": sha256(FREEZE_PATH),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
