#!/usr/bin/env python3
"""Freeze V9-B artifacts before the first test-label access."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reports/v9b_cifar100_test_freeze.json"
CONFIRMATION_PATH = REPO_ROOT / "reports/v9b_cifar100_test_confirmation.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    if OUTPUT_PATH.exists():
        raise RuntimeError("V9-B test freeze already exists; refusing to overwrite it")
    if CONFIRMATION_PATH.exists():
        raise RuntimeError("refusing to freeze after a V9-B confirmation appeared")

    paths = [
        Path("data/checkpoints/ViT-B-16.pt"),
        Path("data/v9_cifar100/cifar-100-python.tar.gz"),
        Path("data/v9_cifar100/cifar-100-python/train"),
        Path("data/v9_cifar100/cifar-100-python/test"),
        Path("data/v9_cifar100/torch_home/hub/checkpoints/resnet18-f37072fd.pth"),
        Path("data/v9_cifar100_features/train.npz"),
        Path("data/v9_cifar100_features/test-unlabeled.npz"),
        Path("docs/v9_independent_candidates_and_direct_soft_plan.md"),
        Path("docs/v9b_cifar100_test_preregistration.md"),
        Path("reports/v9b0_cifar100_preparation.json"),
        Path("reports/v9b0_cifar100_train_feature_cache.json"),
        Path("reports/v9b0_cifar100_test-unlabeled_feature_cache.json"),
        Path("reports/v9b0_cifar100_feasibility.json"),
        Path("runs/v9_cifar100/candidate_heads.npz"),
        Path("runs/v9_cifar100/final_routers.pkl"),
        Path("scripts/cache_v9_cifar100_features.py"),
        Path("scripts/evaluate_v9b_cifar100_sealed.py"),
        Path("scripts/freeze_v9b_cifar100_test.py"),
        Path("scripts/prepare_v9_cifar100.py"),
        Path("scripts/run_v9b0_cifar100_feasibility.py"),
        Path("tests/test_v9_cifar100.py"),
    ]
    missing = [str(path) for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"cannot freeze; missing V9-B artifacts: {missing}")

    training_cache = json.loads(
        (REPO_ROOT / "reports/v9b0_cifar100_train_feature_cache.json").read_text()
    )
    test_cache = json.loads(
        (REPO_ROOT / "reports/v9b0_cifar100_test-unlabeled_feature_cache.json").read_text()
    )
    feasibility = json.loads((REPO_ROOT / "reports/v9b0_cifar100_feasibility.json").read_text())
    if training_cache.get("test_labels_accessed") is not False:
        raise RuntimeError("training cache report has invalid test-label state")
    if test_cache.get("test_labels_accessed") is not False:
        raise RuntimeError("test cache was not label blind")
    if test_cache.get("split") != "test-unlabeled" or test_cache.get("sample_count") != 10_000:
        raise RuntimeError("unexpected label-blind test cache report")
    if feasibility.get("official_test_accessed") is not False:
        raise RuntimeError("B0 report has invalid test-access state")
    required = (
        feasibility["feasibility"]["oracle_beyond_selected_constant_positive"],
        feasibility["feasibility"]["hard_beyond_selected_constant_interval_positive"],
    )
    if not all(required):
        raise RuntimeError("B0 did not meet the preregistered gate for B1")
    for report, cache_name in (
        (training_cache, "train.npz"),
        (test_cache, "test-unlabeled.npz"),
    ):
        cache_path = REPO_ROOT / "data/v9_cifar100_features" / cache_name
        if report["artifact_sha256"]["output_cache"] != sha256(cache_path):
            raise RuntimeError(f"feature-cache hash mismatch: {cache_name}")

    report = {
        "schema_version": 1,
        "protocol": "v9b-cifar100-prospective-local-test-freeze",
        "frozen": True,
        "test_pixels_accessed": True,
        "test_labels_accessed": False,
        "local_seal_not_third_party_hidden_labels": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 20260914,
        "cluster_count": 20,
        "registered_rules": {
            "independent_family_opportunity": "mean > 0 and coarse-cluster interval lower > 0",
            "recoverable_hard_routing_gain": "mean > 0 and coarse-cluster interval lower > 0",
            "continuous_action_advantage": "mean > 0 and coarse-cluster interval lower > 0",
            "overall_confirmation": "opportunity and recoverable hard routing both pass",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "artifact_sha256": {str(path): sha256(REPO_ROOT / path) for path in sorted(paths)},
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
