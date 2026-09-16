#!/usr/bin/env python3
"""Freeze the V10 post-confirmation robustness audit before computing outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "reports/v10_matched_baseline_calibration_freeze.json"
RESULT_PATH = REPO_ROOT / "reports/v10_matched_baseline_calibration.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    if OUTPUT_PATH.exists() or RESULT_PATH.exists():
        raise RuntimeError("refusing to overwrite V10 freeze or result")
    v9_freeze_path = REPO_ROOT / "reports/v9b_cifar100_test_freeze.json"
    v9_confirmation_path = REPO_ROOT / "reports/v9b_cifar100_test_confirmation.json"
    v9_freeze = json.loads(v9_freeze_path.read_text())
    if v9_freeze.get("frozen") is not True:
        raise RuntimeError("invalid V9 freeze")
    for relative_path, expected in v9_freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"V9 frozen artifact changed: {relative_path}")
    v9_confirmation = json.loads(v9_confirmation_path.read_text())
    if v9_confirmation.get("test_labels_accessed") is not True:
        raise RuntimeError("V10 must be labeled post-confirmation")

    paths = [
        Path("data/v9_cifar100/cifar-100-python/test"),
        Path("data/v9_cifar100_features/train.npz"),
        Path("data/v9_cifar100_features/test-unlabeled.npz"),
        Path("docs/v10_matched_baseline_calibration_plan.md"),
        Path("reports/v9b0_cifar100_feasibility.json"),
        Path("reports/v9b_cifar100_test_freeze.json"),
        Path("reports/v9b_cifar100_test_confirmation.json"),
        Path("runs/v9_cifar100/candidate_heads.npz"),
        Path("runs/v9_cifar100/final_routers.pkl"),
        Path("scripts/evaluate_v9b_cifar100_sealed.py"),
        Path("scripts/freeze_v10_matched_baseline_calibration.py"),
        Path("scripts/run_v9b0_cifar100_feasibility.py"),
        Path("scripts/run_v10_matched_baseline_calibration.py"),
        Path("tests/test_v10_matched_baselines.py"),
    ]
    missing = [str(path) for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing V10 input: {missing}")
    report = {
        "schema_version": 1,
        "protocol": "v10-post-confirmation-robustness-freeze",
        "frozen": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "v9_test_already_accessed": True,
        "confirmatory_status": "posthoc_only",
        "modifies_v9_freeze": False,
        "artifact_sha256": {str(path): sha256(REPO_ROOT / path) for path in sorted(paths)},
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
