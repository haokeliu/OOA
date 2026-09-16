#!/usr/bin/env python3
"""Freeze V3-C before the one-time VOC 2012 validation-label reveal."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    output_path = REPO_ROOT / "reports/v3c_voc2012_external_freeze.json"
    if output_path.exists():
        raise SystemExit("V3-C external freeze already exists; refusing to overwrite it")
    paths = [
        Path("configs/pilot.yaml"),
        Path("docs/v3c_preregistration.md"),
        Path("data/voc2012_downloads/VOCtrainval_11-May-2012.tar"),
        Path("data/voc2012_manifests/val_unlabeled.jsonl"),
        Path("data/voc2012_features/val_unlabeled/index.json"),
        Path("reports/v3c_voc2012_preparation.json"),
        Path("reports/v3b_voc_external_freeze.json"),
        Path("scripts/evaluate_v3c_voc2012_external.py"),
        Path("scripts/evaluate_v3b_voc_external.py"),
        Path("scripts/audit_observable_opportunity.py"),
        Path("scripts/train_gate.py"),
        Path("scripts/train_v2b_gain_models.py"),
        Path("scripts/prepare_voc2007.py"),
        Path("src/edcr/cache.py"),
        Path("src/edcr/config.py"),
        Path("src/edcr/gating.py"),
        Path("src/edcr/metrics.py"),
        Path("src/edcr/models.py"),
        Path("src/edcr/observability.py"),
    ]
    for seed in SEEDS:
        paths.extend(
            [
                Path(f"runs/v3b_voc_candidates_seed{seed}/candidates.pt"),
                Path(f"runs/v3b_voc_candidates_seed{seed}/metrics.json"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/gain_models.pt"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/tree_models.pkl"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/metrics.json"),
            ]
        )
    missing = [str(path) for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze; missing V3-C artifacts: {missing}")
    index = json.loads(
        (REPO_ROOT / "data/voc2012_features/val_unlabeled/index.json").read_text(
            encoding="utf-8"
        )
    )
    if index.get("complete") is not True:
        raise SystemExit("cannot freeze incomplete VOC 2012 feature cache")
    report = {
        "schema_version": 1,
        "protocol": "v3-C-voc2012-external-freeze",
        "frozen": True,
        "labels_accessed": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "relation_count": 8,
        "representations": ["confidence", "summary", "view_logits", "LogitMLP", "RCVI"],
        "primary_rule": (
            "VOC2007 checkpoint observable Spearman with VOC2012 realized is positive in every "
            "seed; pooled observable Spearman exceeds pooled oracle Spearman by at least 0.20"
        ),
        "artifact_sha256": {
            str(path): sha256(REPO_ROOT / path) for path in sorted(paths)
        },
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
