#!/usr/bin/env python3
"""Freeze every V3-B artifact before the one-time VOC test-label reveal."""

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
    output_path = REPO_ROOT / "reports/v3b_voc_external_freeze.json"
    if output_path.exists():
        raise SystemExit("external freeze already exists; refusing to overwrite it")
    paths = [
        Path("docs/v2_preregistration.md"),
        Path("data/voc2007_manifests/fit.jsonl"),
        Path("data/voc2007_manifests/gate.jsonl"),
        Path("data/voc2007_manifests/test_unlabeled.jsonl"),
        Path("data/voc2007_manifests/selected_relations.json"),
        Path("data/voc2007_features/fit/index.json"),
        Path("data/voc2007_features/gate/index.json"),
        Path("data/voc2007_features/test_unlabeled/index.json"),
        Path("scripts/train_candidates.py"),
        Path("scripts/train_v3b_voc_selectors.py"),
        Path("scripts/evaluate_v3b_voc_external.py"),
        Path("src/edcr/models.py"),
        Path("src/edcr/gating.py"),
        Path("src/edcr/observability.py"),
    ]
    for seed in SEEDS:
        paths.extend(
            [
                Path(f"runs/v3b_voc_candidates_seed{seed}/candidates.pt"),
                Path(f"runs/v3b_voc_candidates_seed{seed}/metrics.json"),
                Path(f"runs/v3b_voc_candidates_seed{seed}/manifest_hashes.json"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/gain_models.pt"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/tree_models.pkl"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/metrics.json"),
                Path(f"runs/v3b_voc_selectors_seed{seed}/input_hashes.json"),
            ]
        )
    missing = [str(path) for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze; missing artifacts: {missing}")
    for split in ("fit", "gate", "test_unlabeled"):
        index = json.loads(
            (REPO_ROOT / f"data/voc2007_features/{split}/index.json").read_text(
                encoding="utf-8"
            )
        )
        if index.get("complete") is not True:
            raise SystemExit(f"cannot freeze incomplete {split} feature cache")
    report = {
        "schema_version": 1,
        "protocol": "v3-B-voc-external-freeze",
        "frozen": True,
        "test_labels_accessed": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "relation_count": 8,
        "representations": ["confidence", "summary", "view_logits", "LogitMLP", "RCVI"],
        "primary_rule": (
            "checkpoint observable Spearman with test realized is positive in every seed; "
            "pooled observable Spearman exceeds pooled oracle Spearman by at least 0.20"
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
