#!/usr/bin/env python3
"""Freeze V6 selectors and protocol before Open Images test access."""

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
    output_path = REPO_ROOT / "reports/v6_openimages_test_freeze.json"
    test_annotations = (
        REPO_ROOT / "data/openimages_v7/oidv7-test-annotations-human-imagelabels.csv"
    )
    test_pixel_root = REPO_ROOT / "data/openimages_v7_pixels/test"
    if output_path.exists():
        raise SystemExit("V6 test freeze already exists; refusing to overwrite it")
    if test_annotations.exists():
        raise SystemExit("refusing to freeze after Open Images test annotations appeared")
    if test_pixel_root.exists() and any(test_pixel_root.iterdir()):
        raise SystemExit("refusing to freeze after Open Images test pixels appeared")

    paths = [
        Path("configs/pilot.yaml"),
        Path("data/checkpoints/ViT-B-16.pt"),
        Path("data/manifests/v3_relation_panel.json"),
        Path("data/manifests/v5_openimages_class_mapping.json"),
        Path("data/manifests/v6_openimages_relation_panel.json"),
        Path("data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"),
        Path("data/openimages_v7_features/v6_training/index.json"),
        Path("data/openimages_v7_manifests/v6_training_unlabeled.jsonl"),
        Path("docs/v6_openimages_test_preregistration.md"),
        Path("reports/v5_openimages_freeze.json"),
        Path("reports/v5_openimages_preparation.json"),
        Path("reports/v6_openimages_training_feature_cache.json"),
        Path("reports/v6_openimages_training_preparation.json"),
        Path("scripts/audit_observable_opportunity.py"),
        Path("scripts/cache_features.py"),
        Path("scripts/download_openimages_subset.py"),
        Path("scripts/evaluate_v6_openimages_test.py"),
        Path("scripts/prepare_v6_openimages_training.py"),
        Path("scripts/prepare_v6_openimages_test.py"),
        Path("scripts/run_v4b_action_grid.py"),
        Path("scripts/run_v4d_soft_mechanism_ablation.py"),
        Path("scripts/select_v6_openimages_panel.py"),
        Path("scripts/train_gate.py"),
        Path("scripts/train_v2b_gain_models.py"),
        Path("scripts/train_v6_openimages_selectors.py"),
        Path("src/edcr/cache.py"),
        Path("src/edcr/config.py"),
        Path("src/edcr/features.py"),
        Path("src/edcr/gating.py"),
        Path("src/edcr/models.py"),
    ]
    for seed in SEEDS:
        paths.extend(
            [
                Path(f"runs/v3a_candidates_seed{seed}/candidates.pt"),
                Path(f"runs/v3a_candidates_seed{seed}/metrics.json"),
                Path(f"runs/v6_openimages_selectors_seed{seed}/input_hashes.json"),
                Path(f"runs/v6_openimages_selectors_seed{seed}/metrics.json"),
                Path(f"runs/v6_openimages_selectors_seed{seed}/models.pkl"),
            ]
        )
    paths.extend(
        sorted(
            path.relative_to(REPO_ROOT)
            for path in (
                REPO_ROOT / "data/openimages_v7_features/v6_training"
            ).glob("shard_*.npz")
        )
    )
    missing = [str(path) for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze; missing V6 artifacts: {missing}")
    cache_index = json.loads(
        (REPO_ROOT / "data/openimages_v7_features/v6_training/index.json").read_text()
    )
    if cache_index.get("complete") is not True:
        raise SystemExit("cannot freeze an incomplete V6 training feature cache")
    shard_count = sum(
        path.name.startswith("shard_") and path.suffix == ".npz"
        for path in paths
    )
    if shard_count != cache_index.get("shard_count"):
        raise SystemExit(
            f"training cache shard-count mismatch: {shard_count} versus "
            f"{cache_index.get('shard_count')}"
        )
    panel = json.loads(
        (REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json").read_text()
    )
    if len(panel) != 19:
        raise SystemExit(f"unexpected V6 relation count: {len(panel)}")

    report = {
        "schema_version": 1,
        "protocol": "v6-openimages-prospective-test-freeze",
        "frozen": True,
        "test_annotations_accessed": False,
        "test_pixels_accessed": False,
        "validation_labels_used_for_training": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "relation_count": len(panel),
        "representations": ["confidence", "summary", "view_logits"],
        "task_count": len(panel) * 3 * len(SEEDS),
        "primary_rule": (
            "mean Open Images test soft-minus-probability-hard is positive and its "
            "20,000-replicate relation-cluster 95% percentile interval excludes zero"
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
