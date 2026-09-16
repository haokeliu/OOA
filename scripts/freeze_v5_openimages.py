#!/usr/bin/env python3
"""Freeze the V5 protocol before Open Images validation annotation access."""

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
    output_path = REPO_ROOT / "reports/v5_openimages_freeze.json"
    annotation_path = (
        REPO_ROOT
        / "data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"
    )
    if output_path.exists():
        raise SystemExit("V5 freeze already exists; refusing to overwrite it")
    if annotation_path.exists():
        raise SystemExit("refusing to freeze after Open Images annotations appeared")

    paths = [
        Path("configs/pilot.yaml"),
        Path("data/checkpoints/ViT-B-16.pt"),
        Path("data/manifests/v3_relation_panel.json"),
        Path("data/manifests/v5_openimages_class_mapping.json"),
        Path("data/openimages_v7_metadata/oidv7-class-descriptions.csv"),
        Path("data/openimages_v7_metadata/oidv7-class-descriptions-boxable.csv"),
        Path("docs/v5_openimages_preregistration.md"),
        Path("scripts/audit_observable_opportunity.py"),
        Path("scripts/audit_openimages_mapping.py"),
        Path("scripts/cache_features.py"),
        Path("scripts/evaluate_v5_openimages.py"),
        Path("scripts/prepare_v5_openimages.py"),
        Path("scripts/run_v4b_action_grid.py"),
        Path("scripts/run_v4d_soft_mechanism_ablation.py"),
        Path("scripts/train_gate.py"),
        Path("scripts/train_v2b_gain_models.py"),
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
            ]
        )
    missing = [str(path) for path in paths if not (REPO_ROOT / path).is_file()]
    if missing:
        raise SystemExit(f"cannot freeze; missing V5 artifacts: {missing}")

    mapping = json.loads(
        (REPO_ROOT / "data/manifests/v5_openimages_class_mapping.json").read_text()
    )
    if mapping.get("annotation_files_accessed") is not False:
        raise SystemExit("ontology mapping does not certify label-blind construction")
    if mapping.get("n_relations") != 30 or not mapping.get("all_mapped_classes_boxable"):
        raise SystemExit("ontology mapping failed the preregistered panel audit")

    report = {
        "schema_version": 1,
        "protocol": "v5-openimages-cross-domain-freeze",
        "frozen": True,
        "annotations_accessed": False,
        "validation_pixels_accessed": False,
        "class_metadata_accessed": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": list(SEEDS),
        "relation_count": 30,
        "representations": ["confidence", "summary", "view_logits"],
        "minimum_positive_and_negative_per_half": 30,
        "minimum_eligible_relations": 8,
        "primary_rule": (
            "mean evaluation soft-minus-probability-hard is positive and its 20,000-replicate "
            "relation-cluster 95% percentile interval excludes zero"
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
