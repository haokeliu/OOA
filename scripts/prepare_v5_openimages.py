#!/usr/bin/env python3
"""Apply the frozen V5 support rule without exposing per-image label values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "reports/v5_openimages_freeze.json"
MAPPING_PATH = REPO_ROOT / "data/manifests/v5_openimages_class_mapping.json"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT
    / "data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"
)
OUTPUT_ROOT = REPO_ROOT / "data/openimages_v7_manifests"
REPORT_PATH = REPO_ROOT / "reports/v5_openimages_preparation.json"
MINIMUM_CLASS_COUNT = 30
MINIMUM_ELIGIBLE_RELATIONS = 8


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def partition(image_id: str) -> str:
    digest = hashlib.sha256(f"v5-oi:{image_id}".encode()).digest()
    return "adaptation" if int.from_bytes(digest[:8], "big") % 2 == 0 else "evaluation"


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("frozen") is not True or freeze.get("annotations_accessed") is not False:
        raise RuntimeError("invalid V5 freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()
    verify_freeze()

    mapping = json.loads(MAPPING_PATH.read_text())
    target_to_relation = {
        row["target_openimages_mid"]: row for row in mapping["relation_mapping"]
    }
    if len(target_to_relation) != mapping["n_relations"]:
        raise RuntimeError("V5 requires one unique target MID per relation")

    values: dict[tuple[str, str], int] = {}
    relevant_rows = 0
    with args.annotations.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "LabelName", "Confidence"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"unexpected annotation header: {reader.fieldnames}")
        for row in reader:
            mid = row["LabelName"]
            if mid not in target_to_relation:
                continue
            confidence = float(row["Confidence"])
            if confidence not in (0.0, 1.0):
                raise RuntimeError(f"non-binary confidence: {confidence}")
            key = (row["ImageID"], mid)
            value = int(confidence)
            if key in values and values[key] != value:
                raise RuntimeError(f"conflicting duplicate annotation: {key}")
            values[key] = value
            relevant_rows += 1

    counts: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: {
            "adaptation": {0: 0, 1: 0},
            "evaluation": {0: 0, 1: 0},
        }
    )
    for (image_id, mid), value in values.items():
        counts[mid][partition(image_id)][value] += 1

    relation_support = []
    eligible_mids = set()
    for mid, relation in target_to_relation.items():
        support = counts[mid]
        eligible = all(
            support[split][value] >= MINIMUM_CLASS_COUNT
            for split in ("adaptation", "evaluation")
            for value in (0, 1)
        )
        if eligible:
            eligible_mids.add(mid)
        relation_support.append(
            {
                "pair_id": relation["pair_id"],
                "target_coco_name": relation["target_coco_name"],
                "target_openimages_name": relation["target_openimages_name"],
                "target_openimages_mid": mid,
                "mapping_type": next(
                    row["mapping_type"]
                    for row in mapping["class_mapping"]
                    if row["openimages_mid"] == mid
                ),
                "adaptation_negative": support["adaptation"][0],
                "adaptation_positive": support["adaptation"][1],
                "evaluation_negative": support["evaluation"][0],
                "evaluation_positive": support["evaluation"][1],
                "eligible": eligible,
            }
        )

    eligible_image_ids = sorted(
        {
            image_id
            for image_id, mid in values
            if mid in eligible_mids
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "eligible_unlabeled.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"oi_v7_val_{image_id}",
                    "image_path": f"validation/{image_id}.jpg",
                    "labels": [],
                    "label_format": "sealed_verified_targets_omitted",
                    "split": partition(image_id),
                },
                sort_keys=True,
            )
            + "\n"
            for image_id in eligible_image_ids
        )
    )
    download_list_path = OUTPUT_ROOT / "eligible_download_list.txt"
    download_list_path.write_text(
        "".join(f"validation/{image_id}\n" for image_id in eligible_image_ids)
    )

    eligible_relations = sum(row["eligible"] for row in relation_support)
    report = {
        "schema_version": 1,
        "protocol": "v5-openimages-support-preparation",
        "annotations_accessed": True,
        "model_outcomes_computed": False,
        "annotation_sha256": sha256(args.annotations),
        "relevant_annotation_rows": relevant_rows,
        "unique_relevant_image_mid_pairs": len(values),
        "minimum_positive_and_negative_per_half": MINIMUM_CLASS_COUNT,
        "minimum_eligible_relations": MINIMUM_ELIGIBLE_RELATIONS,
        "eligible_relation_count": eligible_relations,
        "feasibility_passed": eligible_relations >= MINIMUM_ELIGIBLE_RELATIONS,
        "eligible_image_count": len(eligible_image_ids),
        "manifest_sha256": sha256(manifest_path),
        "download_list_sha256": sha256(download_list_path),
        "relation_support": relation_support,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["feasibility_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
