#!/usr/bin/env python3
"""Prepare label-omitting V6 training image manifest from Open Images validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = (
    REPO_ROOT
    / "data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"
)
PANEL_PATH = REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json"
OUTPUT_ROOT = REPO_ROOT / "data/openimages_v7_manifests"
REPORT_PATH = REPO_ROOT / "reports/v6_openimages_training_preparation.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    panel = json.loads(PANEL_PATH.read_text())
    target_mids = {row["target_openimages_mid"] for row in panel}
    values: dict[tuple[str, str], int] = {}
    with ANNOTATIONS.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mid = row["LabelName"]
            if mid not in target_mids:
                continue
            value = int(float(row["Confidence"]))
            if value not in (0, 1):
                raise RuntimeError(f"non-binary confidence: {row['Confidence']}")
            key = (row["ImageID"], mid)
            if key in values and values[key] != value:
                raise RuntimeError(f"conflicting duplicate annotation: {key}")
            values[key] = value

    support = {
        mid: {
            "negative": sum(value == 0 for (image_id, label), value in values.items() if label == mid),
            "positive": sum(value == 1 for (image_id, label), value in values.items() if label == mid),
        }
        for mid in target_mids
    }
    for relation in panel:
        mid = relation["target_openimages_mid"]
        observed = support[mid]
        if observed["negative"] != relation["validation_verified_negative"]:
            raise RuntimeError(f"negative support drift for {relation['pair_id']}")
        if observed["positive"] != relation["validation_verified_positive"]:
            raise RuntimeError(f"positive support drift for {relation['pair_id']}")

    image_ids = sorted({image_id for image_id, _mid in values})
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "v6_training_unlabeled.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"oi_v7_val_{image_id}",
                    "image_path": f"validation/{image_id}.jpg",
                    "labels": [],
                    "label_format": "verified_targets_omitted_from_feature_manifest",
                    "split": "validation_training",
                },
                sort_keys=True,
            )
            + "\n"
            for image_id in image_ids
        )
    )
    download_list = OUTPUT_ROOT / "v6_training_download_list.txt"
    download_list.write_text("".join(f"validation/{image_id}\n" for image_id in image_ids))
    report = {
        "schema_version": 1,
        "protocol": "v6-openimages-validation-training-preparation",
        "annotation_sha256": sha256(ANNOTATIONS),
        "relation_panel_sha256": sha256(PANEL_PATH),
        "relation_count": len(panel),
        "image_count": len(image_ids),
        "verified_image_target_pairs": len(values),
        "manifest_sha256": sha256(manifest_path),
        "download_list_sha256": sha256(download_list),
        "support": support,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
