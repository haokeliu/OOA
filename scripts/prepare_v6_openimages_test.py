#!/usr/bin/env python3
"""Prepare the fixed V6 Open Images test subset after protocol freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = REPO_ROOT / "reports/v6_openimages_test_freeze.json"
PANEL_PATH = REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data/openimages_v7/oidv7-test-annotations-human-imagelabels.csv"
)
OUTPUT_ROOT = REPO_ROOT / "data/openimages_v7_manifests"
REPORT_PATH = REPO_ROOT / "reports/v6_openimages_test_preparation.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("frozen") is not True or freeze.get("test_annotations_accessed") is not False:
        raise RuntimeError("invalid V6 freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    args = parser.parse_args()
    verify_freeze()
    panel = json.loads(PANEL_PATH.read_text())
    target_mids = {row["target_openimages_mid"] for row in panel}
    values: dict[tuple[str, str], int] = {}
    with args.annotations.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "LabelName", "Confidence"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"unexpected annotation header: {reader.fieldnames}")
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

    relation_support = []
    for relation in panel:
        mid = relation["target_openimages_mid"]
        observed = [value for (image_id, label), value in values.items() if label == mid]
        if not observed:
            raise RuntimeError(f"no explicit test labels for {relation['pair_id']}")
        relation_support.append(
            {
                "pair_id": relation["pair_id"],
                "target_openimages_mid": mid,
                "negative": observed.count(0),
                "positive": observed.count(1),
                "total": len(observed),
            }
        )

    image_ids = sorted({image_id for image_id, _mid in values})
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "v6_test_unlabeled.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"oi_v7_test_{image_id}",
                    "image_path": f"test/{image_id}.jpg",
                    "labels": [],
                    "label_format": "sealed_verified_targets_omitted",
                    "split": "test",
                },
                sort_keys=True,
            )
            + "\n"
            for image_id in image_ids
        )
    )
    download_list = OUTPUT_ROOT / "v6_test_download_list.txt"
    download_list.write_text("".join(f"test/{image_id}\n" for image_id in image_ids))
    report = {
        "schema_version": 1,
        "protocol": "v6-openimages-test-preparation",
        "test_annotations_accessed": True,
        "model_outcomes_computed": False,
        "annotation_sha256": sha256(args.annotations),
        "relation_panel_sha256": sha256(PANEL_PATH),
        "relation_count": len(panel),
        "image_count": len(image_ids),
        "verified_image_target_pairs": len(values),
        "manifest_sha256": sha256(manifest_path),
        "download_list_sha256": sha256(download_list),
        "relation_support": relation_support,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
