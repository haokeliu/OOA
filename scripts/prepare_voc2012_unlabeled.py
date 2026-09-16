#!/usr/bin/env python3
"""Prepare a label-blind VOC 2012 validation manifest for V3-C."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VOC2007_ROOT = REPO_ROOT / "data/voc2007/VOCdevkit/VOC2007"
VOC2012_ROOT = REPO_ROOT / "data/voc2012/VOCdevkit/VOC2012"
OUTPUT_ROOT = REPO_ROOT / "data/voc2012_manifests"
EXPECTED_ARCHIVE_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"


def digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    archive = REPO_ROOT / "data/voc2012_downloads/VOCtrainval_11-May-2012.tar"
    archive_md5 = digest(archive, "md5")
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise SystemExit(f"VOC 2012 archive MD5 mismatch: {archive_md5}")
    voc2007_paths = {
        row["sample_id"]: VOC2007_ROOT / row["image_path"]
        for split in ("fit", "gate", "test_unlabeled")
        for row in (
            json.loads(line)
            for line in (
                REPO_ROOT / f"data/voc2007_manifests/{split}.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        )
    }
    voc2007_by_digest: dict[str, list[str]] = {}
    for sample_id, path in voc2007_paths.items():
        voc2007_by_digest.setdefault(digest(path), []).append(sample_id)

    validation_ids = (
        VOC2012_ROOT / "ImageSets/Main/val.txt"
    ).read_text(encoding="utf-8").splitlines()
    retained = []
    excluded = []
    for image_id in validation_ids:
        image_path = VOC2012_ROOT / "JPEGImages" / f"{image_id}.jpg"
        image_digest = digest(image_path)
        if image_digest in voc2007_by_digest:
            excluded.append(
                {
                    "voc2012_id": image_id,
                    "jpeg_sha256": image_digest,
                    "matching_voc2007_ids": voc2007_by_digest[image_digest],
                }
            )
            continue
        retained.append(
            {
                "sample_id": f"voc2012_{image_id}",
                "image_path": f"JPEGImages/{image_id}.jpg",
                "labels": [],
                "label_format": "sealed_until_v3c_replication",
                "split": "val_unlabeled",
                "jpeg_sha256": image_digest,
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "val_unlabeled.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in retained),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "protocol": "v3-C-label-blind-preparation",
        "annotation_files_accessed": False,
        "archive_md5": archive_md5,
        "archive_sha256": digest(archive),
        "official_val_count": len(validation_ids),
        "retained_count": len(retained),
        "jpeg_overlap_excluded_count": len(excluded),
        "excluded": excluded,
    }
    (REPO_ROOT / "reports/v3c_voc2012_preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
