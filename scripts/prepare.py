#!/usr/bin/env python3
"""Create authoritative COCO split manifests and fit-only relation statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.config import load_config
from edcr.data import (
    deterministic_split,
    image_labels,
    load_coco,
    mapping_from_categories,
    relation_counts,
    select_relations,
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    root_value = args.data_root or os.environ.get("EDCR_DATA_ROOT") or config["data"]["root"]
    if root_value == "REQUIRED_DATA_ROOT":
        raise SystemExit("COCO root is required via --data-root or EDCR_DATA_ROOT")
    data_root = Path(root_value).expanduser().resolve()
    annotation_path = data_root / "annotations/instances_train2017.json"
    val_annotation_path = data_root / "annotations/instances_val2017.json"
    if not annotation_path.is_file():
        raise SystemExit(f"missing annotation file: {annotation_path}")
    coco = load_coco(annotation_path)
    mapping = mapping_from_categories(coco["categories"])
    if len(mapping["names"]) != 80:
        raise SystemExit(f"expected 80 COCO categories, found {len(mapping['names'])}")
    images = sorted(coco["images"], key=lambda row: int(row["id"]))
    if args.limit is not None:
        images = images[: args.limit]
    allowed_ids = {int(row["id"]) for row in images}
    annotations = [ann for ann in coco["annotations"] if int(ann["image_id"]) in allowed_ids]
    labels = image_labels(annotations, mapping)
    val_coco = load_coco(val_annotation_path)
    val_mapping = mapping_from_categories(val_coco["categories"])
    if val_mapping["sha256"] != mapping["sha256"]:
        raise SystemExit("train/val category mappings differ")
    val_labels = image_labels(val_coco["annotations"], mapping)
    assignment = deterministic_split(
        allowed_ids,
        seed=config["data"]["split_seed"],
        proportions=config["data"]["splits"],
    )
    manifests = {name: [] for name in ("fit", "gate", "modelval", "calib")}
    for image in images:
        image_id = int(image["id"])
        split = assignment[image_id]
        relative_path = Path("train2017") / image["file_name"]
        manifests[split].append(
            {
                "sample_id": f"coco_train2017_{image_id}",
                "source_image_id": image_id,
                "split": split,
                "image_path": str(relative_path),
                "labels": labels.get(image_id, []),
                "label_format": "positive_contiguous_ids",
                "parent_sha256": None,
            }
        )
    test_manifest = [
        {
            "sample_id": f"coco_val2017_{int(image['id'])}",
            "source_image_id": int(image["id"]),
            "split": "test",
            "image_path": str(Path("val2017") / image["file_name"]),
            "labels": val_labels.get(int(image["id"]), []),
            "label_format": "positive_contiguous_ids",
            "parent_sha256": None,
        }
        for image in sorted(val_coco["images"], key=lambda row: int(row["id"]))
    ]
    fit_sets = [set(row["labels"]) for row in manifests["fit"]]
    relations = relation_counts(fit_sets, len(mapping["names"]))
    thresholds = {"n11": 200, "n10": 50, "n01": 200, "n00": 200}
    eligible = [
        row for row in relations if all(int(row[key]) >= value for key, value in thresholds.items())
    ]
    eligible.sort(
        key=lambda row: (-float(row["smoothed_lift"]), row["target_id"], row["source_id"])
    )
    selected = select_relations(eligible, mapping["names"], config["data"]["pair_count"])

    for relation in selected:
        source = int(relation["source_id"])
        target = int(relation["target_id"])
        support: dict[str, dict[str, int]] = {}
        for split_name, manifest_rows in manifests.items():
            split_sets = [set(row["labels"]) for row in manifest_rows]
            n11 = sum(source in labels and target in labels for labels in split_sets)
            n10 = sum(source not in labels and target in labels for labels in split_sets)
            n01 = sum(source in labels and target not in labels for labels in split_sets)
            support[split_name] = {
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": len(split_sets) - n11 - n10 - n01,
            }
        relation["support_by_split"] = support
    if args.dry_run:
        print(
            json.dumps(
                {
                    "image_counts": {key: len(value) for key, value in manifests.items()},
                    "test_image_count": len(test_manifest),
                    "mapping_sha256": mapping["sha256"],
                    "eligible_relation_count": len(eligible),
                    "selected_relations": selected,
                },
                indent=2,
            )
        )
        return 0
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "category_mapping.json").write_text(
        json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "selected_relations.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, rows in manifests.items():
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "test.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_manifest:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "fit_relation_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(relations[0]))
        writer.writeheader()
        writer.writerows(relations)
    metadata = {
        "schema_version": 1,
        "source_annotation_sha256": sha256(annotation_path),
        "test_annotation_sha256": sha256(val_annotation_path),
        "split_seed": config["data"]["split_seed"],
        "split_algorithm": "sha256-rank-largest-remainder-v1",
        "image_counts": {key: len(value) for key, value in manifests.items()},
        "test_image_count": len(test_manifest),
        "mapping_sha256": mapping["sha256"],
        "eligible_relation_count": len(eligible),
        "selected_relation_count": len(selected),
        "selected_pair_ids": [row["pair_id"] for row in selected],
        "thresholds": thresholds,
        "limited": args.limit is not None,
    }
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
