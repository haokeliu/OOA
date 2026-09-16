#!/usr/bin/env python3
"""Prepare trainval-only VOC manifests and a label-blind test image manifest."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VOC_ROOT = REPO_ROOT / "data/voc2007/VOCdevkit/VOC2007"
OUTPUT_ROOT = REPO_ROOT / "data/voc2007_manifests"
CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
PANEL_SIZE = 8


def split_name(image_id: str) -> str:
    residue = int.from_bytes(
        hashlib.sha256(f"voc2007:{image_id}".encode()).digest()[:8]
    ) % 10
    return "fit" if residue < 7 else "gate"


def labels_from_trainval_xml(image_id: str) -> list[int]:
    root = ET.parse(VOC_ROOT / "Annotations" / f"{image_id}.xml").getroot()
    names = {node.text for node in root.findall("object/name")}
    return [index for index, name in enumerate(CLASS_NAMES) if name in names]


def relation_counts(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    total = len(rows)
    label_sets = [set(row["labels"]) for row in rows]
    for source in range(len(CLASS_NAMES)):
        for target in range(len(CLASS_NAMES)):
            if source == target:
                continue
            n11 = sum(source in labels and target in labels for labels in label_sets)
            n10 = sum(source in labels and target not in labels for labels in label_sets)
            n01 = sum(source not in labels and target in labels for labels in label_sets)
            n00 = total - n11 - n10 - n01
            source_count = n11 + n10
            target_count = n11 + n01
            lift = ((n11 + 1) * (total + 2)) / ((source_count + 1) * (target_count + 1))
            output.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "source_name": CLASS_NAMES[source],
                    "target_name": CLASS_NAMES[target],
                    "pair_id": f"{CLASS_NAMES[source]}_to_{CLASS_NAMES[target]}",
                    "n11": n11,
                    "n10": n10,
                    "n01": n01,
                    "n00": n00,
                    "smoothed_lift": lift,
                }
            )
    return output


def select_relations(counts: list[dict[str, object]]) -> list[dict[str, object]]:
    eligible = [
        row
        for row in counts
        if row["n11"] >= 20
        and row["n10"] >= 20
        and row["n01"] >= 20
        and row["n00"] >= 200
    ]
    eligible.sort(
        key=lambda row: (
            -row["smoothed_lift"],
            -row["n11"],
            row["source_id"],
            row["target_id"],
        )
    )
    target_ids: set[int] = set()
    unordered_pairs: set[tuple[int, int]] = set()
    selected = []
    for row in eligible:
        unordered = tuple(sorted((row["source_id"], row["target_id"])))
        if row["target_id"] in target_ids or unordered in unordered_pairs:
            continue
        selected.append(row)
        target_ids.add(row["target_id"])
        unordered_pairs.add(unordered)
        if len(selected) == PANEL_SIZE:
            break
    if len(selected) != PANEL_SIZE:
        raise RuntimeError(f"only {len(selected)} VOC relations satisfy the frozen rule")
    return selected


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trainval_ids = (
        VOC_ROOT / "ImageSets/Main/trainval.txt"
    ).read_text(encoding="utf-8").splitlines()
    splits = {"fit": [], "gate": []}
    for image_id in trainval_ids:
        split = split_name(image_id)
        splits[split].append(
            {
                "sample_id": f"voc2007_{image_id}",
                "image_path": f"JPEGImages/{image_id}.jpg",
                "labels": labels_from_trainval_xml(image_id),
                "label_format": "positive_contiguous_ids",
                "split": split,
            }
        )
    for split, rows in splits.items():
        write_jsonl(OUTPUT_ROOT / f"{split}.jsonl", rows)
    test_ids = (VOC_ROOT / "ImageSets/Main/test.txt").read_text(encoding="utf-8").splitlines()
    test_rows = [
        {
            "sample_id": f"voc2007_{image_id}",
            "image_path": f"JPEGImages/{image_id}.jpg",
            "labels": [],
            "label_format": "sealed_until_v3b_confirmation",
            "split": "test_unlabeled",
        }
        for image_id in test_ids
    ]
    write_jsonl(OUTPUT_ROOT / "test_unlabeled.jsonl", test_rows)
    counts = relation_counts(splits["fit"])
    relations = select_relations(counts)
    (OUTPUT_ROOT / "selected_relations.json").write_text(
        json.dumps(relations, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "protocol": "v3-B",
        "test_labels_accessed": False,
        "class_names": list(CLASS_NAMES),
        "fit_count": len(splits["fit"]),
        "gate_count": len(splits["gate"]),
        "test_image_count": len(test_rows),
        "split_rule": "sha256('voc2007:<image_id>') modulo 10; residues 0-6 fit",
        "relation_rule": "first 8 eligible by lift,n11,ids; unique targets/unordered pairs",
        "relations": relations,
    }
    (REPO_ROOT / "reports/v3b_voc_preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
