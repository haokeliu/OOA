#!/usr/bin/env python3
"""Audit a label-blind COCO-to-Open-Images class mapping for V5.

This script intentionally reads class-description metadata only.  It must not
read Open Images validation annotations or images.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIONS_PATH = ROOT / "data/manifests/v3_relation_panel.json"
ALL_CLASSES_PATH = (
    ROOT / "data/openimages_v7_metadata/oidv7-class-descriptions.csv"
)
BOXABLE_CLASSES_PATH = (
    ROOT / "data/openimages_v7_metadata/oidv7-class-descriptions-boxable.csv"
)
OUTPUT_PATH = ROOT / "data/manifests/v5_openimages_class_mapping.json"

# Semantic aliases are fixed from public ontology names, without consulting
# validation labels or class frequencies.  In particular, COCO "mouse" means
# the computer peripheral, not the animal.
ALIASES = {
    "keyboard": "Computer keyboard",
    "microwave": "Microwave oven",
    "mouse": "Computer mouse",
    "orange": "Orange (fruit)",
    "potted plant": "Houseplant",
    "remote": "Remote control",
    "skis": "Ski",
    "sports ball": "Ball (Object)",
    "tv": "Television",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_classes(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    by_name: dict[str, str] = {}
    for mid, name in rows:
        if name in by_name:
            raise ValueError(f"duplicate Open Images display name: {name}")
        by_name[name] = mid
    return by_name


def main() -> None:
    relations = json.loads(RELATIONS_PATH.read_text())
    all_classes = read_classes(ALL_CLASSES_PATH)
    boxable_classes = read_classes(BOXABLE_CLASSES_PATH)
    all_names_casefolded = {name.casefold(): name for name in all_classes}

    coco_names = sorted(
        {relation[key] for relation in relations for key in ("source_name", "target_name")}
    )
    class_mapping = []
    for coco_name in coco_names:
        oi_name = ALIASES.get(coco_name)
        if oi_name is None:
            oi_name = all_names_casefolded.get(coco_name.casefold(), "")
        if oi_name not in all_classes:
            raise ValueError(f"unmapped COCO class: {coco_name} -> {oi_name}")
        if oi_name not in boxable_classes:
            raise ValueError(f"mapped class is not boxable: {coco_name} -> {oi_name}")
        if all_classes[oi_name] != boxable_classes[oi_name]:
            raise ValueError(f"MID mismatch for {oi_name}")
        class_mapping.append(
            {
                "coco_name": coco_name,
                "openimages_name": oi_name,
                "openimages_mid": all_classes[oi_name],
                "mapping_type": "semantic_alias" if coco_name in ALIASES else "exact_name",
            }
        )

    by_coco = {row["coco_name"]: row for row in class_mapping}
    relation_mapping = []
    for relation in relations:
        source = by_coco[relation["source_name"]]
        target = by_coco[relation["target_name"]]
        relation_mapping.append(
            {
                "pair_id": relation["pair_id"],
                "source_coco_id": relation["source_id"],
                "source_coco_name": relation["source_name"],
                "source_openimages_mid": source["openimages_mid"],
                "source_openimages_name": source["openimages_name"],
                "target_coco_id": relation["target_id"],
                "target_coco_name": relation["target_name"],
                "target_openimages_mid": target["openimages_mid"],
                "target_openimages_name": target["openimages_name"],
            }
        )

    output = {
        "schema_version": 1,
        "protocol": "v5-openimages-label-blind-ontology-audit",
        "annotation_files_accessed": False,
        "mapping_basis": "public class descriptions and COCO class semantics only",
        "n_relations": len(relation_mapping),
        "n_unique_classes": len(class_mapping),
        "n_semantic_aliases": len(ALIASES),
        "all_mapped_classes_boxable": True,
        "metadata_sha256": {
            str(ALL_CLASSES_PATH.relative_to(ROOT)): sha256(ALL_CLASSES_PATH),
            str(BOXABLE_CLASSES_PATH.relative_to(ROOT)): sha256(BOXABLE_CLASSES_PATH),
            str(RELATIONS_PATH.relative_to(ROOT)): sha256(RELATIONS_PATH),
        },
        "class_mapping": class_mapping,
        "relation_mapping": relation_mapping,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")
    print(
        f"mapped {len(class_mapping)} classes and {len(relation_mapping)} relations; "
        f"aliases={len(ALIASES)}; all_boxable=true"
    )


if __name__ == "__main__":
    main()
