#!/usr/bin/env python3
"""Freeze natural negative-context and COCO-small weak-target groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.config import load_config
from edcr.data import load_coco, maximum_noncrowd_area_ratios, maximum_noncrowd_areas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-root", default="data/coco2017")
    parser.add_argument("--splits", nargs="+", default=["gate", "modelval"])
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    mapping = json.loads(
        (REPO_ROOT / "data/manifests/category_mapping.json").read_text(encoding="utf-8")
    )
    relations = json.loads(
        (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
    )
    annotations_dir = REPO_ROOT / args.data_root / "annotations"
    coco_by_source = {
        "train": load_coco(annotations_dir / "instances_train2017.json"),
        "test": load_coco(annotations_dir / "instances_val2017.json"),
    }
    maximum_areas_by_source = {
        source: maximum_noncrowd_areas(coco["annotations"], mapping)
        for source, coco in coco_by_source.items()
    }
    maximum_area_ratios_by_source = {
        source: maximum_noncrowd_area_ratios(coco["annotations"], coco["images"], mapping)
        for source, coco in coco_by_source.items()
    }
    threshold = float(config["evaluation"]["weak_small_area_pixels"])
    ratio_threshold = float(config["evaluation"]["weak_max_area_ratio"])
    output_dir = REPO_ROOT / "data/groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    for split in args.splits:
        annotation_source = "test" if split == "test" else "train"
        maximum_areas = maximum_areas_by_source[annotation_source]
        maximum_area_ratios = maximum_area_ratios_by_source[annotation_source]
        rows = [
            json.loads(line)
            for line in (REPO_ROOT / f"data/manifests/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        weak = np.zeros((len(rows), len(relations)), dtype=bool)
        weak_coco_small = np.zeros_like(weak)
        negative = np.zeros_like(weak)
        for row_index, row in enumerate(rows):
            labels = set(row["labels"])
            image_id = int(row["source_image_id"])
            for relation_index, relation in enumerate(relations):
                source = int(relation["source_id"])
                target = int(relation["target_id"])
                negative[row_index, relation_index] = source in labels and target not in labels
                area_ratio = maximum_area_ratios.get((image_id, target))
                weak[row_index, relation_index] = (
                    target in labels and area_ratio is not None and area_ratio < ratio_threshold
                )
                area = maximum_areas.get((image_id, target))
                weak_coco_small[row_index, relation_index] = (
                    target in labels and area is not None and area < threshold
                )
        sample_ids = np.asarray([row["sample_id"] for row in rows])
        np.savez_compressed(
            output_dir / f"natural_{split}.npz",
            sample_id=sample_ids,
            weak=weak,
            weak_coco_small=weak_coco_small,
            negative=negative,
            weak_area_threshold=np.asarray(threshold),
            weak_area_ratio_threshold=np.asarray(ratio_threshold),
        )
        ratio_support = {}
        for diagnostic_ratio in (0.01, 0.02, 0.04):
            ratio_support[str(diagnostic_ratio)] = {
                relation["pair_id"]: sum(
                    int(relation["target_id"]) in set(row["labels"])
                    and maximum_area_ratios.get(
                        (int(row["source_image_id"]), int(relation["target_id"])), 1.0
                    )
                    < diagnostic_ratio
                    for row in rows
                )
                for relation in relations
            }
        summaries[split] = {
            "sample_count": len(rows),
            "per_relation": {
                relation["pair_id"]: {
                    "negative_count": int(negative[:, index].sum()),
                    "weak_count": int(weak[:, index].sum()),
                    "coco_small_count": int(weak_coco_small[:, index].sum()),
                }
                for index, relation in enumerate(relations)
            },
            "relative_area_support_diagnostic": ratio_support,
        }
    report_path = REPO_ROOT / "reports/p3_natural_group_support.json"
    prior_splits: dict[str, object] = {}
    if report_path.is_file():
        prior_report = json.loads(report_path.read_text(encoding="utf-8"))
        prior_splits = prior_report.get("splits", {})
    report = {
        "schema_version": 1,
        "definition": "target-positive and largest non-crowd target instance < 2% of image area",
        "weak_area_threshold": threshold,
        "weak_area_ratio_threshold": ratio_threshold,
        "splits": {**prior_splits, **summaries},
    }
    if ratio_threshold != float(config["evaluation"]["weak_max_area_ratio"]):
        raise AssertionError("primary weak-area threshold changed during support diagnostics")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
