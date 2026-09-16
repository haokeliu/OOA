#!/usr/bin/env python3
"""Select leakage-safe COCO intervention candidates; image editing follows review."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.config import load_config
from edcr.data import load_coco
from edcr.interventions import decode_union, stable_rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=["fit", "gate", "modelval", "calib"], required=True)
    parser.add_argument("--limit-per-pair", type=int, default=30)
    parser.add_argument("--max-single-area-ratio", type=float)
    parser.add_argument("--max-combined-area-ratio", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    max_single_area_ratio = (
        args.max_single_area_ratio
        if args.max_single_area_ratio is not None
        else float(config["data"]["max_single_edit_area_ratio"])
    )
    max_combined_area_ratio = (
        args.max_combined_area_ratio
        if args.max_combined_area_ratio is not None
        else float(config["data"]["max_combined_edit_area_ratio"])
    )
    root_value = args.data_root or os.environ.get("EDCR_DATA_ROOT") or config["data"]["root"]
    if root_value == "REQUIRED_DATA_ROOT":
        raise SystemExit("COCO root is required via --data-root or EDCR_DATA_ROOT")
    data_root = Path(root_value).expanduser().resolve()
    coco = load_coco(data_root / "annotations/instances_train2017.json")
    manifest_path = REPO_ROOT / "data/manifests" / f"{args.split}.jsonl"
    selected_path = REPO_ROOT / "data/manifests/selected_relations.json"
    if not manifest_path.is_file() or not selected_path.is_file():
        raise SystemExit("run scripts/prepare.py before selecting intervention candidates")

    split_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    split_ids = {int(row["source_image_id"]) for row in split_rows}
    image_by_id = {int(row["id"]): row for row in coco["images"] if int(row["id"]) in split_ids}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in split_ids:
            annotations_by_image[image_id].append(annotation)
    relations = json.loads(selected_path.read_text(encoding="utf-8"))
    mapping = json.loads(
        (REPO_ROOT / "data/manifests/category_mapping.json").read_text(encoding="utf-8")
    )
    original_ids = mapping["contiguous_to_original"]
    accepted: list[dict[str, Any]] = []
    audit_counts: dict[str, dict[str, int]] = {}

    for relation in relations:
        pair_id = relation["pair_id"]
        source_category = int(original_ids[int(relation["source_id"])])
        target_category = int(original_ids[int(relation["target_id"])])
        counts: defaultdict[str, int] = defaultdict(int)
        candidates: list[dict[str, Any]] = []
        for image_id, image_annotations in annotations_by_image.items():
            source_anns = [
                ann for ann in image_annotations if int(ann["category_id"]) == source_category
            ]
            target_anns = [
                ann for ann in image_annotations if int(ann["category_id"]) == target_category
            ]
            if not source_anns or not target_anns:
                continue
            counts["cooccurring"] += 1
            if any(int(ann.get("iscrowd", 0)) for ann in source_anns + target_anns):
                counts["rejected_crowd"] += 1
                continue
            image = image_by_id[image_id]
            try:
                source_mask = decode_union(source_anns, int(image["height"]), int(image["width"]))
                target_mask = decode_union(target_anns, int(image["height"]), int(image["width"]))
            except (TypeError, ValueError, KeyError):
                counts["rejected_mask"] += 1
                continue
            min_area = min(int(source_mask.sum()), int(target_mask.sum()))
            if min_area == 0:
                counts["rejected_empty_mask"] += 1
                continue
            overlap = int((source_mask & target_mask).sum()) / min_area
            if overlap > float(config["data"]["max_overlap_ratio"]):
                counts["rejected_overlap"] += 1
                continue
            image_area = int(image["height"]) * int(image["width"])
            target_area_ratio = float(target_mask.sum()) / image_area
            context_area_ratio = float(source_mask.sum()) / image_area
            combined_area_ratio = float((target_mask | source_mask).sum()) / image_area
            if max(target_area_ratio, context_area_ratio) > max_single_area_ratio:
                counts["rejected_single_area"] += 1
                continue
            if combined_area_ratio > max_combined_area_ratio:
                counts["rejected_combined_area"] += 1
                continue
            candidates.append(
                {
                    "sample_id": f"{args.split}_{pair_id}_{image_id}",
                    "source_image_id": image_id,
                    "split": args.split,
                    "image_path": str(Path("train2017") / image["file_name"]),
                    "pair_id": pair_id,
                    "target_id": int(relation["target_id"]),
                    "context_id": int(relation["source_id"]),
                    "state": "11",
                    "degradation": "none",
                    "editor_id": None,
                    "edit_seed": config["training"]["seeds"][0],
                    "quality_status": "pending",
                    "reviewer_type": None,
                    "parent_sha256": None,
                    "image_sha256": None,
                    "mask_overlap_ratio": overlap,
                    "target_area_ratio": target_area_ratio,
                    "context_area_ratio": context_area_ratio,
                    "combined_area_ratio": combined_area_ratio,
                    "target_annotation_ids": sorted(int(ann["id"]) for ann in target_anns),
                    "context_annotation_ids": sorted(int(ann["id"]) for ann in source_anns),
                }
            )
        candidates.sort(
            key=lambda row: stable_rank(
                config["data"]["split_seed"], pair_id, int(row["source_image_id"])
            )
        )
        selected = candidates[: args.limit_per_pair]
        counts["eligible"] = len(candidates)
        counts["selected"] = len(selected)
        audit_counts[pair_id] = dict(counts)
        accepted.extend(selected)

    summary = {
        "schema_version": 1,
        "split": args.split,
        "requested_per_pair": args.limit_per_pair,
        "selected_total": len(accepted),
        "quality_status": "pending",
        "editing_started": False,
        "max_single_area_ratio": max_single_area_ratio,
        "max_combined_area_ratio": max_combined_area_ratio,
        "counts": audit_counts,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "data/interventions" / args.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("candidates.jsonl", "review_queue.jsonl"):
        with (output_dir / filename).open("w", encoding="utf-8") as handle:
            for row in accepted:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(row["selected"] == args.limit_per_pair for row in audit_counts.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
