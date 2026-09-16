#!/usr/bin/env python3
"""Compute fixed, method-independent image-edit diagnostics for P1 review."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.data import load_coco
from edcr.interventions import decode_union, dilate_mask


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def variant_name(sample_id: str) -> str:
    names = (
        "target_contrast_medium",
        "target_blur_medium",
        "target_blur_light",
        "x11",
        "x10",
        "x01",
        "x00",
    )
    for name in names:
        if sample_id.endswith(f"_{name}"):
            return name
    raise ValueError(f"unknown variant: {sample_id}")


def quality_metrics(original: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if original.shape != edited.shape:
        raise ValueError("edited image dimensions changed")
    binary = mask.astype(bool)
    ring = dilate_mask(binary, 6).astype(bool) & ~binary
    difference = np.abs(edited.astype(np.float32) - original.astype(np.float32)).mean(axis=2)
    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
    edited_gray = cv2.cvtColor(edited, cv2.COLOR_RGB2GRAY).astype(np.float32)
    area = float(binary.mean())
    return {
        "edit_area_ratio": area,
        "inside_mae": float(difference[binary].mean()) if binary.any() else 0.0,
        "outside_mae": float(difference[~binary].mean()) if (~binary).any() else 0.0,
        "boundary_ring_mae": float(difference[ring].mean()) if ring.any() else 0.0,
        "original_inside_std": float(original_gray[binary].std()) if binary.any() else 0.0,
        "edited_inside_std": float(edited_gray[binary].std()) if binary.any() else 0.0,
        "surrounding_std": float(original_gray[ring].std()) if ring.any() else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/interventions/gate/generated/manifest.jsonl")
    parser.add_argument(
        "--annotations", default="data/coco2017/annotations/instances_train2017.json"
    )
    parser.add_argument("--output-csv", default="reports/p1_auto_quality.csv")
    parser.add_argument("--report", default="reports/p1_auto_quality_summary.json")
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / args.manifest).read_text(encoding="utf-8").splitlines()
    ]
    coco = load_coco(REPO_ROOT / args.annotations)
    needed_ids = {
        int(value)
        for row in rows
        for key in ("target_annotation_ids", "context_annotation_ids")
        for value in row[key]
    }
    annotations = {
        int(row["id"]): row for row in coco["annotations"] if int(row["id"]) in needed_ids
    }
    originals = {
        (row["pair_id"], int(row["source_image_id"])): row
        for row in rows
        if variant_name(row["sample_id"]) == "x11"
    }
    output_rows: list[dict[str, Any]] = []
    summary_counts: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        variant = variant_name(row["sample_id"])
        if variant == "x11":
            continue
        original_row = originals[(row["pair_id"], int(row["source_image_id"]))]
        original = load_rgb(REPO_ROOT / original_row["image_path"])
        edited = load_rgb(REPO_ROOT / row["image_path"])
        height, width = original.shape[:2]
        target_mask = decode_union(
            [annotations[int(value)] for value in row["target_annotation_ids"]], height, width
        )
        context_mask = decode_union(
            [annotations[int(value)] for value in row["context_annotation_ids"]], height, width
        )
        if variant == "x10":
            mask = context_mask
        elif variant == "x01" or variant.startswith("target_"):
            mask = target_mask
        else:
            mask = target_mask | context_mask
        mask = dilate_mask(mask, 3).astype(bool)
        metrics = quality_metrics(original, edited, mask)
        deletion = variant in {"x10", "x01", "x00"}
        texture_ratio = metrics["edited_inside_std"] / max(metrics["surrounding_std"], 1e-6)
        flags = {
            "large_edit_area": metrics["edit_area_ratio"] > 0.10,
            "extreme_edit_area": metrics["edit_area_ratio"] > 0.25,
            "low_texture_fill": deletion
            and metrics["edit_area_ratio"] > 0.01
            and texture_ratio < 0.25,
            "low_change": deletion and metrics["inside_mae"] < 5.0,
        }
        record = {
            "sample_id": row["sample_id"],
            "source_image_id": row["source_image_id"],
            "pair_id": row["pair_id"],
            "variant": variant,
            **metrics,
            "texture_ratio": texture_ratio,
            **flags,
        }
        output_rows.append(record)
        key = f"{row['pair_id']}::{variant}"
        summary_counts[key]["count"] += 1
        for flag, enabled in flags.items():
            summary_counts[key][flag] += int(enabled)

    output_csv = REPO_ROOT / args.output_csv
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "schema_version": 1,
        "scientific_result": False,
        "quality_acceptance": "pending_human_review",
        "heuristics": {
            "large_edit_area": "mask area > 10% of image",
            "extreme_edit_area": "mask area > 25% of image",
            "low_texture_fill": "deletion mask > 1% and edited/surround texture std ratio < 0.25",
            "low_change": "deletion inside-mask RGB MAE < 5",
        },
        "warning": "Heuristics prioritize manual review and must not automatically accept edits.",
        "groups": {key: dict(value) for key, value in sorted(summary_counts.items())},
        "record_count": len(output_rows),
    }
    report_path = REPO_ROOT / args.report
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
