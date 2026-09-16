#!/usr/bin/env python3
"""Audit authoritative split leakage and multi-label frequency drift."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("fit", "gate", "modelval", "calib")


def main() -> int:
    mapping = json.loads(
        (REPO_ROOT / "data/manifests/category_mapping.json").read_text(encoding="utf-8")
    )
    class_count = len(mapping["names"])
    ids_by_split: dict[str, set[int]] = {}
    counts_by_split: dict[str, np.ndarray] = {}
    rows_by_split: dict[str, int] = {}
    for split in SPLITS:
        rows = [
            json.loads(line)
            for line in (REPO_ROOT / "data/manifests" / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        ids = {int(row["source_image_id"]) for row in rows}
        if len(ids) != len(rows):
            raise SystemExit(f"duplicate source image in {split}")
        ids_by_split[split] = ids
        rows_by_split[split] = len(rows)
        counts = np.zeros(class_count, dtype=np.int64)
        for row in rows:
            labels = [int(value) for value in row["labels"]]
            if len(labels) != len(set(labels)) or any(
                value < 0 or value >= class_count for value in labels
            ):
                raise SystemExit(f"invalid label list in {split}: {row['sample_id']}")
            counts[labels] += 1
        counts_by_split[split] = counts
    intersections = {
        f"{left}::{right}": len(ids_by_split[left] & ids_by_split[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1 :]
    }
    if any(intersections.values()):
        raise SystemExit(f"split leakage detected: {intersections}")
    total_count = sum(rows_by_split.values())
    total_labels = sum(counts_by_split.values(), np.zeros(class_count, dtype=np.int64))
    total_prevalence = total_labels / total_count
    frequency_rows: list[dict[str, str | int | float]] = []
    max_deviation = 0.0
    for split in SPLITS:
        prevalence = counts_by_split[split] / rows_by_split[split]
        for class_index, name in enumerate(mapping["names"]):
            deviation = float(prevalence[class_index] - total_prevalence[class_index])
            max_deviation = max(max_deviation, abs(deviation))
            frequency_rows.append(
                {
                    "split": split,
                    "class_id": class_index,
                    "class_name": name,
                    "positive_count": int(counts_by_split[split][class_index]),
                    "prevalence": float(prevalence[class_index]),
                    "overall_prevalence": float(total_prevalence[class_index]),
                    "absolute_deviation": deviation,
                }
            )
    report_dir = REPO_ROOT / "reports"
    with (report_dir / "p0_split_frequency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frequency_rows[0]))
        writer.writeheader()
        writer.writerows(frequency_rows)
    report = {
        "schema_version": 1,
        "split_algorithm": "sha256-rank-largest-remainder-v1",
        "image_counts": rows_by_split,
        "pairwise_source_image_intersections": intersections,
        "total_unique_images": len(set().union(*ids_by_split.values())),
        "max_absolute_class_prevalence_deviation": max_deviation,
        "mapping_sha256": mapping["sha256"],
        "passed": not any(intersections.values()) and total_count == 118_287,
    }
    (report_dir / "p0_split_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
