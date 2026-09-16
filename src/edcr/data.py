"""COCO label mapping, leakage-safe splitting, and relation statistics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("fit", "gate", "modelval", "calib")


def mapping_from_categories(categories: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(categories, key=lambda row: int(row["id"]))
    original_to_contiguous = {str(row["id"]): idx for idx, row in enumerate(ordered)}
    contiguous_to_original = [int(row["id"]) for row in ordered]
    names = [str(row["name"]) for row in ordered]
    canonical = json.dumps(
        {"original_to_contiguous": original_to_contiguous, "names": names},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "original_to_contiguous": original_to_contiguous,
        "contiguous_to_original": contiguous_to_original,
        "names": names,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def image_labels(
    annotations: Iterable[dict[str, Any]], mapping: dict[str, Any]
) -> dict[int, list[int]]:
    labels: dict[int, set[int]] = defaultdict(set)
    forward = mapping["original_to_contiguous"]
    for ann in annotations:
        labels[int(ann["image_id"])].add(forward[str(ann["category_id"])])
    return {image_id: sorted(values) for image_id, values in labels.items()}


def maximum_noncrowd_areas(
    annotations: Iterable[dict[str, Any]], mapping: dict[str, Any]
) -> dict[tuple[int, int], float]:
    """Return the largest instance area for every image/category pair."""
    output: dict[tuple[int, int], float] = {}
    forward = mapping["original_to_contiguous"]
    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        key = (
            int(annotation["image_id"]),
            int(forward[str(annotation["category_id"])]),
        )
        output[key] = max(output.get(key, 0.0), float(annotation["area"]))
    return output


def maximum_noncrowd_area_ratios(
    annotations: Iterable[dict[str, Any]],
    images: Iterable[dict[str, Any]],
    mapping: dict[str, Any],
) -> dict[tuple[int, int], float]:
    dimensions = {int(image["id"]): float(image["width"] * image["height"]) for image in images}
    areas = maximum_noncrowd_areas(annotations, mapping)
    return {key: area / dimensions[key[0]] for key, area in areas.items()}


def remove_instances(
    annotations: Iterable[dict[str, Any]], *, image_id: int, category_id: int
) -> list[dict[str, Any]]:
    """Remove every instance of one category from one image."""
    return [
        ann
        for ann in annotations
        if not (int(ann["image_id"]) == image_id and int(ann["category_id"]) == category_id)
    ]


def _allocation_counts(total: int, proportions: list[float]) -> list[int]:
    raw = [total * value for value in proportions]
    base = [math.floor(value) for value in raw]
    remainder = total - sum(base)
    order = sorted(range(len(raw)), key=lambda idx: (-(raw[idx] - base[idx]), idx))
    for idx in order[:remainder]:
        base[idx] += 1
    return base


def deterministic_split(
    image_ids: Iterable[int], *, seed: int, proportions: list[float]
) -> dict[int, str]:
    unique = sorted({int(image_id) for image_id in image_ids})
    ranked = sorted(
        unique,
        key=lambda image_id: hashlib.sha256(f"{seed}:{image_id}".encode()).digest(),
    )
    counts = _allocation_counts(len(ranked), proportions)
    result: dict[int, str] = {}
    offset = 0
    for name, count in zip(SPLIT_NAMES, counts, strict=True):
        for image_id in ranked[offset : offset + count]:
            result[image_id] = name
        offset += count
    return result


def relation_counts(
    label_sets: Iterable[set[int]], class_count: int
) -> list[dict[str, int | float]]:
    rows = list(label_sets)
    total = len(rows)
    output: list[dict[str, int | float]] = []
    for source in range(class_count):
        for target in range(class_count):
            if source == target:
                continue
            n11 = sum(source in labels and target in labels for labels in rows)
            n10 = sum(source not in labels and target in labels for labels in rows)
            n01 = sum(source in labels and target not in labels for labels in rows)
            n00 = total - n11 - n10 - n01
            # Symmetric smoothed lift is only the ranking score; direction is assigned below.
            p_joint = (n11 + 0.5) / (total + 1.0)
            p_source = (n11 + n01 + 0.5) / (total + 1.0)
            p_target = (n11 + n10 + 0.5) / (total + 1.0)
            output.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "n11": n11,
                    "n10": n10,
                    "n01": n01,
                    "n00": n00,
                    "smoothed_lift": p_joint / (p_source * p_target),
                }
            )
    return output


def select_relations(
    rows: Iterable[dict[str, int | float]],
    names: list[str],
    pair_count: int,
) -> list[dict[str, int | float | str]]:
    """Select diverse directed relations with a frozen, label-only tie-break."""
    by_unordered_pair: dict[tuple[int, int], list[dict[str, int | float]]] = defaultdict(list)
    for row in rows:
        source = int(row["source_id"])
        target = int(row["target_id"])
        by_unordered_pair[tuple(sorted((source, target)))].append(row)

    directed: list[dict[str, int | float]] = []
    for variants in by_unordered_pair.values():
        # n10 is target-only support. Prefer the direction whose target has more of it.
        variants.sort(key=lambda row: (-int(row["n10"]), int(row["target_id"])))
        directed.append(variants[0])
    directed.sort(
        key=lambda row: (
            -float(row["smoothed_lift"]),
            int(row["target_id"]),
            int(row["source_id"]),
        )
    )

    selected: list[dict[str, int | float | str]] = []
    used_targets: set[int] = set()
    person_source_count = 0
    for row in directed:
        source = int(row["source_id"])
        target = int(row["target_id"])
        if target in used_targets:
            continue
        if names[source] == "person" and person_source_count >= 1:
            continue
        result: dict[str, int | float | str] = dict(row)
        result.update(
            {
                "pair_id": f"{names[source].replace(' ', '_')}_to_{names[target].replace(' ', '_')}",
                "source_name": names[source],
                "target_name": names[target],
            }
        )
        selected.append(result)
        used_targets.add(target)
        person_source_count += int(names[source] == "person")
        if len(selected) == pair_count:
            break
    return selected


def load_coco(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
