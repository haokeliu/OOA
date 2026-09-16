#!/usr/bin/env python3
"""Generate deterministic four-tuples and controlled target degradations."""

from __future__ import annotations

import argparse
import hashlib
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
from edcr.interventions import decode_union, degrade_target, dilate_mask, inpaint_rgb


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_rgb(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part.jpg")
    Image.fromarray(image).save(temporary, format="JPEG", quality=95, optimize=True)
    temporary.replace(path)
    return sha256(path)


def output_record(
    parent: dict[str, Any],
    *,
    variant: str,
    path: Path,
    digest: str,
    labels: list[int],
    state: str,
    degradation: str,
    edit_count: int,
    held_out_operation: bool = False,
    editor_id: str,
) -> dict[str, Any]:
    return {
        **parent,
        "sample_id": f"{parent['sample_id']}_{variant}",
        "image_path": str(path.relative_to(REPO_ROOT)),
        "labels": labels,
        "label_format": "positive_contiguous_ids",
        "state": state,
        "degradation": degradation,
        "editor_id": editor_id if edit_count else "original",
        "quality_status": "pending",
        "reviewer_type": None,
        "image_sha256": digest,
        "edit_count": edit_count,
        "held_out_operation": held_out_operation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="data/interventions/gate/candidates.jsonl")
    parser.add_argument(
        "--annotations", default="data/coco2017/annotations/instances_train2017.json"
    )
    parser.add_argument("--data-root", default="data/coco2017")
    parser.add_argument("--base-manifest", default="data/manifests/gate.jsonl")
    parser.add_argument("--output-dir", default="data/interventions/gate/generated")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inpaint-method", choices=["telea", "ns"], default="telea")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    editor_id = f"opencv_{args.inpaint_method}_{cv2.__version__}_radius3_dilate3"

    queue_path = REPO_ROOT / args.queue
    rows = [json.loads(line) for line in queue_path.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.dry_run:
        print(json.dumps({"parents": len(rows), "expected_records": len(rows) * 7}, indent=2))
        return 0
    data_root = REPO_ROOT / args.data_root
    output_dir = REPO_ROOT / args.output_dir
    coco = load_coco(REPO_ROOT / args.annotations)
    needed_ids = {
        int(value)
        for row in rows
        for key in ("target_annotation_ids", "context_annotation_ids")
        for value in row[key]
    }
    annotations_by_id = {
        int(row["id"]): row for row in coco["annotations"] if int(row["id"]) in needed_ids
    }
    labels_by_image = {
        int(row["source_image_id"]): row["labels"]
        for row in (
            json.loads(line)
            for line in (REPO_ROOT / args.base_manifest).read_text(encoding="utf-8").splitlines()
        )
    }
    generated: list[dict[str, Any]] = []
    variant_counts: defaultdict[str, int] = defaultdict(int)
    for parent in rows:
        original_path = data_root / parent["image_path"]
        with Image.open(original_path) as pil_image:
            original = np.asarray(pil_image.convert("RGB"))
        height, width = original.shape[:2]
        target_annotations = [
            annotations_by_id[int(value)] for value in parent["target_annotation_ids"]
        ]
        context_annotations = [
            annotations_by_id[int(value)] for value in parent["context_annotation_ids"]
        ]
        target_mask = dilate_mask(decode_union(target_annotations, height, width), 3)
        context_mask = dilate_mask(decode_union(context_annotations, height, width), 3)
        original_labels = sorted(
            int(value) for value in labels_by_image[int(parent["source_image_id"])]
        )
        target_labels = [value for value in original_labels if value != int(parent["target_id"])]
        context_labels = [value for value in original_labels if value != int(parent["context_id"])]
        neither_labels = [
            value
            for value in original_labels
            if value not in {int(parent["target_id"]), int(parent["context_id"])}
        ]
        base_name = f"{parent['source_image_id']}"
        pair_dir = output_dir / parent["pair_id"]
        original_digest = sha256(original_path)
        generated.append(
            output_record(
                parent,
                variant="x11",
                path=original_path,
                digest=original_digest,
                labels=original_labels,
                state="11",
                degradation="none",
                edit_count=0,
                editor_id=editor_id,
            )
        )
        variant_counts["x11"] += 1

        variants = {
            "x10": (
                inpaint_rgb(original, context_mask, method=args.inpaint_method),
                context_labels,
                "10",
                "none",
                1,
                False,
            ),
            "x01": (
                inpaint_rgb(original, target_mask, method=args.inpaint_method),
                target_labels,
                "01",
                "none",
                1,
                False,
            ),
        }
        x00_first = inpaint_rgb(original, target_mask, method=args.inpaint_method)
        variants["x00"] = (
            inpaint_rgb(x00_first, context_mask, method=args.inpaint_method),
            neither_labels,
            "00",
            "none",
            2,
            False,
        )
        for strength in ("light", "medium"):
            name = f"target_blur_{strength}"
            variants[name] = (
                degrade_target(original, target_mask, "blur", strength),
                original_labels,
                "11",
                f"blur_{strength}",
                1,
                False,
            )
        variants["target_contrast_medium"] = (
            degrade_target(original, target_mask, "contrast", "medium"),
            original_labels,
            "11",
            "contrast_medium",
            1,
            True,
        )
        for variant, (image, labels, state, degradation, edit_count, held_out) in variants.items():
            path = pair_dir / f"{base_name}_{variant}.jpg"
            digest = write_rgb(path, image)
            generated.append(
                output_record(
                    parent,
                    variant=variant,
                    path=path,
                    digest=digest,
                    labels=labels,
                    state=state,
                    degradation=degradation,
                    edit_count=edit_count,
                    held_out_operation=held_out,
                    editor_id=editor_id,
                )
            )
            variant_counts[variant] += 1

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in generated:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "editor_id": editor_id,
        "parent_count": len(rows),
        "record_count": len(generated),
        "variant_counts": dict(sorted(variant_counts.items())),
        "quality_status": "pending",
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
