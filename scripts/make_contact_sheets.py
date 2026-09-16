#!/usr/bin/env python3
"""Render deterministic mask-overlay contact sheets for human P1 review."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.data import load_coco
from edcr.interventions import decode_union

TARGET_COLOR = (230, 159, 0, 95)
CONTEXT_COLOR = (0, 114, 178, 95)


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int, int]) -> None:
    layer = Image.new("RGBA", image.size, color[:3] + (0,))
    alpha = Image.fromarray(mask.astype(np.uint8) * color[3])
    layer.putalpha(alpha)
    image.alpha_composite(layer)


def render_cell(
    row: dict[str, Any],
    annotations_by_id: dict[int, dict[str, Any]],
    data_root: Path,
    cell_width: int,
    cell_height: int,
) -> Image.Image:
    path = data_root / row["image_path"]
    with Image.open(path) as source_image:
        rgba = source_image.convert("RGBA")
    width, height = rgba.size
    target_annotations = [annotations_by_id[int(value)] for value in row["target_annotation_ids"]]
    context_annotations = [annotations_by_id[int(value)] for value in row["context_annotation_ids"]]
    overlay_mask(rgba, decode_union(context_annotations, height, width), CONTEXT_COLOR)
    overlay_mask(rgba, decode_union(target_annotations, height, width), TARGET_COLOR)
    thumbnail = ImageOps.contain(rgba.convert("RGB"), (cell_width - 12, cell_height - 38))
    cell = Image.new("RGB", (cell_width, cell_height), "white")
    x = (cell_width - thumbnail.width) // 2
    y = 24 + (cell_height - 30 - thumbnail.height) // 2
    cell.paste(thumbnail, (x, y))
    draw = ImageDraw.Draw(cell)
    draw.text(
        (6, 5),
        f"image {row['source_image_id']}  overlap={row['mask_overlap_ratio']:.3f}",
        fill="black",
    )
    return cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", default="data/interventions/gate/review_queue.jsonl")
    parser.add_argument(
        "--annotations", default="data/coco2017/annotations/instances_train2017.json"
    )
    parser.add_argument("--data-root", default="data/coco2017")
    parser.add_argument("--output-dir", default="reports/contact_sheets/gate")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=260)
    parser.add_argument("--cell-height", type=int, default=230)
    args = parser.parse_args()

    queue_path = (REPO_ROOT / args.queue).resolve()
    data_root = (REPO_ROOT / args.data_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows_by_pair[row["pair_id"]].append(row)
    coco = load_coco(REPO_ROOT / args.annotations)
    needed_ids = {
        int(value)
        for rows in rows_by_pair.values()
        for row in rows
        for key in ("target_annotation_ids", "context_annotation_ids")
        for value in row[key]
    }
    annotations_by_id = {
        int(row["id"]): row for row in coco["annotations"] if int(row["id"]) in needed_ids
    }
    outputs: list[str] = []
    for pair_id, rows in sorted(rows_by_pair.items()):
        rows.sort(key=lambda row: int(row["source_image_id"]))
        columns = args.columns
        row_count = (len(rows) + columns - 1) // columns
        header_height = 54
        sheet = Image.new(
            "RGB",
            (columns * args.cell_width, header_height + row_count * args.cell_height),
            (245, 245, 245),
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"{pair_id} | {len(rows)} pending-review originals", fill="black")
        draw.rectangle((10, 30, 24, 44), fill=TARGET_COLOR[:3])
        draw.text((29, 30), "target", fill="black")
        draw.rectangle((95, 30, 109, 44), fill=CONTEXT_COLOR[:3])
        draw.text((114, 30), "context source", fill="black")
        for index, row in enumerate(rows):
            cell = render_cell(row, annotations_by_id, data_root, args.cell_width, args.cell_height)
            x = index % columns * args.cell_width
            y = header_height + index // columns * args.cell_height
            sheet.paste(cell, (x, y))
        output_path = output_dir / f"{pair_id}.jpg"
        sheet.save(output_path, quality=92, optimize=True)
        outputs.append(str(output_path.relative_to(REPO_ROOT)))
    print(json.dumps({"contact_sheets": outputs, "count": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
