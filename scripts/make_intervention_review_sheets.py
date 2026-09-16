#!/usr/bin/env python3
"""Render side-by-side P1 intervention variants for visual quality review."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANT_ORDER = (
    "x11",
    "x10",
    "x01",
    "x00",
    "target_blur_light",
    "target_blur_medium",
    "target_contrast_medium",
)


def variant_name(sample_id: str) -> str:
    for name in sorted(VARIANT_ORDER, key=len, reverse=True):
        if sample_id.endswith(f"_{name}"):
            return name
    raise ValueError(f"unknown variant in sample_id: {sample_id}")


def render_thumbnail(path: Path, width: int, height: int) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    thumbnail = ImageOps.contain(rgb, (width - 8, height - 24))
    cell = Image.new("RGB", (width, height), "white")
    cell.paste(thumbnail, ((width - thumbnail.width) // 2, 21))
    return cell


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/interventions/gate/generated/manifest.jsonl")
    parser.add_argument("--output-dir", default="reports/intervention_review/gate")
    parser.add_argument("--cell-width", type=int, default=210)
    parser.add_argument("--cell-height", type=int, default=180)
    parser.add_argument("--rows-per-page", type=int, default=10)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / args.manifest).read_text(encoding="utf-8").splitlines()
    ]
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        grouped[row["pair_id"]][int(row["source_image_id"])][variant_name(row["sample_id"])] = row
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    header_height = 48
    row_label_height = 18
    for pair_id, by_image in sorted(grouped.items()):
        row_height = args.cell_height + row_label_height
        image_items = sorted(by_image.items())
        pages = [
            image_items[offset : offset + args.rows_per_page]
            for offset in range(0, len(image_items), args.rows_per_page)
        ]
        for page_index, page in enumerate(pages, start=1):
            sheet = Image.new(
                "RGB",
                (
                    len(VARIANT_ORDER) * args.cell_width,
                    header_height + len(page) * row_height,
                ),
                (242, 242, 242),
            )
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (8, 6),
                f"{pair_id} | P1 review | page {page_index}/{len(pages)} | pending",
                fill="black",
            )
            for column, variant in enumerate(VARIANT_ORDER):
                draw.text((column * args.cell_width + 6, 27), variant, fill="black")
            for row_index, (image_id, variants) in enumerate(page):
                y = header_height + row_index * row_height
                draw.text((6, y + 2), f"image {image_id}", fill="black")
                for column, variant in enumerate(VARIANT_ORDER):
                    record = variants[variant]
                    path = REPO_ROOT / record["image_path"]
                    thumbnail = render_thumbnail(path, args.cell_width, args.cell_height)
                    sheet.paste(thumbnail, (column * args.cell_width, y + row_label_height))
            output_path = output_dir / f"{pair_id}_p{page_index:02d}.jpg"
            sheet.save(output_path, quality=90, optimize=True)
            outputs.append(str(output_path.relative_to(REPO_ROOT)))
    print(json.dumps({"review_sheets": outputs, "count": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
