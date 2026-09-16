#!/usr/bin/env python3
"""Render dependency-light paper figures from frozen EDCR reports."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
INK = "#172033"
MUTED = "#667085"
GRID = "#d9dee8"
BLUE = "#2f6fed"
ORANGE = "#f28e2b"
GREEN = "#2a9d78"
RED = "#d64b4b"
PURPLE = "#7d57c2"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((95, 55), title, fill=INK, font=font(50, True))
    draw.text((97, 118), subtitle, fill=MUTED, font=font(25))
    return image, draw


def axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    x_label: str,
    y_label: str,
) -> None:
    left, top, right, bottom = bounds
    draw.line((left, top, left, bottom), fill=INK, width=3)
    draw.line((left, bottom, right, bottom), fill=INK, width=3)
    draw.text(((left + right) // 2, bottom + 75), x_label, anchor="mm", fill=INK, font=font(27))
    draw.text((left, top - 20), y_label, anchor="ls", fill=INK, font=font(24, True))


def opportunity_figure() -> None:
    versions = ["P2 v1", "P2b", "P2c", "P2d", "P2e"]
    values = [0.305, 0.939, 0.952, 2.886, 12.251]
    image, draw = canvas(
        "Candidate opportunity emerged only after safe diversification",
        "Seed-17 model-validation oracle BCE improvement; red line is the frozen 2% criterion",
    )
    bounds = (150, 205, 1510, 820)
    axes(draw, bounds, "candidate version", "oracle BCE gain (%)")
    left, top, right, bottom = bounds
    maximum = 14.0
    for tick in range(0, 15, 2):
        y = bottom - int((tick / maximum) * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 22, y), str(tick), anchor="rm", fill=MUTED, font=font(22))
    threshold_y = bottom - int((2.0 / maximum) * (bottom - top))
    draw.line((left, threshold_y, right, threshold_y), fill=RED, width=3)
    bar_slot = (right - left) / len(values)
    for index, (version, value) in enumerate(zip(versions, values, strict=True)):
        center = left + bar_slot * (index + 0.5)
        width = 125
        y = bottom - int((value / maximum) * (bottom - top))
        color = GREEN if value >= 2.0 else "#aeb7c7"
        draw.rounded_rectangle((center - width / 2, y, center + width / 2, bottom), 10, fill=color)
        draw.text((center, y - 18), f"{value:.3f}%", anchor="ms", fill=INK, font=font(24, True))
        draw.text((center, bottom + 25), version, anchor="ma", fill=INK, font=font(25))
    image.save(OUTPUT_DIR / "candidate_opportunity.png", dpi=(180, 180))


def tradeoff_figure(summary: dict[str, object]) -> None:
    shown = ["B1", "B5", "B9", "M1", "M2", "O1"]
    colors = {"B1": INK, "B5": ORANGE, "B9": PURPLE, "M1": BLUE, "M2": GREEN, "O1": RED}
    image, draw = canvas(
        "Frozen test: learned gates close only a small part of the oracle gap",
        "Mean across seeds; bars show ±1 seed standard deviation. Upper-left is better.",
    )
    bounds = (180, 205, 1510, 820)
    axes(draw, bounds, "selected-target macro AP", "context false-positive rate")
    left, top, right, bottom = bounds
    x_min, x_max = 0.800, 0.855
    y_min, y_max = 0.30, 0.45

    def xy(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - x_min) / (x_max - x_min) * (right - left))
        py = bottom - int((y - y_min) / (y_max - y_min) * (bottom - top))
        return px, py

    for tick in (0.80, 0.81, 0.82, 0.83, 0.84, 0.85):
        x, _ = xy(tick, y_min)
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        draw.text((x, bottom + 24), f"{tick:.2f}", anchor="ma", fill=MUTED, font=font(21))
    for tick in (0.30, 0.33, 0.36, 0.39, 0.42, 0.45):
        _, y = xy(x_min, tick)
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 22, y), f"{tick:.2f}", anchor="rm", fill=MUTED, font=font(21))
    methods = summary["methods"]
    label_offsets = {
        "B1": (-12, -38),
        "B5": (16, 12),
        "B9": (-40, 18),
        "M1": (-44, -42),
        "M2": (18, -34),
        "O1": (-255, -34),
    }
    for method in shown:
        row = methods[method]
        x_mean = row["selected_macro_ap"]["mean"]
        y_mean = row["macro_cfpr"]["mean"]
        x_std = row["selected_macro_ap"]["sample_std"]
        y_std = row["macro_cfpr"]["sample_std"]
        x, y = xy(x_mean, y_mean)
        x_low, _ = xy(x_mean - x_std, y_mean)
        x_high, _ = xy(x_mean + x_std, y_mean)
        _, y_low = xy(x_mean, y_mean - y_std)
        _, y_high = xy(x_mean, y_mean + y_std)
        draw.line((x_low, y, x_high, y), fill=colors[method], width=4)
        draw.line((x, y_low, x, y_high), fill=colors[method], width=4)
        radius = 12 if method != "O1" else 15
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors[method])
        dx, dy = label_offsets[method]
        label = f"{method} (diagnostic)" if method == "O1" else method
        draw.text((x + dx, y + dy), label, fill=colors[method], font=font(24, True))
    image.save(OUTPUT_DIR / "test_ap_cfpr_tradeoff.png", dpi=(180, 180))


def observability_figure() -> None:
    relations = json.loads(
        (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
    )
    labels = [row["pair_id"].replace("_to_", "→").replace("_", " ") for row in relations]
    values: dict[str, list[float]] = {"confidence": [], "full": []}
    for kind, kind_values in values.items():
        for relation in range(len(relations)):
            per_seed = []
            for seed in (17, 29, 43):
                report = json.loads(
                    (REPO_ROOT / f"runs/q11_probe_seed{seed}/metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                per_seed.append(report["probes"][kind]["per_relation"][relation]["auroc"])
            kind_values.append(sum(per_seed) / len(per_seed))
    image, draw = canvas(
        "Inference features weakly separate the two natural risk groups",
        "Model-validation AUROC, averaged across seeds; 0.5 denotes chance",
    )
    bounds = (180, 205, 1510, 820)
    axes(draw, bounds, "relation", "group AUROC")
    left, top, right, bottom = bounds
    y_min, y_max = 0.45, 0.82
    for tick in (0.5, 0.6, 0.7, 0.8):
        y = bottom - int((tick - y_min) / (y_max - y_min) * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((left - 22, y), f"{tick:.1f}", anchor="rm", fill=MUTED, font=font(22))
    slot = (right - left) / len(labels)
    bar_width = 70
    for index, label in enumerate(labels):
        center = left + slot * (index + 0.5)
        for shift, kind, color in ((-40, "confidence", "#9db6ef"), (40, "full", BLUE)):
            value = values[kind][index]
            y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
            draw.rectangle((center + shift - bar_width / 2, y, center + shift + bar_width / 2, bottom), fill=color)
            draw.text((center + shift, y - 9), f"{value:.2f}", anchor="ms", fill=INK, font=font(18))
        draw.text((center, bottom + 28), label, anchor="ma", fill=INK, font=font(20))
    draw.rectangle((1110, 165, 1140, 190), fill="#9db6ef")
    draw.text((1152, 177), "confidence", anchor="lm", fill=INK, font=font(21))
    draw.rectangle((1300, 165, 1330, 190), fill=BLUE)
    draw.text((1342, 177), "full inputs", anchor="lm", fill=INK, font=font(21))
    image.save(OUTPUT_DIR / "group_observability.png", dpi=(180, 180))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads(
        (REPO_ROOT / "reports/final_test_summary.json").read_text(encoding="utf-8")
    )
    opportunity_figure()
    tradeoff_figure(summary)
    observability_figure()
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "figure_count": 3}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
