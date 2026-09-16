"""Fixed five-view geometry and feature normalization utilities."""

from __future__ import annotations

from PIL import Image


def view_boxes(width: int, height: int, crop_ratio: float) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1]")
    crop_width = max(1, round(width * crop_ratio))
    crop_height = max(1, round(height * crop_ratio))
    return [
        (0, 0, width, height),
        (0, 0, crop_width, crop_height),
        (width - crop_width, 0, width, crop_height),
        (0, height - crop_height, crop_width, height),
        (width - crop_width, height - crop_height, width, height),
    ]


def make_views(image: Image.Image, crop_ratio: float) -> list[Image.Image]:
    rgb = image.convert("RGB")
    return [rgb.crop(box) for box in view_boxes(rgb.width, rgb.height, crop_ratio)]
