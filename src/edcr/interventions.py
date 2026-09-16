"""Mask utilities and deterministic selection helpers for interventions."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def decode_union(annotations: list[dict[str, Any]], height: int, width: int) -> np.ndarray:
    union = np.zeros((height, width), dtype=bool)
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if not segmentation:
            raise ValueError("missing segmentation")
        if isinstance(segmentation, list) or (
            isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list)
        ):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        union |= decoded.astype(bool)
    return union


def stable_rank(seed: int, pair_id: str, image_id: int) -> bytes:
    return hashlib.sha256(f"{seed}:{pair_id}:{image_id}".encode()).digest()


def dilate_mask(mask: np.ndarray, pixels: int = 3) -> np.ndarray:
    if pixels < 0:
        raise ValueError("pixels must be non-negative")
    binary = mask.astype(np.uint8)
    if pixels == 0:
        return binary
    size = pixels * 2 + 1
    kernel = np.ones((size, size), dtype=np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


def inpaint_rgb(
    image: np.ndarray, mask: np.ndarray, radius: float = 3.0, method: str = "telea"
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3 RGB")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask and image dimensions differ")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    methods = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}
    if method not in methods:
        raise ValueError(f"unsupported inpainting method: {method}")
    result = cv2.inpaint(bgr, (mask.astype(np.uint8) * 255), radius, methods[method])
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def degrade_target(
    image: np.ndarray, mask: np.ndarray, operation: str, strength: str
) -> np.ndarray:
    if operation not in {"blur", "contrast"}:
        raise ValueError(f"unsupported degradation operation: {operation}")
    if strength not in {"light", "medium"}:
        raise ValueError(f"unsupported degradation strength: {strength}")
    if operation == "blur":
        kernel_size = 15 if strength == "light" else 31
        changed = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
    else:
        alpha = 0.65 if strength == "light" else 0.4
        mean = image.mean(axis=(0, 1), keepdims=True)
        changed = np.clip(alpha * image + (1.0 - alpha) * mean, 0, 255).astype(np.uint8)
    output = image.copy()
    output[mask.astype(bool)] = changed[mask.astype(bool)]
    return output
