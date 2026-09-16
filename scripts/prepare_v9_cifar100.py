#!/usr/bin/env python3
"""Download V9 training data and independent backbone weights without parsing test labels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data/v9_cifar100"
TORCH_HOME = DATA_ROOT / "torch_home"
OUTPUT_PATH = REPO_ROOT / "reports/v9b0_cifar100_preparation.json"
CLIP_CHECKPOINT = REPO_ROOT / "data/checkpoints/ViT-B-16.pt"
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

from torchvision.datasets import CIFAR100
from torchvision.models import ResNet18_Weights, resnet18


def sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    training = CIFAR100(root=DATA_ROOT, train=True, download=True)
    _ = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    archive = DATA_ROOT / "cifar-100-python.tar.gz"
    train_batch = DATA_ROOT / "cifar-100-python/train"
    test_batch = DATA_ROOT / "cifar-100-python/test"
    resnet_checkpoint = TORCH_HOME / "hub/checkpoints/resnet18-f37072fd.pth"
    required = (archive, train_batch, test_batch, resnet_checkpoint, CLIP_CHECKPOINT)
    missing = [str(file_path) for file_path in required if not file_path.is_file()]
    if missing:
        raise RuntimeError(f"missing downloaded artifacts: {missing}")

    report = {
        "schema_version": 1,
        "protocol": "v9b0-independent-candidate-feasibility-preparation",
        "training_count": len(training),
        "training_class_count": len(training.classes),
        "training_labels_accessed": True,
        "test_pixels_accessed": False,
        "test_labels_accessed": False,
        "resnet_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "artifact_sha256": {
            str(file_path.relative_to(REPO_ROOT)): sha256(file_path)
            for file_path in required
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
