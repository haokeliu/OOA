#!/usr/bin/env python3
"""Cache frozen CLIP and ResNet-18 features for V9 CIFAR-100."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data/v9_cifar100"
FEATURE_ROOT = REPO_ROOT / "data/v9_cifar100_features"
TORCH_HOME = DATA_ROOT / "torch_home"
CLIP_CHECKPOINT = REPO_ROOT / "data/checkpoints/ViT-B-16.pt"
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))

from torchvision.models import ResNet18_Weights, resnet18


class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, transform) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(Image.fromarray(self.images[index]))


def sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_batch(split: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    batch_path = DATA_ROOT / "cifar-100-python" / ("train" if split == "train" else "test")
    with batch_path.open("rb") as handle:
        payload = pickle.load(handle, encoding="bytes")
    images = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    if split != "train":
        # The official pickle physically contains labels. They are deliberately never indexed,
        # summarized, returned, or written by the label-blind cache stage.
        return images, None, None
    return (
        images,
        np.asarray(payload[b"fine_labels"], dtype=np.int64),
        np.asarray(payload[b"coarse_labels"], dtype=np.int64),
    )


def deterministic_training_partition(labels: np.ndarray) -> np.ndarray:
    partition = np.empty(len(labels), dtype="U20")
    for class_index in range(100):
        indices = np.flatnonzero(labels == class_index)
        if len(indices) != 500:
            raise RuntimeError(f"unexpected CIFAR-100 class support: {class_index}={len(indices)}")
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"v9-cifar100:{class_index}:{index}".encode()
            ).digest(),
        )
        partition[ranked[:350]] = "candidate_fit"
        partition[ranked[350:400]] = "candidate_calib"
        partition[ranked[400:]] = "router"
    return partition


def encode(model: nn.Module, loader: DataLoader, clip: bool) -> np.ndarray:
    values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            if clip:
                features = model.encode_image(batch, normalize=True)
            else:
                features = model(batch)
                features = torch.nn.functional.normalize(features, dim=-1)
            values.append(features.cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test-unlabeled"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    split = "train" if args.split == "train" else "test"
    output_path = FEATURE_ROOT / f"{args.split}.npz"
    report_path = REPO_ROOT / f"reports/v9b0_cifar100_{args.split}_feature_cache.json"
    if output_path.exists() or report_path.exists():
        raise RuntimeError(f"refusing to overwrite existing V9 cache: {output_path}")

    started = time.perf_counter()
    images, fine_labels, coarse_labels = load_batch(split)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    clip_model = open_clip.load_openai_model(
        str(CLIP_CHECKPOINT), device="cpu", precision="fp32"
    )
    clip_transform = open_clip.image_transform(
        clip_model.visual.image_size,
        is_train=False,
        mean=getattr(clip_model.visual, "image_mean", None),
        std=getattr(clip_model.visual, "image_std", None),
    )
    clip_loader = DataLoader(
        ImageDataset(images, clip_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    clip_features = encode(clip_model, clip_loader, clip=True)
    del clip_model, clip_loader

    weights = ResNet18_Weights.IMAGENET1K_V1
    resnet_model = resnet18(weights=weights)
    resnet_model.fc = nn.Identity()
    resnet_loader = DataLoader(
        ImageDataset(images, weights.transforms()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    resnet_features = encode(resnet_model, resnet_loader, clip=False)

    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "sample_index": np.arange(len(images), dtype=np.int64),
        "clip": clip_features,
        "resnet18": resnet_features,
    }
    if args.split == "train":
        if fine_labels is None or coarse_labels is None:
            raise RuntimeError("training labels missing")
        payload.update(
            {
                "fine_labels": fine_labels,
                "coarse_labels": coarse_labels,
                "partition": deterministic_training_partition(fine_labels),
            }
        )
    np.savez_compressed(output_path, **payload)

    input_batch = DATA_ROOT / "cifar-100-python" / split
    report = {
        "schema_version": 1,
        "protocol": "v9-cifar100-independent-backbone-feature-cache",
        "split": args.split,
        "sample_count": len(images),
        "feature_dimensions": {
            "clip": clip_features.shape[1],
            "resnet18": resnet_features.shape[1],
        },
        "training_labels_accessed": args.split == "train",
        "test_labels_accessed": False,
        "artifact_sha256": {
            "input_batch": sha256(input_batch),
            "clip_checkpoint": sha256(CLIP_CHECKPOINT),
            "resnet18_checkpoint": sha256(
                TORCH_HOME / "hub/checkpoints/resnet18-f37072fd.pth"
            ),
            "output_cache": sha256(output_path),
            "script": sha256(Path(__file__)),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
