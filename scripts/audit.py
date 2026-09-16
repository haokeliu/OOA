#!/usr/bin/env python3
"""Run the P0 environment, repository, dependency, and data readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.config import load_config


def digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_command(args: list[str]) -> str | None:
    try:
        return (
            subprocess.run(
                args, capture_output=True, text=True, check=False, timeout=10
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--output", default="reports/p0_environment.json")
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    config = load_config(config_path)
    data_root_value = args.data_root or os.environ.get("EDCR_DATA_ROOT") or config["data"]["root"]
    data_root = (
        None if data_root_value == "REQUIRED_DATA_ROOT" else Path(data_root_value).expanduser()
    )
    train_annotations_ready = bool(
        data_root and (data_root / "annotations/instances_train2017.json").is_file()
    )
    val_annotations_ready = bool(
        data_root and (data_root / "annotations/instances_val2017.json").is_file()
    )
    train_image_count = (
        sum(1 for _ in (data_root / "train2017").glob("*.jpg"))
        if data_root and (data_root / "train2017").is_dir()
        else 0
    )
    val_image_count = (
        sum(1 for _ in (data_root / "val2017").glob("*.jpg"))
        if data_root and (data_root / "val2017").is_dir()
        else 0
    )
    data_checks = {
        "train_annotations": train_annotations_ready,
        "val_annotations": val_annotations_ready,
        "train_images_complete": train_image_count == 118_287,
        "val_images_complete": val_image_count == 5_000,
    }
    modules = [
        "numpy",
        "PIL",
        "pycocotools",
        "yaml",
        "sklearn",
        "torch",
        "torchvision",
        "open_clip",
        "cv2",
    ]
    dependency_checks = {name: importlib.util.find_spec(name) is not None for name in modules}
    distributions = {
        "numpy": "numpy",
        "PIL": "pillow",
        "pycocotools": "pycocotools",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "torch": "torch",
        "torchvision": "torchvision",
        "open_clip": "open-clip-torch",
        "cv2": "opencv-python-headless",
    }
    dependency_versions = {
        module: importlib.metadata.version(distribution) if dependency_checks[module] else None
        for module, distribution in distributions.items()
    }
    disk = shutil.disk_usage(REPO_ROOT)
    git_commit = safe_command(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "protocol_version": config["protocol_version"],
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": digest(config_path),
        "protocol_sha256": digest(REPO_ROOT / "实验执行方案.md"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
        },
        "resources": {
            "logical_cpu_count": os.cpu_count(),
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
        },
        "git": {"is_repository": git_commit is not None, "commit": git_commit},
        "dependencies": {
            name: {"available": dependency_checks[name], "version": dependency_versions[name]}
            for name in modules
        },
        "data": {
            "root_configured": data_root is not None,
            "root": str(data_root) if data_root else None,
            "checks": data_checks,
            "train_image_count": train_image_count,
            "val_image_count": val_image_count,
            "ready": all(data_checks.values()),
        },
        "checkpoint": {
            "model": config["features"]["backbone"],
            "pretrained": config["features"]["checkpoint"],
            "expected_sha256": config["features"]["checkpoint_sha256"],
            "path": "data/checkpoints/ViT-B-16.pt",
            "downloaded": (REPO_ROOT / "data/checkpoints/ViT-B-16.pt").is_file(),
            "actual_sha256": digest(REPO_ROOT / "data/checkpoints/ViT-B-16.pt"),
        },
    }
    try:
        import torch

        report["accelerators"] = {
            "cuda_available": torch.cuda.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }
    except ImportError:
        report["accelerators"] = None
    checkpoint_ready = (
        report["checkpoint"]["actual_sha256"] == report["checkpoint"]["expected_sha256"]
    )
    report["p0_ready_for_p1_metadata"] = (
        all(dependency_checks.values()) and train_annotations_ready and checkpoint_ready
    )
    report["ready_for_full_pipeline"] = (
        report["p0_ready_for_p1_metadata"] and report["data"]["ready"]
    )
    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_full_pipeline"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
