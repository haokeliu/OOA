#!/usr/bin/env python3
"""Cache independently encoded fixed five-view CLIP features."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.config import load_config
from edcr.features import make_views, view_boxes


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_image(row: dict[str, Any], data_root: Path) -> Path:
    relative = Path(row["image_path"])
    candidates = (data_root / relative, REPO_ROOT / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image not found in allowed roots: {relative}")


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def encode_rows(
    rows: list[dict[str, Any]],
    *,
    model: torch.nn.Module,
    preprocess: Any,
    data_root: Path,
    crop_ratio: float,
    device: str,
    microbatch: int,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    encoded: list[torch.Tensor] = []
    pending: list[torch.Tensor] = []
    sample_ids: list[str] = []
    image_hashes: list[str] = []
    box_records: list[str] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending).to(device)
        features = model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        encoded.append(features.cpu())
        pending.clear()

    with torch.inference_mode():
        for row in rows:
            path = resolve_image(row, data_root)
            image_hashes.append(sha256(path))
            sample_ids.append(str(row["sample_id"]))
            with Image.open(path) as image:
                box_records.append(json.dumps(view_boxes(image.width, image.height, crop_ratio)))
                for view in make_views(image, crop_ratio):
                    pending.append(preprocess(view))
                    if len(pending) == microbatch:
                        flush()
        flush()
    if device == "cuda":
        torch.cuda.synchronize()
    array = torch.cat(encoded).reshape(len(rows), 5, -1).numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("non-finite feature detected")
    return array, sample_ids, image_hashes, box_records


def cache_shards(
    rows: list[dict[str, Any]],
    *,
    model: torch.nn.Module,
    preprocess: Any,
    data_root: Path,
    crop_ratio: float,
    device: str,
    microbatch: int,
    shard_dir: Path,
    shard_size: int,
    checkpoint_hash: str,
    manifest_hash: str,
) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    total_shards = (len(rows) + shard_size - 1) // shard_size
    completed = 0
    resumed = 0
    encoded_images = 0
    started = time.perf_counter()
    for shard_index in range(total_shards):
        shard_rows = rows[shard_index * shard_size : (shard_index + 1) * shard_size]
        shard_path = shard_dir / f"shard_{shard_index:05d}.npz"
        if shard_path.is_file():
            with np.load(shard_path) as existing:
                valid = (
                    existing["embeddings"].shape[0] == len(shard_rows)
                    and str(existing["checkpoint_sha256"]) == checkpoint_hash
                    and str(existing["manifest_sha256"]) == manifest_hash
                )
            if valid:
                resumed += 1
                completed += 1
                continue
            raise ValueError(f"stale or invalid existing shard: {shard_path}")
        array, sample_ids, image_hashes, boxes = encode_rows(
            shard_rows,
            model=model,
            preprocess=preprocess,
            data_root=data_root,
            crop_ratio=crop_ratio,
            device=device,
            microbatch=microbatch,
        )
        temporary = shard_path.with_suffix(".part.npz")
        np.savez_compressed(
            temporary,
            sample_id=np.asarray(sample_ids),
            image_sha256=np.asarray(image_hashes),
            view_boxes=np.asarray(boxes),
            embeddings=array,
            method=np.asarray("openai_clip_vit_b16_five_views_fp32"),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            manifest_sha256=np.asarray(manifest_hash),
            crop_ratio=np.asarray(crop_ratio),
        )
        temporary.replace(shard_path)
        encoded_images += len(shard_rows)
        completed += 1
        state = {
            "schema_version": 1,
            "completed_shards": completed,
            "total_shards": total_shards,
            "encoded_images_this_run": encoded_images,
            "resumed_shards": resumed,
        }
        (shard_dir / "state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elapsed = time.perf_counter() - started
    index = {
        "schema_version": 1,
        "complete": completed == total_shards,
        "sample_count": len(rows),
        "view_count": len(rows) * 5,
        "shard_size": shard_size,
        "shard_count": total_shards,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "crop_ratio": crop_ratio,
        "dtype": "float32",
        "encoded_images_this_run": encoded_images,
        "resumed_shards": resumed,
        "elapsed_seconds_this_run": elapsed,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    (shard_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--manifest", default="data/interventions/gate/candidates.jsonl")
    parser.add_argument("--data-root", default="data/coco2017")
    parser.add_argument("--checkpoint", default="data/checkpoints/ViT-B-16.pt")
    parser.add_argument("--output", default="data/features/gate_candidate_profile.npz")
    parser.add_argument("--shard-dir")
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--report", default="reports/p0_feature_profile.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    checkpoint = REPO_ROOT / args.checkpoint
    actual_checkpoint_hash = sha256(checkpoint)
    expected_checkpoint_hash = config["features"]["checkpoint_sha256"]
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise SystemExit(
            f"checkpoint hash mismatch: expected {expected_checkpoint_hash}, got {actual_checkpoint_hash}"
        )
    manifest_path = REPO_ROOT / args.manifest
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    unique_rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for row in rows:
        if row["image_path"] not in seen_paths:
            unique_rows.append(row)
            seen_paths.add(row["image_path"])
    if not args.all:
        unique_rows = unique_rows[: args.limit]
    device = choose_device(config["execution"]["device"])
    microbatch = args.microbatch or (32 if device != "cpu" else 16)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "unique_images": len(unique_rows),
                    "view_count": len(unique_rows) * 5,
                    "device": device,
                    "microbatch": microbatch,
                },
                indent=2,
            )
        )
        return 0

    started = time.perf_counter()
    model = open_clip.load_openai_model(str(checkpoint), device=device, precision="fp32")
    model.eval()
    preprocess = open_clip.image_transform(
        model.visual.image_size,
        is_train=False,
        mean=model.visual.image_mean,
        std=model.visual.image_std,
    )
    load_seconds = time.perf_counter() - started
    crop_ratio = float(config["features"]["crop_ratio"])
    if args.shard_dir:
        report = cache_shards(
            unique_rows,
            model=model,
            preprocess=preprocess,
            data_root=REPO_ROOT / args.data_root,
            crop_ratio=crop_ratio,
            device=device,
            microbatch=microbatch,
            shard_dir=REPO_ROOT / args.shard_dir,
            shard_size=args.shard_size,
            checkpoint_hash=actual_checkpoint_hash,
            manifest_hash=sha256(manifest_path),
        )
        report.update(
            {
                "scientific_result": False,
                "purpose": "resumable frozen feature cache",
                "device": device,
                "microbatch": microbatch,
                "model_load_seconds": load_seconds,
            }
        )
        report_path = REPO_ROOT / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    tensors: list[torch.Tensor] = []
    source_paths: list[Path] = []
    for row in unique_rows:
        path = resolve_image(row, REPO_ROOT / args.data_root)
        with Image.open(path) as image:
            views = make_views(image, crop_ratio)
            tensors.extend(preprocess(view) for view in views)
        source_paths.append(path)
    encoded: list[torch.Tensor] = []
    encode_started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(tensors), microbatch):
            batch = torch.stack(tensors[offset : offset + microbatch]).to(device)
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            encoded.append(features.cpu())
    if device == "cuda":
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started
    array = torch.cat(encoded).reshape(len(unique_rows), 5, -1).numpy().astype(np.float32)
    if not np.isfinite(array).all():
        raise SystemExit("non-finite feature detected")
    output = REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_id=np.asarray([row["sample_id"] for row in unique_rows]),
        image_sha256=np.asarray([sha256(path) for path in source_paths]),
        embeddings=array,
        method=np.asarray("openai_clip_vit_b16_five_views_fp32"),
        checkpoint_sha256=np.asarray(actual_checkpoint_hash),
        crop_ratio=np.asarray(crop_ratio),
    )
    image_throughput = len(unique_rows) / encode_seconds
    full_image_count = 118_287
    report = {
        "schema_version": 1,
        "scientific_result": False,
        "purpose": "P0 throughput and numerical profile",
        "device": device,
        "microbatch": microbatch,
        "image_count": len(unique_rows),
        "view_count": len(tensors),
        "embedding_shape": list(array.shape),
        "embedding_dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "load_seconds": load_seconds,
        "encode_seconds": encode_seconds,
        "images_per_second": image_throughput,
        "views_per_second": len(tensors) / encode_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "checkpoint_sha256": actual_checkpoint_hash,
        "cache_bytes": output.stat().st_size,
        "estimated_full_encode_hours": full_image_count / image_throughput / 3600,
        "estimated_full_float32_cache_bytes": full_image_count * 5 * array.shape[-1] * 4,
    }
    report_path = REPO_ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
