#!/usr/bin/env python3
"""Cache fixed source/target CLIP patch-similarity maps without labels or boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cache_features import choose_device, resolve_image

from edcr.config import load_config


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def relation_prompts() -> tuple[list[dict[str, Any]], list[str]]:
    relations = json.loads(
        (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
    )
    prompts = []
    for relation in relations:
        prompts.extend(
            (
                f"a photo of a {relation['source_name']}",
                f"a photo of a {relation['target_name']}",
            )
        )
    return relations, prompts


def text_queries(model: torch.nn.Module, prompts: list[str], device: str) -> torch.Tensor:
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    with torch.inference_mode():
        features = model.encode_text(tokenizer(prompts).to(device))
    return functional.normalize(features, dim=-1)


def encode_rows(
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    preprocess: Any,
    queries: torch.Tensor,
    relation_count: int,
    data_root: Path,
    device: str,
    microbatch: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    maps: list[torch.Tensor] = []
    pending: list[torch.Tensor] = []
    sample_ids: list[str] = []
    image_hashes: list[str] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending).to(device)
        output = model.visual.forward_intermediates(
            batch,
            indices=1,
            normalize_intermediates=True,
            intermediates_only=True,
            output_fmt="NLC",
        )
        patches = output["image_intermediates"][0]
        patches = functional.normalize(patches @ model.visual.proj, dim=-1)
        similarities = patches @ queries.T
        side = round(similarities.shape[1] ** 0.5)
        if side * side != similarities.shape[1]:
            raise ValueError("patch-token count is not a square grid")
        similarities = similarities.reshape(
            len(batch), side, side, relation_count, 2
        ).permute(0, 3, 4, 1, 2)
        maps.append(similarities.cpu())
        pending.clear()

    with torch.inference_mode():
        for row in rows:
            path = resolve_image(row, data_root)
            sample_ids.append(str(row["sample_id"]))
            image_hashes.append(sha256(path))
            with Image.open(path) as image:
                pending.append(preprocess(image.convert("RGB")))
            if len(pending) == microbatch:
                flush()
        flush()
    if device == "cuda":
        torch.cuda.synchronize()
    array = torch.cat(maps).numpy().astype(np.float16)
    if not np.isfinite(array).all():
        raise ValueError("non-finite dense map detected")
    return array, sample_ids, image_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", choices=("gate", "calib", "modelval"), required=True)
    parser.add_argument("--data-root", default="data/coco2017")
    parser.add_argument("--checkpoint", default="data/checkpoints/ViT-B-16.pt")
    parser.add_argument("--output-root", default="data/features_dense")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--microbatch", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    manifest_path = REPO_ROOT / f"data/manifests/{args.split}.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        rows = rows[: args.limit]
    checkpoint = REPO_ROOT / args.checkpoint
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != config["features"]["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash differs from frozen configuration")
    output_dir = REPO_ROOT / args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config["execution"]["device"])
    microbatch = args.microbatch or (32 if device != "cpu" else 8)
    model = open_clip.load_openai_model(str(checkpoint), device=device, precision="fp32")
    model.eval()
    preprocess = open_clip.image_transform(
        model.visual.image_size,
        is_train=False,
        mean=model.visual.image_mean,
        std=model.visual.image_std,
    )
    relations, prompts = relation_prompts()
    queries = text_queries(model, prompts, device)
    manifest_hash = sha256(manifest_path)
    total_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    completed = 0
    resumed = 0
    started = time.perf_counter()
    for shard_index in range(total_shards):
        shard_rows = rows[shard_index * args.shard_size : (shard_index + 1) * args.shard_size]
        shard_path = output_dir / f"shard_{shard_index:05d}.npz"
        if shard_path.is_file():
            with np.load(shard_path) as existing:
                valid = (
                    existing["maps"].shape[0] == len(shard_rows)
                    and str(existing["checkpoint_sha256"]) == checkpoint_hash
                    and str(existing["manifest_sha256"]) == manifest_hash
                )
            if not valid:
                raise ValueError(f"stale or invalid existing shard: {shard_path}")
            completed += 1
            resumed += 1
            continue
        array, sample_ids, image_hashes = encode_rows(
            shard_rows,
            model,
            preprocess,
            queries,
            len(relations),
            REPO_ROOT / args.data_root,
            device,
            microbatch,
        )
        temporary = shard_path.with_suffix(".part.npz")
        np.savez_compressed(
            temporary,
            sample_id=np.asarray(sample_ids),
            image_sha256=np.asarray(image_hashes),
            maps=array,
            prompts=np.asarray(prompts),
            method=np.asarray("clip_vit_b16_final_patch_text_cosine_fp16"),
            checkpoint_sha256=np.asarray(checkpoint_hash),
            manifest_sha256=np.asarray(manifest_hash),
        )
        temporary.replace(shard_path)
        completed += 1
        (output_dir / "state.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "completed_shards": completed,
                    "total_shards": total_shards,
                    "resumed_shards": resumed,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    first_shape: list[int] | None = None
    if total_shards:
        with np.load(output_dir / "shard_00000.npz") as first:
            first_shape = list(first["maps"].shape[1:])
    index = {
        "schema_version": 1,
        "complete": completed == total_shards,
        "split": args.split,
        "sample_count": len(rows),
        "relation_count": len(relations),
        "map_shape": first_shape,
        "dtype": "float16",
        "shard_size": args.shard_size,
        "shard_count": total_shards,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "prompts": prompts,
        "device": device,
        "microbatch": microbatch,
        "resumed_shards": resumed,
        "elapsed_seconds_this_run": time.perf_counter() - started,
        "test_accessed": False,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
