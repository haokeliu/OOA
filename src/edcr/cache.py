"""Verified feature-shard loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_feature_shards(
    cache_dir: str | Path, manifest_path: str | Path, class_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = Path(cache_dir)
    index = json.loads((directory / "index.json").read_text(encoding="utf-8"))
    if index.get("complete") is not True:
        raise ValueError(f"feature cache is incomplete: {directory}")
    embeddings: list[np.ndarray] = []
    sample_ids: list[np.ndarray] = []
    for path in sorted(directory.glob("shard_*.npz")):
        with np.load(path) as shard:
            if str(shard["checkpoint_sha256"]) != index["checkpoint_sha256"]:
                raise ValueError(f"checkpoint mismatch in {path}")
            if str(shard["manifest_sha256"]) != index["manifest_sha256"]:
                raise ValueError(f"manifest mismatch in {path}")
            embeddings.append(shard["embeddings"].astype(np.float32))
            sample_ids.append(shard["sample_id"].astype(str))
    if not embeddings:
        raise ValueError(f"no feature shards found: {directory}")
    features = np.concatenate(embeddings)
    ids = np.concatenate(sample_ids)
    if len(ids) != int(index["sample_count"]):
        raise ValueError("feature count does not match cache index")
    manifest_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        )
    }
    labels = np.zeros((len(ids), class_count), dtype=np.float32)
    for row_index, sample_id in enumerate(ids):
        if sample_id not in manifest_rows:
            raise ValueError(f"cached sample missing from manifest: {sample_id}")
        labels[row_index, manifest_rows[sample_id]["labels"]] = 1.0
    return ids, features, labels
