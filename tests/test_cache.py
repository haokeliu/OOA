import json

import numpy as np
import pytest

from edcr.cache import load_feature_shards


def test_load_feature_shards_preserves_order_and_builds_labels(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "sample_b", "labels": [1]},
        {"sample_id": "sample_a", "labels": [0, 2]},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    np.savez_compressed(
        cache_dir / "shard_00000.npz",
        sample_id=np.asarray(["sample_b", "sample_a"]),
        embeddings=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        checkpoint_sha256=np.asarray("checkpoint"),
        manifest_sha256=np.asarray("manifest"),
    )
    (cache_dir / "index.json").write_text(
        json.dumps(
            {
                "complete": True,
                "sample_count": 2,
                "checkpoint_sha256": "checkpoint",
                "manifest_sha256": "manifest",
            }
        )
    )

    sample_ids, features, labels = load_feature_shards(cache_dir, manifest, 3)

    assert sample_ids.tolist() == ["sample_b", "sample_a"]
    assert features.shape == (2, 3, 4)
    assert labels.tolist() == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]


def test_load_feature_shards_rejects_incomplete_cache(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="incomplete"):
        load_feature_shards(cache_dir, tmp_path / "unused.jsonl", 3)
