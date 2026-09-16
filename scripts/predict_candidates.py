#!/usr/bin/env python3
"""Generate raw split predictions from a frozen P2 candidate checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.metrics import candidate_opportunity
from edcr.models import CandidateModel, MultiViewHead, RelationIndex


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["gate", "modelval", "calib", "test"], required=True)
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    run_dir = REPO_ROOT / "runs" / args.run
    checkpoint = torch.load(run_dir / "candidates.pt", map_location="cpu", weights_only=True)
    cache_dir = REPO_ROOT / (args.cache_dir or f"data/features/{args.split}")
    manifest_path = REPO_ROOT / "data/manifests" / f"{args.split}.jsonl"
    sample_ids, features_np, labels = load_feature_shards(cache_dir, manifest_path, 80)
    feature_dim = features_np.shape[-1]
    base_head = MultiViewHead(feature_dim, 80, config["features"]["pooling_temperature"])
    full_head = nn.Linear(feature_dim, 80)
    relations = [RelationIndex(**row) for row in checkpoint["relations"]]
    base_view_indices = tuple(checkpoint.get("base_view_indices", [0, 1, 2, 3, 4]))
    model = CandidateModel(
        base_head,
        full_head,
        relations,
        config["training"]["residual_bound"],
        base_view_indices,
        torch.tensor(checkpoint.get("context_centers", [0.0] * len(relations))),
        torch.tensor(checkpoint.get("context_scales", [1.0] * len(relations))),
        checkpoint.get("context_signal", "probability"),
    )
    model.load_state_dict(checkpoint["candidate"])
    model.freeze_heads()
    model.eval()
    features = torch.from_numpy(features_np)
    batch_size = config["training"]["cached_batch_size"]
    collected = {"B0": [], "B1": [], "B2": []}
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = features[offset : offset + batch_size]
            output = model(batch)
            collected["B0"].append(model.full_head(batch[:, 0]))
            collected["B1"].append(output["base"])
            collected["B2"].append(output["full"])
    logits_by_method: dict[str, np.ndarray] = {}
    for method, chunks in collected.items():
        logits = torch.cat(chunks).numpy().astype(np.float32)
        logits_by_method[method] = logits
        np.savez_compressed(
            run_dir / f"predictions_{method}_{args.split}.npz",
            sample_id=sample_ids,
            logits=logits,
            method=np.asarray(method),
            split=np.asarray(args.split),
        )
    if args.split != "test":
        target_indices = [relation.target for relation in relations]
        opportunity = candidate_opportunity(
            labels,
            logits_by_method["B1"],
            logits_by_method["B2"],
            target_indices,
            float(config["training"]["weak_delta_epsilon"]),
        )
        opportunity["split"] = args.split
        opportunity["target_indices"] = target_indices
        opportunity["pair_ids"] = [
            row["pair_id"]
            for row in json.loads(
                (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
            )
        ]
        (run_dir / f"opportunity_{args.split}.json").write_text(
            json.dumps(opportunity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"run": args.run, "split": args.split, "samples": len(sample_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
