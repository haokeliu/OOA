#!/usr/bin/env python3
"""Train the frozen V3-B selector bank on four fifths of VOC gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import (
    aggregate,
    candidate_gain,
    feature_matrices,
    summarize_relation,
)
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    predict_model,
    standardizer,
    train_model,
)

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.observability import LogitGainMLP, RelationConditionedViewGain

TREE_REPRESENTATIONS = ("confidence", "summary", "view_logits")
NEURAL_REPRESENTATIONS = ("LogitMLP", "RCVI")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not registered")

    started = time.perf_counter()
    output_dir = REPO_ROOT / "runs" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_dir = REPO_ROOT / "runs" / args.candidate_run
    candidate = load_candidate(candidate_dir, config)
    manifest_path = REPO_ROOT / "data/voc2007_manifests/gate.jsonl"
    relation_path = REPO_ROOT / "data/voc2007_manifests/selected_relations.json"
    cache_dir = REPO_ROOT / "data/voc2007_features/gate"
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards(
        cache_dir, manifest_path, 20
    )
    gate_features = torch.from_numpy(gate_features_np)
    gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
    gate_gain_np = candidate_gain(gate_tensors)
    gate_gain = torch.from_numpy(gate_gain_np)
    train_mask, checkpoint_mask = internal_split(gate_ids)
    relation_rows = json.loads(relation_path.read_text(encoding="utf-8"))
    feature_sets = feature_matrices(candidate, gate_features, gate_tensors)

    results: dict[str, object] = {}
    tree_models: dict[str, list[HistGradientBoostingRegressor]] = {}
    for representation in TREE_REPRESENTATIONS:
        tree_models[representation] = []
        per_relation = []
        for relation, relation_row in enumerate(relation_rows):
            model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=args.seed,
            )
            model.fit(
                feature_sets[representation][train_mask, relation],
                gate_gain_np[train_mask, relation],
            )
            prediction = model.predict(
                feature_sets[representation][checkpoint_mask, relation]
            )
            tree_models[representation].append(model)
            per_relation.append(
                {
                    "pair_id": relation_row["pair_id"],
                    **summarize_relation(
                        gate_gain_np[checkpoint_mask, relation], prediction
                    ),
                }
            )
        results[representation] = {
            "feature_dimension": int(feature_sets[representation].shape[-1]),
            "checkpoint": aggregate(per_relation),
            "per_relation": per_relation,
        }

    logit_mean, logit_scale = standardizer(feature_sets["view_logits"], train_mask)
    numeric_mean, numeric_scale = standardizer(gate_tensors.numeric.numpy(), train_mask)
    normalized_logits = (feature_sets["view_logits"] - logit_mean) / logit_scale
    normalized_numeric = (gate_tensors.numeric.numpy() - numeric_mean) / numeric_scale
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    source_queries = torch.stack(
        [class_embeddings[relation.source] for relation in candidate.relations]
    )
    target_queries = torch.stack(
        [class_embeddings[relation.target] for relation in candidate.relations]
    )
    relation_count = len(candidate.relations)
    inputs = {
        "logits": torch.from_numpy(normalized_logits).float(),
        "views": gate_features,
        "numeric": torch.from_numpy(normalized_numeric).float(),
        "relation_indices": torch.arange(relation_count),
        "source_queries": source_queries,
        "target_queries": target_queries,
    }
    torch.manual_seed(args.seed)
    neural_models = {
        "LogitMLP": LogitGainMLP(18, relation_count),
        "RCVI": RelationConditionedViewGain(
            gate_features.shape[-1], gate_features.shape[1], gate_tensors.numeric.shape[-1]
        ),
    }
    neural_checkpoints = {}
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
        for representation, model in neural_models.items():
            model, training = train_model(
                representation,
                model,
                train_mask,
                checkpoint_mask,
                inputs,
                gate_gain,
                args.seed,
                log_handle,
            )
            prediction = predict_model(
                representation, model, inputs, len(gate_features), relation_count
            )
            per_relation = [
                {
                    "pair_id": relation_rows[relation]["pair_id"],
                    **summarize_relation(
                        gate_gain_np[checkpoint_mask, relation],
                        prediction[checkpoint_mask, relation],
                    ),
                }
                for relation in range(relation_count)
            ]
            results[representation] = {
                "training": training,
                "checkpoint": aggregate(per_relation),
                "per_relation": per_relation,
            }
            neural_checkpoints[representation] = model.state_dict()

    with (output_dir / "tree_models.pkl").open("wb") as handle:
        pickle.dump(tree_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(
        {
            "models": neural_checkpoints,
            "logit_mean": logit_mean,
            "logit_scale": logit_scale,
            "numeric_mean": numeric_mean,
            "numeric_scale": numeric_scale,
        },
        output_dir / "gain_models.pt",
    )
    report = {
        "schema_version": 1,
        "protocol": "v3-B-voc-selector-freeze",
        "test_labels_accessed": False,
        "seed": args.seed,
        "candidate_run": args.candidate_run,
        "training_split": "gate_internal_4_of_5_hash_residues",
        "checkpoint_split": "gate_internal_1_of_5_hash_residues",
        "train_count": int(train_mask.sum()),
        "checkpoint_count": int(checkpoint_mask.sum()),
        "representations": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "candidate_checkpoint": candidate_dir / "candidates.pt",
        "gate_cache_index": cache_dir / "index.json",
        "gate_manifest": manifest_path,
        "relation_manifest": relation_path,
        "preregistration": REPO_ROOT / "docs/v2_preregistration.md",
        "model_code": REPO_ROOT / "src/edcr/observability.py",
        "training_code": Path(__file__).resolve(),
    }
    (output_dir / "input_hashes.json").write_text(
        json.dumps(
            {name: sha256(path) for name, path in provenance.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
