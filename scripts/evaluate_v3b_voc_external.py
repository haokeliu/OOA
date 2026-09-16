#!/usr/bin/env python3
"""Open VOC test labels once and evaluate the already-frozen V3-B study."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import candidate_gain, feature_matrices, summarize_relation
from prepare_voc2007 import CLASS_NAMES
from train_gate import load_candidate
from train_v2b_gain_models import build_tensors, predict_model

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.observability import LogitGainMLP, RelationConditionedViewGain

SEEDS = (17, 29, 43)
TREE_REPRESENTATIONS = ("confidence", "summary", "view_logits")
NEURAL_REPRESENTATIONS = ("LogitMLP", "RCVI")
REPRESENTATIONS = (*TREE_REPRESENTATIONS, *NEURAL_REPRESENTATIONS)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_freeze() -> dict[str, object]:
    freeze_path = REPO_ROOT / "reports/v3b_voc_external_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("frozen") is not True or freeze.get("test_labels_accessed") is not False:
        raise RuntimeError("invalid external-confirmation freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        path = REPO_ROOT / relative_path
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def reveal_labels(sample_ids: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    annotation_root = REPO_ROOT / "data/voc2007/VOCdevkit/VOC2007/Annotations"
    labels = np.zeros((len(sample_ids), len(CLASS_NAMES)), dtype=np.float32)
    rows = []
    for row_index, sample_id in enumerate(sample_ids.astype(str)):
        image_id = sample_id.removeprefix("voc2007_")
        root = ET.parse(annotation_root / f"{image_id}.xml").getroot()
        names = {node.text for node in root.findall("object/name")}
        unknown = names.difference(class_to_id)
        if unknown:
            raise RuntimeError(f"unknown VOC class in {image_id}: {sorted(unknown)}")
        positive = sorted(class_to_id[name] for name in names)
        labels[row_index, positive] = 1.0
        rows.append(
            {
                "sample_id": sample_id,
                "labels": positive,
                "label_format": "positive_contiguous_ids_revealed_after_v3b_freeze",
                "split": "test",
            }
        )
    return labels, rows


def safe_spearman(x: list[float], y: list[float]) -> float:
    value = float(spearmanr(x, y).statistic)
    if not np.isfinite(value):
        raise RuntimeError("external Spearman correlation is undefined")
    return value


def macro_ap(labels: np.ndarray, logits: np.ndarray, indices: list[int]) -> float:
    return float(
        np.mean(
            [average_precision_score(labels[:, index], logits[:, index]) for index in indices]
        )
    )


def neural_predictions(
    candidate,
    features: torch.Tensor,
    tensors,
    checkpoint: dict[str, object],
) -> dict[str, np.ndarray]:
    relation_count = len(candidate.relations)
    sets = feature_matrices(candidate, features, tensors)
    logits = (sets["view_logits"] - checkpoint["logit_mean"]) / checkpoint["logit_scale"]
    numeric = (tensors.numeric.numpy() - checkpoint["numeric_mean"]) / checkpoint[
        "numeric_scale"
    ]
    class_embeddings = functional.normalize(candidate.base_head.linear.weight.detach(), dim=-1)
    inputs = {
        "logits": torch.from_numpy(logits).float(),
        "views": features,
        "numeric": torch.from_numpy(numeric).float(),
        "relation_indices": torch.arange(relation_count),
        "source_queries": torch.stack(
            [class_embeddings[relation.source] for relation in candidate.relations]
        ),
        "target_queries": torch.stack(
            [class_embeddings[relation.target] for relation in candidate.relations]
        ),
    }
    models = {
        "LogitMLP": LogitGainMLP(18, relation_count),
        "RCVI": RelationConditionedViewGain(
            features.shape[-1], features.shape[1], tensors.numeric.shape[-1]
        ),
    }
    output = {}
    for name, model in models.items():
        model.load_state_dict(checkpoint["models"][name])
        output[name] = predict_model(name, model, inputs, len(features), relation_count)
    return output


def main() -> int:
    freeze = verify_freeze()
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    manifest_path = REPO_ROOT / "data/voc2007_manifests/test_unlabeled.jsonl"
    cache_dir = REPO_ROOT / "data/voc2007_features/test_unlabeled"
    sample_ids, features_np, _sealed_labels = load_feature_shards(
        cache_dir, manifest_path, 20
    )
    labels_np, revealed_rows = reveal_labels(sample_ids)
    revealed_path = REPO_ROOT / "data/voc2007_manifests/test_labeled_after_freeze.jsonl"
    revealed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in revealed_rows),
        encoding="utf-8",
    )
    relation_rows = json.loads(
        (REPO_ROOT / "data/voc2007_manifests/selected_relations.json").read_text(
            encoding="utf-8"
        )
    )
    features = torch.from_numpy(features_np)
    rows: list[dict[str, object]] = []
    downstream = []
    for seed in SEEDS:
        candidate = load_candidate(REPO_ROOT / f"runs/v3b_voc_candidates_seed{seed}", config)
        tensors = build_tensors(candidate, features, labels_np)
        gain = candidate_gain(tensors)
        feature_sets = feature_matrices(candidate, features, tensors)
        selector_dir = REPO_ROOT / f"runs/v3b_voc_selectors_seed{seed}"
        checkpoint_report = json.loads(
            (selector_dir / "metrics.json").read_text(encoding="utf-8")
        )
        with (selector_dir / "tree_models.pkl").open("rb") as handle:
            tree_models = pickle.load(handle)
        predictions: dict[str, np.ndarray] = {}
        for representation in TREE_REPRESENTATIONS:
            predictions[representation] = np.column_stack(
                [
                    model.predict(feature_sets[representation][:, relation])
                    for relation, model in enumerate(tree_models[representation])
                ]
            )
        neural_checkpoint = torch.load(
            selector_dir / "gain_models.pt", map_location="cpu", weights_only=False
        )
        predictions.update(
            neural_predictions(candidate, features, tensors, neural_checkpoint)
        )
        with torch.inference_mode():
            candidate_output = candidate(features)
        base_logits = candidate_output["base"].numpy()
        full_logits = candidate_output["full"].numpy()
        target_ids = [relation.target for relation in candidate.relations]
        for representation in REPRESENTATIONS:
            mixed_logits = base_logits.copy()
            action = predictions[representation] > 0
            for relation, target in enumerate(target_ids):
                mixed_logits[:, target] = np.where(
                    action[:, relation], full_logits[:, target], base_logits[:, target]
                )
            downstream.append(
                {
                    "seed": seed,
                    "representation": representation,
                    "selected_macro_ap": macro_ap(labels_np, mixed_logits, target_ids),
                    "all_class_macro_ap": macro_ap(
                        labels_np, mixed_logits, list(range(len(CLASS_NAMES)))
                    ),
                }
            )
            checkpoint_rows = checkpoint_report["representations"][representation][
                "per_relation"
            ]
            for relation, relation_row in enumerate(relation_rows):
                external = summarize_relation(
                    gain[:, relation], predictions[representation][:, relation]
                )
                checkpoint = checkpoint_rows[relation]
                rows.append(
                    {
                        "seed": seed,
                        "pair_id": relation_row["pair_id"],
                        "representation": representation,
                        "checkpoint_oracle": checkpoint["oracle_improvement"],
                        "checkpoint_observable": checkpoint["achieved_improvement"],
                        "test_oracle": external["oracle_improvement"],
                        "test_realized": external["achieved_improvement"],
                        "test_recovered_oracle_fraction": external[
                            "recovered_oracle_fraction"
                        ],
                        "test_recovered_beyond_constant_fraction": external[
                            "recovered_beyond_constant_fraction"
                        ],
                    }
                )

    pooled_oracle = safe_spearman(
        [float(row["checkpoint_oracle"]) for row in rows],
        [float(row["test_realized"]) for row in rows],
    )
    pooled_observable = safe_spearman(
        [float(row["checkpoint_observable"]) for row in rows],
        [float(row["test_realized"]) for row in rows],
    )
    per_seed = {}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        per_seed[str(seed)] = {
            "oracle_to_test_realized_spearman": safe_spearman(
                [float(row["checkpoint_oracle"]) for row in selected],
                [float(row["test_realized"]) for row in selected],
            ),
            "observable_to_test_realized_spearman": safe_spearman(
                [float(row["checkpoint_observable"]) for row in selected],
                [float(row["test_realized"]) for row in selected],
            ),
        }
    representation_summary = {}
    for representation in REPRESENTATIONS:
        selected = [row for row in rows if row["representation"] == representation]
        representation_summary[representation] = {
            "mean_test_achieved_improvement": float(
                np.mean([row["test_realized"] for row in selected])
            ),
            "mean_test_oracle_improvement": float(
                np.mean([row["test_oracle"] for row in selected])
            ),
            "mean_test_recovered_oracle_fraction": float(
                np.mean([row["test_recovered_oracle_fraction"] for row in selected])
            ),
        }
    primary_pass = (
        all(
            values["observable_to_test_realized_spearman"] > 0
            for values in per_seed.values()
        )
        and pooled_observable - pooled_oracle >= 0.20
    )
    report = {
        "schema_version": 1,
        "protocol": "v3-B-voc-external-confirmation",
        "freeze_created_utc": freeze["created_utc"],
        "test_labels_accessed": True,
        "test_sample_count": len(sample_ids),
        "task_count": len(rows),
        "pooled_oracle_to_test_realized_spearman": pooled_oracle,
        "pooled_observable_to_test_realized_spearman": pooled_observable,
        "pooled_spearman_advantage": pooled_observable - pooled_oracle,
        "per_seed_correlations": per_seed,
        "representation_summary": representation_summary,
        "rcvi_exceeds_summary": (
            representation_summary["RCVI"]["mean_test_achieved_improvement"]
            > representation_summary["summary"]["mean_test_achieved_improvement"]
        ),
        "passes_preregistered_primary_rule": primary_pass,
        "downstream_macro_ap": downstream,
        "rows": rows,
    }
    output_path = REPO_ROOT / "reports/v3b_voc_external_confirmation.json"
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (REPO_ROOT / "reports/v3b_voc_external_tasks.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
