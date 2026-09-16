#!/usr/bin/env python3
"""Open VOC 2012 validation labels once and evaluate frozen V3-B selectors."""

from __future__ import annotations

import json
import pickle
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import candidate_gain, feature_matrices, summarize_relation
from evaluate_v3b_voc_external import macro_ap, neural_predictions, safe_spearman, sha256
from prepare_voc2007 import CLASS_NAMES
from train_gate import load_candidate
from train_v2b_gain_models import build_tensors

from edcr.cache import load_feature_shards
from edcr.config import load_config

SEEDS = (17, 29, 43)
TREE_REPRESENTATIONS = ("confidence", "summary", "view_logits")
NEURAL_REPRESENTATIONS = ("LogitMLP", "RCVI")
REPRESENTATIONS = (*TREE_REPRESENTATIONS, *NEURAL_REPRESENTATIONS)


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(
        (REPO_ROOT / "reports/v3c_voc2012_external_freeze.json").read_text(
            encoding="utf-8"
        )
    )
    if freeze.get("frozen") is not True or freeze.get("labels_accessed") is not False:
        raise RuntimeError("invalid V3-C freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def reveal_labels(sample_ids: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    class_to_id = {name: index for index, name in enumerate(CLASS_NAMES)}
    annotation_root = REPO_ROOT / "data/voc2012/VOCdevkit/VOC2012/Annotations"
    labels = np.zeros((len(sample_ids), len(CLASS_NAMES)), dtype=np.float32)
    rows = []
    for row_index, sample_id in enumerate(sample_ids.astype(str)):
        image_id = sample_id.removeprefix("voc2012_")
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
                "label_format": "positive_contiguous_ids_revealed_after_v3c_freeze",
                "split": "val",
            }
        )
    return labels, rows


def main() -> int:
    freeze = verify_freeze()
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    sample_ids, features_np, _sealed_labels = load_feature_shards(
        REPO_ROOT / "data/voc2012_features/val_unlabeled",
        REPO_ROOT / "data/voc2012_manifests/val_unlabeled.jsonl",
        20,
    )
    labels_np, revealed_rows = reveal_labels(sample_ids)
    (REPO_ROOT / "data/voc2012_manifests/val_labeled_after_freeze.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in revealed_rows),
        encoding="utf-8",
    )
    relation_rows = json.loads(
        (REPO_ROOT / "data/voc2007_manifests/selected_relations.json").read_text(
            encoding="utf-8"
        )
    )
    features = torch.from_numpy(features_np)
    rows = []
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
        predictions = {
            representation: np.column_stack(
                [
                    model.predict(feature_sets[representation][:, relation])
                    for relation, model in enumerate(tree_models[representation])
                ]
            )
            for representation in TREE_REPRESENTATIONS
        }
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
            action = predictions[representation] > 0
            mixed_logits = base_logits.copy()
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
                        "voc2012_oracle": external["oracle_improvement"],
                        "voc2012_realized": external["achieved_improvement"],
                        "voc2012_recovered_oracle_fraction": external[
                            "recovered_oracle_fraction"
                        ],
                    }
                )

    pooled_oracle = safe_spearman(
        [row["checkpoint_oracle"] for row in rows],
        [row["voc2012_realized"] for row in rows],
    )
    pooled_observable = safe_spearman(
        [row["checkpoint_observable"] for row in rows],
        [row["voc2012_realized"] for row in rows],
    )
    per_seed = {}
    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        per_seed[str(seed)] = {
            "oracle_to_realized_spearman": safe_spearman(
                [row["checkpoint_oracle"] for row in selected],
                [row["voc2012_realized"] for row in selected],
            ),
            "observable_to_realized_spearman": safe_spearman(
                [row["checkpoint_observable"] for row in selected],
                [row["voc2012_realized"] for row in selected],
            ),
        }
    representation_summary = {}
    for representation in REPRESENTATIONS:
        selected = [row for row in rows if row["representation"] == representation]
        representation_summary[representation] = {
            "mean_achieved_improvement": float(
                np.mean([row["voc2012_realized"] for row in selected])
            ),
            "mean_oracle_improvement": float(
                np.mean([row["voc2012_oracle"] for row in selected])
            ),
            "mean_recovered_oracle_fraction": float(
                np.mean([row["voc2012_recovered_oracle_fraction"] for row in selected])
            ),
        }
    report = {
        "schema_version": 1,
        "protocol": "v3-C-voc2012-prospective-replication",
        "freeze_created_utc": freeze["created_utc"],
        "labels_accessed": True,
        "sample_count": len(sample_ids),
        "task_count": len(rows),
        "pooled_oracle_to_realized_spearman": pooled_oracle,
        "pooled_observable_to_realized_spearman": pooled_observable,
        "pooled_spearman_advantage": pooled_observable - pooled_oracle,
        "per_seed_correlations": per_seed,
        "representation_summary": representation_summary,
        "rcvi_exceeds_summary": (
            representation_summary["RCVI"]["mean_achieved_improvement"]
            > representation_summary["summary"]["mean_achieved_improvement"]
        ),
        "passes_preregistered_primary_rule": (
            pooled_observable - pooled_oracle >= 0.20
            and all(
                values["observable_to_realized_spearman"] > 0
                for values in per_seed.values()
            )
        ),
        "downstream_macro_ap": downstream,
        "rows": rows,
    }
    (REPO_ROOT / "reports/v3c_voc2012_external_confirmation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
