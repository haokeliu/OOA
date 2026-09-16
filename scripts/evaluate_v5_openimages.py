#!/usr/bin/env python3
"""One-shot evaluation of the frozen V5 Open Images replication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import feature_matrices
from run_v4b_action_grid import bce_gain, soft_action, tree_classifier, tree_regressor
from run_v4d_soft_mechanism_ablation import expected_endpoint_gain
from train_gate import load_candidate
from train_v2b_gain_models import build_tensors

from edcr.cache import load_feature_shards
from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260911
MINIMUM_CLASS_COUNT = 30
MINIMUM_ELIGIBLE_RELATIONS = 8
FREEZE_PATH = REPO_ROOT / "reports/v5_openimages_freeze.json"
MAPPING_PATH = REPO_ROOT / "data/manifests/v5_openimages_class_mapping.json"
PREPARATION_PATH = REPO_ROOT / "reports/v5_openimages_preparation.json"
MANIFEST_PATH = REPO_ROOT / "data/openimages_v7_manifests/eligible_unlabeled.jsonl"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT
    / "data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"
)
DEFAULT_CACHE = REPO_ROOT / "data/openimages_v7_features/eligible"
OUTPUT_PATH = REPO_ROOT / "reports/v5_openimages_confirmation.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def partition(image_id: str) -> str:
    digest = hashlib.sha256(f"v5-oi:{image_id}".encode()).digest()
    return "adaptation" if int.from_bytes(digest[:8], "big") % 2 == 0 else "evaluation"


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("frozen") is not True or freeze.get("annotations_accessed") is not False:
        raise RuntimeError("invalid V5 freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def read_verified_targets(
    annotations: Path, target_mids: set[str]
) -> dict[tuple[str, str], int]:
    values: dict[tuple[str, str], int] = {}
    with annotations.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ImageID", "LabelName", "Confidence"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError(f"unexpected annotation header: {reader.fieldnames}")
        for row in reader:
            mid = row["LabelName"]
            if mid not in target_mids:
                continue
            confidence = float(row["Confidence"])
            if confidence not in (0.0, 1.0):
                raise RuntimeError(f"non-binary confidence: {confidence}")
            key = (row["ImageID"], mid)
            value = int(confidence)
            if key in values and values[key] != value:
                raise RuntimeError(f"conflicting duplicate annotation: {key}")
            values[key] = value
    return values


def bootstrap(values: np.ndarray, pair_ids: list[str], generator) -> dict[str, object]:
    relations = sorted(set(pair_ids))
    blocks = [np.flatnonzero(np.asarray(pair_ids) == relation) for relation in relations]
    replicates = np.empty(BOOTSTRAP_REPLICATES)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        replicates[replicate] = values[indices].mean()
    return {
        "task_count": len(values),
        "relation_count": len(relations),
        "mean": float(values.mean()),
        "relation_cluster_95_percentile_interval": np.quantile(
            replicates, [0.025, 0.975]
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
    }


def summarize(selected_rows: list[dict[str, object]], generator) -> dict[str, object]:
    if not selected_rows:
        return {"task_count": 0, "relation_count": 0}
    pair_ids = [str(row["pair_id"]) for row in selected_rows]
    soft_hard = np.asarray(
        [row["soft_minus_probability_hard"] for row in selected_rows], dtype=np.float64
    )
    hard_direct = np.asarray(
        [
            row["probability_hard_minus_direct_gain_hard"]
            for row in selected_rows
        ],
        dtype=np.float64,
    )
    return {
        "soft_minus_probability_hard": bootstrap(soft_hard, pair_ids, generator),
        "probability_hard_minus_direct_gain_hard": bootstrap(
            hard_direct, pair_ids, generator
        ),
        "mean_probability_hard_achieved": float(
            np.mean([row["probability_hard_achieved"] for row in selected_rows])
        ),
        "mean_soft_achieved": float(
            np.mean([row["soft_achieved"] for row in selected_rows])
        ),
        "mean_direct_gain_hard_achieved": float(
            np.mean([row["direct_gain_hard_achieved"] for row in selected_rows])
        ),
        "mean_constant_achieved": float(
            np.mean([row["constant_achieved"] for row in selected_rows])
        ),
        "mean_endpoint_oracle": float(
            np.mean([row["endpoint_oracle"] for row in selected_rows])
        ),
        "mean_soft_interior_fraction": float(
            np.mean([row["soft_interior_fraction"] for row in selected_rows])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()
    started = time.perf_counter()
    freeze = verify_freeze()
    preparation = json.loads(PREPARATION_PATH.read_text())
    if not preparation["feasibility_passed"]:
        raise RuntimeError("V5 feasibility rule did not pass")
    if preparation["eligible_relation_count"] < MINIMUM_ELIGIBLE_RELATIONS:
        raise RuntimeError("too few eligible relations")
    if sha256(args.annotations) != preparation["annotation_sha256"]:
        raise RuntimeError("annotation file changed after preparation")
    if sha256(MANIFEST_PATH) != preparation["manifest_sha256"]:
        raise RuntimeError("prepared manifest changed")

    mapping = json.loads(MAPPING_PATH.read_text())
    support_by_pair = {
        row["pair_id"]: row
        for row in preparation["relation_support"]
        if row["eligible"]
    }
    relation_rows = [
        row for row in mapping["relation_mapping"] if row["pair_id"] in support_by_pair
    ]
    target_mids = {row["target_openimages_mid"] for row in relation_rows}
    verified = read_verified_targets(args.annotations, target_mids)

    sample_ids, features_np, sealed_labels = load_feature_shards(
        args.cache, MANIFEST_PATH, 80
    )
    if np.any(sealed_labels):
        raise RuntimeError("feature manifest unexpectedly exposes labels")
    image_ids = np.asarray(
        [sample_id.removeprefix("oi_v7_val_") for sample_id in sample_ids.astype(str)]
    )
    partitions = np.asarray([partition(image_id) for image_id in image_ids])
    features = torch.from_numpy(features_np)
    dummy_labels = np.zeros((len(features), 80), dtype=np.float32)
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    class_mapping = {row["coco_name"]: row for row in mapping["class_mapping"]}
    rows: list[dict[str, object]] = []
    maximum_soft_oracle_excess = -np.inf

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        tensors = build_tensors(candidate, features, dummy_labels)
        feature_sets = feature_matrices(candidate, features, tensors)
        base_all = tensors.base.numpy()
        residual_all = tensors.residual.numpy()
        candidate_relation_index = {
            (relation.source, relation.target): index
            for index, relation in enumerate(candidate.relations)
        }
        for relation_row in relation_rows:
            pair_id = relation_row["pair_id"]
            relation = candidate_relation_index[
                (
                    relation_row["source_coco_id"],
                    relation_row["target_coco_id"],
                )
            ]
            mid = relation_row["target_openimages_mid"]
            known = np.asarray([(image_id, mid) in verified for image_id in image_ids])
            labels = np.asarray(
                [verified.get((image_id, mid), 0) for image_id in image_ids],
                dtype=np.float64,
            )
            adaptation = known & (partitions == "adaptation")
            evaluation = known & (partitions == "evaluation")
            for mask_name, mask in (("adaptation", adaptation), ("evaluation", evaluation)):
                values, counts = np.unique(labels[mask], return_counts=True)
                support = dict(zip(values.astype(int), counts, strict=True))
                if min(support.get(0, 0), support.get(1, 0)) < MINIMUM_CLASS_COUNT:
                    raise RuntimeError(f"support drift for {pair_id} on {mask_name}")

            base_adapt = base_all[adaptation, relation]
            residual_adapt = residual_all[adaptation, relation]
            label_adapt = labels[adaptation]
            endpoint_gain_adapt = bce_gain(
                base_adapt, residual_adapt, label_adapt, 1.0
            )
            base_eval = base_all[evaluation, relation]
            residual_eval = residual_all[evaluation, relation]
            label_eval = labels[evaluation]
            endpoint_gain_eval = bce_gain(base_eval, residual_eval, label_eval, 1.0)

            source_mapping = class_mapping[relation_row["source_coco_name"]]
            target_mapping = class_mapping[relation_row["target_coco_name"]]
            exact_name_relation = (
                source_mapping["mapping_type"] == "exact_name"
                and target_mapping["mapping_type"] == "exact_name"
            )
            for representation in REPRESENTATIONS:
                matrix = feature_sets[representation][:, relation]
                classifier = tree_classifier(seed)
                classifier.fit(matrix[adaptation], label_adapt)
                probability = classifier.predict_proba(matrix[evaluation])[:, 1]
                gain_model = tree_regressor(seed)
                gain_model.fit(matrix[adaptation], endpoint_gain_adapt)
                predicted_gain = gain_model.predict(matrix[evaluation])

                q_soft = soft_action(probability, base_eval, residual_eval)
                q_hard = (
                    expected_endpoint_gain(probability, base_eval, residual_eval) > 0
                ).astype(np.float64)
                q_direct = (predicted_gain > 0).astype(np.float64)
                soft_gain = bce_gain(
                    base_eval, residual_eval, label_eval, q_soft
                )
                hard_gain = bce_gain(
                    base_eval, residual_eval, label_eval, q_hard
                )
                direct_gain = endpoint_gain_eval * q_direct
                maximum_soft_oracle_excess = max(
                    maximum_soft_oracle_excess,
                    float(np.max(soft_gain - np.maximum(endpoint_gain_eval, 0.0))),
                )
                rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "pair_id": pair_id,
                        "target_openimages_mid": mid,
                        "target_openimages_name": relation_row[
                            "target_openimages_name"
                        ],
                        "exact_name_relation": exact_name_relation,
                        "adaptation_count": int(adaptation.sum()),
                        "evaluation_count": int(evaluation.sum()),
                        "probability_hard_achieved": float(hard_gain.mean()),
                        "soft_achieved": float(soft_gain.mean()),
                        "direct_gain_hard_achieved": float(direct_gain.mean()),
                        "constant_achieved": max(float(endpoint_gain_eval.mean()), 0.0),
                        "endpoint_oracle": float(np.maximum(endpoint_gain_eval, 0).mean()),
                        "soft_minus_probability_hard": float(
                            soft_gain.mean() - hard_gain.mean()
                        ),
                        "probability_hard_minus_direct_gain_hard": float(
                            hard_gain.mean() - direct_gain.mean()
                        ),
                        "soft_interior_fraction": float(
                            np.mean((q_soft > 0) & (q_soft < 1))
                        ),
                    }
                )

    generator = np.random.default_rng(RANDOM_SEED)
    primary = summarize(rows, generator)
    interval = primary["soft_minus_probability_hard"][
        "relation_cluster_95_percentile_interval"
    ]
    report = {
        "schema_version": 1,
        "protocol": "v5-openimages-cross-domain-continuous-action-replication",
        "freeze_created_utc": freeze["created_utc"],
        "annotations_accessed": True,
        "evaluation_labels_accessed": True,
        "eligible_relation_count": len(relation_rows),
        "encoded_image_count": len(sample_ids),
        "task_count": len(rows),
        "primary": primary,
        "passes_preregistered_primary_rule": (
            primary["soft_minus_probability_hard"]["mean"] > 0
            and interval[0] > 0
        ),
        "by_representation": {
            representation: summarize(
                [row for row in rows if row["representation"] == representation],
                generator,
            )
            for representation in REPRESENTATIONS
        },
        "by_seed": {
            str(seed): summarize(
                [row for row in rows if row["seed"] == seed], generator
            )
            for seed in SEEDS
        },
        "exact_name_relation_sensitivity": summarize(
            [row for row in rows if row["exact_name_relation"]], generator
        ),
        "maximum_pointwise_soft_excess_over_endpoint_oracle": float(
            maximum_soft_oracle_excess
        ),
        "shared_endpoint_oracle_difference": 0.0,
        "annotation_sha256": sha256(args.annotations),
        "feature_cache_index_sha256": sha256(args.cache / "index.json"),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if maximum_soft_oracle_excess > 1e-8:
        raise RuntimeError(
            "soft action exceeded revealed-label endpoint oracle: "
            f"{maximum_soft_oracle_excess}"
        )
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
