#!/usr/bin/env python3
"""Run the one-shot V6 Open Images test evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import feature_matrices
from run_v4b_action_grid import bce_gain, soft_action
from run_v4d_soft_mechanism_ablation import expected_endpoint_gain
from train_gate import load_candidate
from train_v2b_gain_models import build_tensors

from edcr.cache import load_feature_shards
from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
BOOTSTRAP_REPLICATES = 20_000
RANDOM_SEED = 20260911
FREEZE_PATH = REPO_ROOT / "reports/v6_openimages_test_freeze.json"
PANEL_PATH = REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json"
MAPPING_PATH = REPO_ROOT / "data/manifests/v5_openimages_class_mapping.json"
PREPARATION_PATH = REPO_ROOT / "reports/v6_openimages_test_preparation.json"
MANIFEST_PATH = REPO_ROOT / "data/openimages_v7_manifests/v6_test_unlabeled.jsonl"
DEFAULT_ANNOTATIONS = (
    REPO_ROOT / "data/openimages_v7/oidv7-test-annotations-human-imagelabels.csv"
)
DEFAULT_CACHE = REPO_ROOT / "data/openimages_v7_features/v6_test"
OUTPUT_PATH = REPO_ROOT / "reports/v6_openimages_test_confirmation.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_freeze() -> dict[str, object]:
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze.get("frozen") is not True or freeze.get("test_annotations_accessed") is not False:
        raise RuntimeError("invalid V6 freeze record")
    for relative_path, expected in freeze["artifact_sha256"].items():
        if sha256(REPO_ROOT / relative_path) != expected:
            raise RuntimeError(f"post-freeze artifact change: {relative_path}")
    return freeze


def read_labels(
    annotations: Path, target_mids: set[str]
) -> dict[tuple[str, str], int]:
    values: dict[tuple[str, str], int] = {}
    with annotations.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mid = row["LabelName"]
            if mid not in target_mids:
                continue
            value = int(float(row["Confidence"]))
            if value not in (0, 1):
                raise RuntimeError(f"non-binary confidence: {row['Confidence']}")
            key = (row["ImageID"], mid)
            if key in values and values[key] != value:
                raise RuntimeError(f"conflicting duplicate annotation: {key}")
            values[key] = value
    return values


def bootstrap(values: np.ndarray, pair_ids: list[str], generator) -> dict[str, object]:
    pair_array = np.asarray(pair_ids)
    relations = sorted(set(pair_ids))
    blocks = [np.flatnonzero(pair_array == relation) for relation in relations]
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


def summarize(rows: list[dict[str, object]], generator) -> dict[str, object]:
    pair_ids = [str(row["pair_id"]) for row in rows]
    soft_hard = np.asarray(
        [row["soft_minus_probability_hard"] for row in rows], dtype=np.float64
    )
    hard_direct = np.asarray(
        [row["probability_hard_minus_direct_gain_hard"] for row in rows],
        dtype=np.float64,
    )
    return {
        "soft_minus_probability_hard": bootstrap(soft_hard, pair_ids, generator),
        "probability_hard_minus_direct_gain_hard": bootstrap(
            hard_direct, pair_ids, generator
        ),
        "mean_probability_hard_achieved": float(
            np.mean([row["probability_hard_achieved"] for row in rows])
        ),
        "mean_soft_achieved": float(np.mean([row["soft_achieved"] for row in rows])),
        "mean_direct_gain_hard_achieved": float(
            np.mean([row["direct_gain_hard_achieved"] for row in rows])
        ),
        "mean_constant_achieved": float(
            np.mean([row["constant_achieved"] for row in rows])
        ),
        "mean_endpoint_oracle": float(
            np.mean([row["endpoint_oracle"] for row in rows])
        ),
        "mean_soft_interior_fraction": float(
            np.mean([row["soft_interior_fraction"] for row in rows])
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
    if sha256(args.annotations) != preparation["annotation_sha256"]:
        raise RuntimeError("test annotation file changed after preparation")
    if sha256(MANIFEST_PATH) != preparation["manifest_sha256"]:
        raise RuntimeError("test manifest changed after preparation")
    panel = json.loads(PANEL_PATH.read_text())
    target_mids = {row["target_openimages_mid"] for row in panel}
    labels_by_image_mid = read_labels(args.annotations, target_mids)
    support_by_pair = {row["pair_id"]: row for row in preparation["relation_support"]}
    sample_ids, features_np, sealed_labels = load_feature_shards(
        args.cache, MANIFEST_PATH, 80
    )
    if np.any(sealed_labels):
        raise RuntimeError("test feature manifest unexpectedly exposes labels")
    image_ids = np.asarray(
        [sample_id.removeprefix("oi_v7_test_") for sample_id in sample_ids.astype(str)]
    )
    features = torch.from_numpy(features_np)
    dummy_labels = np.zeros((len(features), 80), dtype=np.float32)
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    mapping = json.loads(MAPPING_PATH.read_text())
    class_mapping = {row["coco_name"]: row for row in mapping["class_mapping"]}
    rows: list[dict[str, object]] = []
    maximum_soft_oracle_excess = -np.inf

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        tensors = build_tensors(candidate, features, dummy_labels)
        feature_sets = feature_matrices(candidate, features, tensors)
        relation_index = {
            (relation.source, relation.target): index
            for index, relation in enumerate(candidate.relations)
        }
        with (
            REPO_ROOT / f"runs/v6_openimages_selectors_seed{seed}/models.pkl"
        ).open("rb") as handle:
            models = pickle.load(handle)
        for relation_row in panel:
            pair_id = relation_row["pair_id"]
            mid = relation_row["target_openimages_mid"]
            relation = relation_index[
                (relation_row["source_coco_id"], relation_row["target_coco_id"])
            ]
            known = np.asarray(
                [(image_id, mid) in labels_by_image_mid for image_id in image_ids]
            )
            if not known.any():
                raise RuntimeError(f"no aligned test labels for {pair_id}")
            labels = np.asarray(
                [labels_by_image_mid.get((image_id, mid), 0) for image_id in image_ids],
                dtype=np.float64,
            )[known]
            support = support_by_pair[pair_id]
            if int(np.sum(labels == 0)) != support["negative"]:
                raise RuntimeError(f"negative test support drift for {pair_id}")
            if int(np.sum(labels == 1)) != support["positive"]:
                raise RuntimeError(f"positive test support drift for {pair_id}")
            base = tensors.base.numpy()[known, relation]
            residual = tensors.residual.numpy()[known, relation]
            endpoint_gain = bce_gain(base, residual, labels, 1.0)
            source_mapping = class_mapping[relation_row["source_coco_name"]]
            target_mapping = class_mapping[relation_row["target_coco_name"]]
            exact_name_relation = (
                source_mapping["mapping_type"] == "exact_name"
                and target_mapping["mapping_type"] == "exact_name"
            )
            for representation in REPRESENTATIONS:
                selected_models = models[representation][pair_id]
                matrix = feature_sets[representation][known, relation]
                probability = selected_models["classifier"].predict_proba(matrix)[:, 1]
                predicted_gain = selected_models["gain_regressor"].predict(matrix)
                q_soft = soft_action(probability, base, residual)
                q_hard = (
                    expected_endpoint_gain(probability, base, residual) > 0
                ).astype(np.float64)
                q_direct = (predicted_gain > 0).astype(np.float64)
                soft_gain = bce_gain(base, residual, labels, q_soft)
                hard_gain = bce_gain(base, residual, labels, q_hard)
                direct_gain = endpoint_gain * q_direct
                maximum_soft_oracle_excess = max(
                    maximum_soft_oracle_excess,
                    float(np.max(soft_gain - np.maximum(endpoint_gain, 0.0))),
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
                        "evaluation_count": int(known.sum()),
                        "evaluation_positive": support["positive"],
                        "evaluation_negative": support["negative"],
                        "probability_hard_achieved": float(hard_gain.mean()),
                        "soft_achieved": float(soft_gain.mean()),
                        "direct_gain_hard_achieved": float(direct_gain.mean()),
                        "constant_achieved": max(float(endpoint_gain.mean()), 0.0),
                        "endpoint_oracle": float(np.maximum(endpoint_gain, 0).mean()),
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

    if maximum_soft_oracle_excess > 1e-8:
        raise RuntimeError(
            "soft action exceeded revealed-label endpoint oracle: "
            f"{maximum_soft_oracle_excess}"
        )
    generator = np.random.default_rng(RANDOM_SEED)
    primary = summarize(rows, generator)
    interval = primary["soft_minus_probability_hard"][
        "relation_cluster_95_percentile_interval"
    ]
    exact_rows = [row for row in rows if row["exact_name_relation"]]
    report = {
        "schema_version": 1,
        "protocol": "v6-openimages-prospective-test-replication",
        "freeze_created_utc": freeze["created_utc"],
        "test_annotations_accessed": True,
        "test_pixels_accessed": True,
        "relation_count": len(panel),
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
        "exact_name_relation_sensitivity": summarize(exact_rows, generator),
        "test_relation_support": preparation["relation_support"],
        "maximum_pointwise_soft_excess_over_endpoint_oracle": float(
            maximum_soft_oracle_excess
        ),
        "shared_endpoint_oracle_difference": 0.0,
        "annotation_sha256": sha256(args.annotations),
        "feature_cache_index_sha256": sha256(args.cache / "index.json"),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
