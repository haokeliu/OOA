#!/usr/bin/env python3
"""Cross-fit a non-contextual candidate-family development study on OI validation."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit, logit
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_gate import load_candidate

from edcr.cache import load_feature_shards
from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("endpoint_confidence", "summary", "view_logits")
FOLD_COUNT = 5
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260912
ANNOTATIONS = REPO_ROOT / "data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"
PANEL_PATH = REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json"
MANIFEST_PATH = REPO_ROOT / "data/openimages_v7_manifests/v6_training_unlabeled.jsonl"
CACHE_PATH = REPO_ROOT / "data/openimages_v7_features/v6_training"
PLAN_PATH = REPO_ROOT / "docs/v7_cross_candidate_development_plan.md"
OUTPUT_PATH = REPO_ROOT / "reports/v7_cross_candidate_development.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_labels(target_mids: set[str]) -> dict[tuple[str, str], int]:
    values: dict[tuple[str, str], int] = {}
    with ANNOTATIONS.open(newline="") as handle:
        for row in csv.DictReader(handle):
            mid = row["LabelName"]
            if mid not in target_mids:
                continue
            value = int(float(row["Confidence"]))
            if value not in (0, 1):
                raise RuntimeError(f"non-binary label: {row['Confidence']}")
            key = (row["ImageID"], mid)
            if key in values and values[key] != value:
                raise RuntimeError(f"conflicting duplicate annotation: {key}")
            values[key] = value
    return values


def stratified_folds(image_ids: np.ndarray, labels: np.ndarray, pair_id: str) -> np.ndarray:
    folds = np.full(len(labels), -1, dtype=np.int64)
    for label_value in (0, 1):
        indices = np.flatnonzero(labels == label_value)
        ranked = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"v7:{pair_id}:{label_value}:{image_ids[index]}".encode()
            ).digest(),
        )
        for rank, index in enumerate(ranked):
            folds[index] = rank % FOLD_COUNT
    if np.any(folds < 0):
        raise RuntimeError("fold assignment incomplete")
    return folds


def classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=7,
        min_samples_leaf=10,
        l2_regularization=1.0,
        random_state=seed,
    )


def feature_sets(
    global_logits: np.ndarray,
    crop_logits: np.ndarray,
    crop_view_logits: np.ndarray,
    source: int,
    target: int,
) -> dict[str, np.ndarray]:
    global_target = global_logits[:, target]
    crop_target = crop_logits[:, target]
    global_source = global_logits[:, source]
    crop_source = crop_logits[:, source]
    target_views = crop_view_logits[:, :, target]
    source_views = crop_view_logits[:, :, source]
    endpoint_confidence = np.column_stack((expit(global_target), expit(crop_target)))
    summary = np.column_stack(
        (
            global_target,
            crop_target,
            crop_target - global_target,
            target_views.std(axis=1),
            global_source,
            crop_source,
        )
    )
    return {
        "endpoint_confidence": endpoint_confidence,
        "summary": summary,
        "view_logits": np.column_stack((summary, target_views, source_views)),
    }


def bce_gain(base: np.ndarray, residual: np.ndarray, labels: np.ndarray, q) -> np.ndarray:
    base_loss = np.logaddexp(0.0, base) - labels * base
    selected = base + q * residual
    return base_loss - (np.logaddexp(0.0, selected) - labels * selected)


def soft_action(probability: np.ndarray, base: np.ndarray, residual: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    action = np.zeros_like(probability, dtype=np.float64)
    stable = np.abs(residual) >= 1e-12
    action[stable] = (logit(probability[stable]) - base[stable]) / residual[stable]
    return np.clip(action, 0.0, 1.0)


def expected_endpoint_gain(
    probability: np.ndarray, base: np.ndarray, residual: np.ndarray
) -> np.ndarray:
    base_risk = np.logaddexp(0.0, base) - probability * base
    endpoint = base + residual
    endpoint_risk = np.logaddexp(0.0, endpoint) - probability * endpoint
    return base_risk - endpoint_risk


def cluster_interval(rows: list[dict[str, object]], generator) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(float(row["soft_minus_probability_hard"]))
    pair_ids = sorted(grouped)
    blocks = [np.asarray(grouped[pair_id], dtype=np.float64) for pair_id in pair_ids]
    values = np.concatenate(blocks)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        chosen = generator.integers(0, len(blocks), size=len(blocks))
        replicates[replicate] = np.concatenate([blocks[index] for index in chosen]).mean()
    return {
        "task_count": len(values),
        "relation_count": len(blocks),
        "mean": float(values.mean()),
        "relation_cluster_95_percentile_interval": np.quantile(
            replicates, (0.025, 0.975)
        ).tolist(),
        "fraction_positive": float(np.mean(replicates > 0)),
    }


def summarize(rows: list[dict[str, object]], generator) -> dict[str, object]:
    result = {"soft_minus_probability_hard": cluster_interval(rows, generator)}
    for key in (
        "mean_corner_endpoint_gain",
        "constant_achieved",
        "probability_hard_achieved",
        "soft_achieved",
        "endpoint_oracle",
        "soft_interior_fraction",
    ):
        result[key] = float(np.mean([float(row[key]) for row in rows]))
    return result


def main() -> int:
    started = time.perf_counter()
    panel = json.loads(PANEL_PATH.read_text())
    labels_by_image_mid = read_labels(
        {row["target_openimages_mid"] for row in panel}
    )
    sample_ids, features_np, sealed_labels = load_feature_shards(
        CACHE_PATH, MANIFEST_PATH, 80
    )
    if np.any(sealed_labels):
        raise RuntimeError("feature manifest unexpectedly exposes labels")
    image_ids = np.asarray(
        [sample_id.removeprefix("oi_v7_val_") for sample_id in sample_ids.astype(str)]
    )
    features = torch.from_numpy(features_np)
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    rows: list[dict[str, object]] = []
    fold_support: dict[str, object] = {}
    maximum_oracle_excess = -np.inf

    for seed in SEEDS:
        candidate = load_candidate(REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config)
        if tuple(candidate.base_view_indices) != (1, 2, 3, 4):
            raise RuntimeError("V7 requires the frozen four-corner candidate head")
        with torch.inference_mode():
            global_logits = candidate.full_head(features[:, 0]).numpy().astype(np.float64)
            crop_view_logits = (
                candidate.base_head.view_logits(features[:, candidate.base_view_indices])
                .numpy()
                .astype(np.float64)
            )
            crop_logits = (
                candidate.base_head(features[:, candidate.base_view_indices])
                .numpy()
                .astype(np.float64)
            )
        relation_index = {
            (relation.source, relation.target): relation for relation in candidate.relations
        }
        for relation_row in panel:
            pair_id = relation_row["pair_id"]
            mid = relation_row["target_openimages_mid"]
            known = np.asarray([(image_id, mid) in labels_by_image_mid for image_id in image_ids])
            known_image_ids = image_ids[known]
            labels = np.asarray(
                [labels_by_image_mid[(image_id, mid)] for image_id in known_image_ids],
                dtype=np.float64,
            )
            folds = stratified_folds(known_image_ids, labels, pair_id)
            if seed == SEEDS[0]:
                fold_support[pair_id] = [
                    {
                        "fold": fold,
                        "negative": int(np.sum((folds == fold) & (labels == 0))),
                        "positive": int(np.sum((folds == fold) & (labels == 1))),
                    }
                    for fold in range(FOLD_COUNT)
                ]
            relation = relation_index[
                (relation_row["source_coco_id"], relation_row["target_coco_id"])
            ]
            matrices = feature_sets(
                global_logits[known],
                crop_logits[known],
                crop_view_logits[known],
                relation.source,
                relation.target,
            )
            base = global_logits[known, relation.target]
            residual = crop_logits[known, relation.target] - base
            endpoint_gain = bce_gain(base, residual, labels, 1.0)
            for representation in REPRESENTATIONS:
                probability = np.empty(len(labels), dtype=np.float64)
                for fold in range(FOLD_COUNT):
                    train = folds != fold
                    evaluate = folds == fold
                    if len(np.unique(labels[train])) != 2:
                        raise RuntimeError(f"one-class training fold for {pair_id}")
                    model = classifier(seed)
                    model.fit(matrices[representation][train], labels[train])
                    probability[evaluate] = model.predict_proba(
                        matrices[representation][evaluate]
                    )[:, 1]
                q_soft = soft_action(probability, base, residual)
                q_hard = (
                    expected_endpoint_gain(probability, base, residual) > 0
                ).astype(np.float64)
                soft_gain = bce_gain(base, residual, labels, q_soft)
                hard_gain = bce_gain(base, residual, labels, q_hard)
                oracle_gain = np.maximum(endpoint_gain, 0.0)
                maximum_oracle_excess = max(
                    maximum_oracle_excess, float(np.max(soft_gain - oracle_gain))
                )
                rows.append(
                    {
                        "seed": seed,
                        "representation": representation,
                        "pair_id": pair_id,
                        "evaluation_count": len(labels),
                        "evaluation_positive": int(np.sum(labels == 1)),
                        "evaluation_negative": int(np.sum(labels == 0)),
                        "mean_corner_endpoint_gain": float(endpoint_gain.mean()),
                        "constant_achieved": max(float(endpoint_gain.mean()), 0.0),
                        "probability_hard_achieved": float(hard_gain.mean()),
                        "soft_achieved": float(soft_gain.mean()),
                        "endpoint_oracle": float(oracle_gain.mean()),
                        "soft_minus_probability_hard": float(
                            soft_gain.mean() - hard_gain.mean()
                        ),
                        "soft_interior_fraction": float(
                            np.mean((q_soft > 0) & (q_soft < 1))
                        ),
                    }
                )

    if maximum_oracle_excess > 1e-10:
        raise RuntimeError(f"soft action exceeded endpoint oracle: {maximum_oracle_excess}")
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    overall = summarize(rows, generator)
    relation_effects = {
        pair_id: float(
            np.mean(
                [
                    row["soft_minus_probability_hard"]
                    for row in rows
                    if row["pair_id"] == pair_id
                ]
            )
        )
        for pair_id in sorted({str(row["pair_id"]) for row in rows})
    }
    relation_values = np.asarray(list(relation_effects.values()), dtype=np.float64)
    positive_relations = int(np.sum(relation_values > 0))
    leave_one_out = np.asarray(
        [
            (relation_values.sum() - value) / (len(relation_values) - 1)
            for value in relation_values
        ]
    )
    report = {
        "schema_version": 1,
        "protocol": "v7-openimages-cross-candidate-development",
        "development_only": True,
        "official_test_accessed_by_v7": False,
        "candidate_family": "full-image-head-versus-four-corner-multiview-head",
        "relation_count": len(panel),
        "task_count": len(rows),
        "encoded_image_count": len(image_ids),
        "fold_count": FOLD_COUNT,
        "overall": overall,
        "positive_relation_count": positive_relations,
        "negative_relation_count": int(np.sum(relation_values < 0)),
        "one_sided_exact_relation_sign_test_p": float(
            binomtest(positive_relations, len(relation_values), 0.5, alternative="greater").pvalue
        ),
        "leave_one_relation_out_mean_range": [
            float(leave_one_out.min()),
            float(leave_one_out.max()),
        ],
        "by_representation": {
            representation: summarize(
                [row for row in rows if row["representation"] == representation], generator
            )
            for representation in REPRESENTATIONS
        },
        "by_seed": {
            str(seed): summarize([row for row in rows if row["seed"] == seed], generator)
            for seed in SEEDS
        },
        "relation_effects": relation_effects,
        "fold_support": fold_support,
        "maximum_pointwise_soft_excess_over_endpoint_oracle": maximum_oracle_excess,
        "input_sha256": {
            "annotations": sha256(ANNOTATIONS),
            "cache_index": sha256(CACHE_PATH / "index.json"),
            "manifest": sha256(MANIFEST_PATH),
            "panel": sha256(PANEL_PATH),
            "plan": sha256(PLAN_PATH),
            "script": sha256(Path(__file__)),
            **{
                f"candidate_seed_{seed}": sha256(
                    REPO_ROOT / f"runs/v3a_candidates_seed{seed}/candidates.pt"
                )
                for seed in SEEDS
            },
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"rows", "fold_support", "relation_effects"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
