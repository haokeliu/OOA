#!/usr/bin/env python3
"""Train V6 Open Images selectors before test-label access."""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_observable_opportunity import feature_matrices
from run_v4b_action_grid import bce_gain, tree_classifier, tree_regressor
from train_gate import load_candidate
from train_v2b_gain_models import build_tensors

from edcr.cache import load_feature_shards
from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
ANNOTATIONS = (
    REPO_ROOT
    / "data/openimages_v7/oidv7-val-annotations-human-imagelabels.csv"
)
PANEL_PATH = REPO_ROOT / "data/manifests/v6_openimages_relation_panel.json"
MANIFEST_PATH = REPO_ROOT / "data/openimages_v7_manifests/v6_training_unlabeled.jsonl"
CACHE_PATH = REPO_ROOT / "data/openimages_v7_features/v6_training"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_labels(target_mids: set[str]) -> dict[tuple[str, str], int]:
    values: dict[tuple[str, str], int] = {}
    with ANNOTATIONS.open(newline="") as handle:
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


def main() -> int:
    started = time.perf_counter()
    panel = json.loads(PANEL_PATH.read_text())
    target_mids = {row["target_openimages_mid"] for row in panel}
    labels_by_image_mid = read_labels(target_mids)
    sample_ids, features_np, sealed_labels = load_feature_shards(
        CACHE_PATH, MANIFEST_PATH, 80
    )
    if np.any(sealed_labels):
        raise RuntimeError("training feature manifest unexpectedly exposes labels")
    image_ids = np.asarray(
        [sample_id.removeprefix("oi_v7_val_") for sample_id in sample_ids.astype(str)]
    )
    features = torch.from_numpy(features_np)
    dummy_labels = np.zeros((len(features), 80), dtype=np.float32)
    config = load_config(REPO_ROOT / "configs/pilot.yaml")

    for seed in SEEDS:
        output_dir = REPO_ROOT / f"runs/v6_openimages_selectors_seed{seed}"
        output_dir.mkdir(parents=True, exist_ok=False)
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        tensors = build_tensors(candidate, features, dummy_labels)
        feature_sets = feature_matrices(candidate, features, tensors)
        relation_index = {
            (relation.source, relation.target): index
            for index, relation in enumerate(candidate.relations)
        }
        models: dict[str, dict[str, object]] = {
            representation: {} for representation in REPRESENTATIONS
        }
        training_rows = []
        for relation_row in panel:
            pair_id = relation_row["pair_id"]
            mid = relation_row["target_openimages_mid"]
            relation = relation_index[
                (relation_row["source_coco_id"], relation_row["target_coco_id"])
            ]
            known = np.asarray(
                [(image_id, mid) in labels_by_image_mid for image_id in image_ids]
            )
            labels = np.asarray(
                [labels_by_image_mid.get((image_id, mid), 0) for image_id in image_ids],
                dtype=np.float64,
            )
            positive = int(np.sum(labels[known] == 1))
            negative = int(np.sum(labels[known] == 0))
            if positive != relation_row["validation_verified_positive"]:
                raise RuntimeError(f"positive support drift for {pair_id}")
            if negative != relation_row["validation_verified_negative"]:
                raise RuntimeError(f"negative support drift for {pair_id}")
            endpoint_gain = bce_gain(
                tensors.base.numpy()[known, relation],
                tensors.residual.numpy()[known, relation],
                labels[known],
                1.0,
            )
            for representation in REPRESENTATIONS:
                matrix = feature_sets[representation][known, relation]
                classifier = tree_classifier(seed)
                classifier.fit(matrix, labels[known])
                gain_regressor = tree_regressor(seed)
                gain_regressor.fit(matrix, endpoint_gain)
                models[representation][pair_id] = {
                    "classifier": classifier,
                    "gain_regressor": gain_regressor,
                }
                training_rows.append(
                    {
                        "pair_id": pair_id,
                        "representation": representation,
                        "sample_count": int(known.sum()),
                        "positive_count": positive,
                        "negative_count": negative,
                    }
                )
        with (output_dir / "models.pkl").open("wb") as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        report = {
            "schema_version": 1,
            "protocol": "v6-openimages-validation-trained-selectors",
            "test_annotations_accessed": False,
            "test_pixels_accessed": False,
            "seed": seed,
            "relation_count": len(panel),
            "representations": list(REPRESENTATIONS),
            "model_count": len(panel) * len(REPRESENTATIONS) * 2,
            "sklearn_version": sklearn.__version__,
            "training_rows": training_rows,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        provenance = {
            "candidate_checkpoint": REPO_ROOT
            / f"runs/v3a_candidates_seed{seed}/candidates.pt",
            "feature_cache_index": CACHE_PATH / "index.json",
            "training_manifest": MANIFEST_PATH,
            "validation_annotations": ANNOTATIONS,
            "relation_panel": PANEL_PATH,
            "training_code": Path(__file__).resolve(),
        }
        (output_dir / "input_hashes.json").write_text(
            json.dumps(
                {name: sha256(path) for name, path in provenance.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"trained seed={seed}: {len(panel) * len(REPRESENTATIONS)} selector pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
