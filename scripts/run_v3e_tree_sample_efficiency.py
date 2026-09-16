#!/usr/bin/env python3
"""Measure fixed tree-selector sample efficiency on COCO development splits."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_v3_observability_law import opportunity
from audit_observable_opportunity import candidate_gain, feature_matrices
from train_gate import load_candidate
from train_v2b_gain_models import (
    build_tensors,
    internal_split,
    load_feature_shards_with_ids,
)

from edcr.config import load_config

SEEDS = (17, 29, 43)
REPRESENTATIONS = ("confidence", "summary", "view_logits")
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def stable_uniform(sample_id: str) -> float:
    value = int.from_bytes(
        hashlib.sha256(f"v3e-sample:{sample_id}".encode()).digest()[:8]
    )
    return value / 2**64


def correlation(first: list[float], second: list[float]) -> float:
    return float(spearmanr(first, second).statistic)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "task_count": len(rows),
        "train_count": int(rows[0]["train_count"]),
        "checkpoint_to_modelval_spearman": correlation(
            [float(row["checkpoint_observable"]) for row in rows],
            [float(row["modelval_realized"]) for row in rows],
        ),
        "mean_checkpoint_achieved": float(
            np.mean([row["checkpoint_observable"] for row in rows])
        ),
        "mean_modelval_achieved": float(
            np.mean([row["modelval_realized"] for row in rows])
        ),
        "mean_modelval_recovered_oracle_fraction": float(
            np.mean([row["modelval_recovered_fraction"] for row in rows])
        ),
    }


def main() -> int:
    started = time.perf_counter()
    config = load_config(REPO_ROOT / "configs/pilot.yaml")
    relation_rows = json.loads(
        (REPO_ROOT / "data/manifests/v3_relation_panel.json").read_text(
            encoding="utf-8"
        )
    )
    gate_ids, gate_features_np, gate_labels_np = load_feature_shards_with_ids("gate")
    _modelval_ids, modelval_features_np, modelval_labels_np = (
        load_feature_shards_with_ids("modelval")
    )
    gate_features = torch.from_numpy(gate_features_np)
    modelval_features = torch.from_numpy(modelval_features_np)
    outer_train, outer_checkpoint = internal_split(gate_ids)
    uniform = np.asarray([stable_uniform(str(sample_id)) for sample_id in gate_ids])
    rows: list[dict[str, object]] = []

    for seed in SEEDS:
        candidate = load_candidate(
            REPO_ROOT / f"runs/v3a_candidates_seed{seed}", config
        )
        gate_tensors = build_tensors(candidate, gate_features, gate_labels_np)
        modelval_tensors = build_tensors(
            candidate, modelval_features, modelval_labels_np
        )
        gate_gain = candidate_gain(gate_tensors)
        modelval_gain = candidate_gain(modelval_tensors)
        gate_sets = feature_matrices(candidate, gate_features, gate_tensors)
        modelval_sets = feature_matrices(
            candidate, modelval_features, modelval_tensors
        )
        for fraction in FRACTIONS:
            train_mask = outer_train & (uniform < fraction)
            for representation in REPRESENTATIONS:
                for relation, relation_row in enumerate(relation_rows):
                    model = HistGradientBoostingRegressor(
                        loss="squared_error",
                        learning_rate=0.05,
                        max_iter=100,
                        max_leaf_nodes=15,
                        min_samples_leaf=40,
                        l2_regularization=1.0,
                        random_state=seed,
                    )
                    model.fit(
                        gate_sets[representation][train_mask, relation],
                        gate_gain[train_mask, relation],
                    )
                    checkpoint_prediction = model.predict(
                        gate_sets[representation][outer_checkpoint, relation]
                    )
                    modelval_prediction = model.predict(
                        modelval_sets[representation][:, relation]
                    )
                    checkpoint = opportunity(
                        gate_gain[outer_checkpoint, relation], checkpoint_prediction
                    )
                    modelval = opportunity(
                        modelval_gain[:, relation], modelval_prediction
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "fraction": fraction,
                            "train_count": int(train_mask.sum()),
                            "representation": representation,
                            "pair_id": relation_row["pair_id"],
                            "checkpoint_oracle": checkpoint["oracle_improvement"],
                            "checkpoint_observable": checkpoint[
                                "achieved_improvement"
                            ],
                            "modelval_oracle": modelval["oracle_improvement"],
                            "modelval_realized": modelval["achieved_improvement"],
                            "modelval_recovered_fraction": modelval[
                                "recovered_oracle_fraction"
                            ],
                        }
                    )

    summaries = {}
    for fraction in FRACTIONS:
        summaries[str(fraction)] = {
            representation: summarize(
                [
                    row
                    for row in rows
                    if row["fraction"] == fraction
                    and row["representation"] == representation
                ]
            )
            for representation in REPRESENTATIONS
        }
    report = {
        "schema_version": 1,
        "protocol": "v3-E-tree-sample-efficiency",
        "development_only": True,
        "test_accessed": False,
        "outer_checkpoint_labels_used_for_training_or_selection": False,
        "fractions": list(FRACTIONS),
        "representations": list(REPRESENTATIONS),
        "outer_train_count": int(outer_train.sum()),
        "outer_checkpoint_count": int(outer_checkpoint.sum()),
        "summary": summaries,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (REPO_ROOT / "reports/v3e_tree_sample_efficiency.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
