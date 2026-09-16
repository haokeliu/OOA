#!/usr/bin/env python3
"""Post hoc cluster-structure diagnostics requested during manuscript review.

The analysis does not change any preregistered decision. It reports relation-level
correlations, within-relation rank associations, partial rank associations, and
one-way ICC/effective-sample-size summaries for the existing COCO and VOC rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).statistic)


def indicators(values: list[str]) -> np.ndarray:
    levels = sorted(set(values))
    return np.asarray([[float(value == level) for level in levels[1:]] for value in values])


def residual_rank_correlation(
    x: np.ndarray, y: np.ndarray, control_columns: list[np.ndarray]
) -> float:
    design = np.column_stack([np.ones(len(x)), *control_columns])
    ranked_x = rankdata(x)
    ranked_y = rankdata(y)
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    return float(np.corrcoef(residual_x, residual_y)[0, 1])


def icc_and_effective_n(values: np.ndarray, clusters: list[str]) -> dict[str, float]:
    unique = sorted(set(clusters))
    groups = [values[np.asarray([cluster == key for cluster in clusters])] for key in unique]
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Balanced clusters are required for this diagnostic")
    m = len(groups[0])
    grand = float(np.mean(values))
    between = m * sum((float(np.mean(group)) - grand) ** 2 for group in groups) / (len(groups) - 1)
    within = sum(float(np.sum((group - np.mean(group)) ** 2)) for group in groups) / (
        len(values) - len(groups)
    )
    icc = (between - within) / (between + (m - 1) * within)
    effective_n = len(values) / (1 + (m - 1) * max(icc, 0.0))
    return {"icc_1_1": float(icc), "design_effect_effective_n": float(effective_n)}


def relation_means(rows: list[dict], x_name: str, y_name: str) -> tuple[np.ndarray, np.ndarray]:
    relations = sorted({str(row["pair_id"]) for row in rows})
    x = np.asarray([np.mean([row[x_name] for row in rows if row["pair_id"] == relation]) for relation in relations])
    y = np.asarray([np.mean([row[y_name] for row in rows if row["pair_id"] == relation]) for relation in relations])
    return x, y


def dataset_diagnostics(rows: list[dict], realized_name: str) -> dict:
    x = np.asarray([row["checkpoint_observable"] for row in rows], dtype=float)
    oracle = np.asarray([row["checkpoint_oracle"] for row in rows], dtype=float)
    y = np.asarray([row[realized_name] for row in rows], dtype=float)
    relations = [str(row["pair_id"]) for row in rows]
    relation_x, relation_y = relation_means(rows, "checkpoint_observable", realized_name)
    relation_o, _ = relation_means(rows, "checkpoint_oracle", realized_name)
    relation_fe = indicators(relations)
    seed_fe = indicators([str(row["seed"]) for row in rows])
    representation_fe = indicators([str(row["representation"]) for row in rows])
    return {
        "relation_count": len(set(relations)),
        "task_count": len(rows),
        "relation_mean_spearman_achieved": spearman(relation_x, relation_y),
        "relation_mean_spearman_oracle": spearman(relation_o, relation_y),
        "within_relation_rank_correlation": residual_rank_correlation(x, y, [relation_fe]),
        "within_relation_adjusted_rank_correlation": residual_rank_correlation(
            x, y, [relation_fe, seed_fe, representation_fe, rankdata(oracle)]
        ),
        "achieved_gain_cluster_structure": icc_and_effective_n(x, relations),
        "realized_gain_cluster_structure": icc_and_effective_n(y, relations),
    }


def main() -> int:
    coco = json.loads((ROOT / "reports/v3a_observability_law.json").read_text())["rows"]
    voc2007 = json.loads((ROOT / "reports/v3b_voc_external_confirmation.json").read_text())["rows"]
    voc2012 = json.loads((ROOT / "reports/v3c_voc2012_external_confirmation.json").read_text())["rows"]

    coco_result = dataset_diagnostics(coco, "modelval_realized")
    x = np.asarray([row["checkpoint_observable"] for row in coco], dtype=float)
    y = np.asarray([row["modelval_realized"] for row in coco], dtype=float)
    controls = [
        rankdata(np.asarray([row["checkpoint_constant"] for row in coco], dtype=float)),
        rankdata(np.asarray([row["checkpoint_oracle"] for row in coco], dtype=float)),
        rankdata(np.asarray([row["smoothed_lift"] for row in coco], dtype=float)),
        indicators([str(row["seed"]) for row in coco]),
        indicators([str(row["representation"]) for row in coco]),
    ]
    coco_result["partial_rank_controlling_C_O_lift_seed_representation"] = residual_rank_correlation(
        x, y, controls
    )

    report = {
        "schema_version": 1,
        "analysis_status": "post hoc response-to-review diagnostic; preregistered rules unchanged",
        "coco": coco_result,
        "voc2007": dataset_diagnostics(voc2007, "test_realized"),
        "voc2012": dataset_diagnostics(voc2012, "voc2012_realized"),
        "notes": [
            "Relation-mean correlations use one row per relation.",
            "Within-relation values are Pearson correlations between residualized marginal ranks.",
            "Effective n is the balanced-cluster design-effect diagnostic N/[1+(m-1)max(ICC,0)].",
            "These descriptive post hoc diagnostics do not replace preregistered relation-block tests.",
        ],
    }
    output = ROOT / "reports/submission_cluster_structure.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
