#!/usr/bin/env python3
"""Paired image-cluster bootstrap for the frozen Q12 test comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 29, 43)
METHODS = ("B1", "B5", "M2", "O1")
COMPARISONS = (("M2", "B1"), ("M2", "B5"), ("B5", "B1"), ("O1", "B1"))
METRICS = (
    "selected_macro_ap",
    "macro_cfpr",
    "macro_recall",
    "macro_weak_recall",
    "macro_coco_small_recall",
)
REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_910
CHUNK_SIZE = 50


def weighted_average_precision_batch(
    labels: np.ndarray, scores: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    """Exact sklearn-style weighted AP for many bootstrap count vectors."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.float64)
    sorted_counts = counts[:, order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_scores) != 0) + 1))
    total_by_score = np.add.reduceat(sorted_counts, starts, axis=1)
    positive_by_score = np.add.reduceat(
        sorted_counts * sorted_labels[None, :], starts, axis=1
    )
    cumulative_positive = np.cumsum(positive_by_score, axis=1)
    cumulative_total = np.cumsum(total_by_score, axis=1)
    precision = np.divide(
        cumulative_positive,
        cumulative_total,
        out=np.zeros_like(cumulative_positive),
        where=cumulative_total > 0,
    )
    positive_total = positive_by_score.sum(axis=1)
    return np.divide(
        (positive_by_score * precision).sum(axis=1),
        positive_total,
        out=np.full(len(counts), np.nan),
        where=positive_total > 0,
    )


def masked_rate_batch(
    predictions: np.ndarray, mask: np.ndarray, counts: np.ndarray
) -> np.ndarray:
    denominator = counts @ mask.astype(np.float64)
    numerator = counts @ (predictions & mask).astype(np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(counts), np.nan),
        where=denominator > 0,
    )


def bootstrap_method(
    arrays: dict[str, np.ndarray], method: str, counts: np.ndarray
) -> dict[str, np.ndarray]:
    labels = arrays["labels"].astype(bool)
    scores = arrays[f"scores_{method}"]
    thresholds = arrays[f"thresholds_{method}"]
    predictions = scores >= thresholds[None, :]
    negative = arrays["negative"].astype(bool)
    weak = arrays["weak"].astype(bool)
    weak_coco_small = arrays["weak_coco_small"].astype(bool)
    per_relation: dict[str, list[np.ndarray]] = {metric: [] for metric in METRICS}
    for relation in range(labels.shape[1]):
        relation_labels = labels[:, relation]
        relation_predictions = predictions[:, relation]
        per_relation["selected_macro_ap"].append(
            weighted_average_precision_batch(
                relation_labels, scores[:, relation], counts
            )
        )
        per_relation["macro_cfpr"].append(
            masked_rate_batch(relation_predictions, negative[:, relation], counts)
        )
        per_relation["macro_recall"].append(
            masked_rate_batch(relation_predictions, relation_labels, counts)
        )
        per_relation["macro_weak_recall"].append(
            masked_rate_batch(relation_predictions, weak[:, relation], counts)
        )
        per_relation["macro_coco_small_recall"].append(
            masked_rate_batch(relation_predictions, weak_coco_small[:, relation], counts)
        )
    return {
        metric: np.nanmean(np.stack(values, axis=1), axis=1)
        for metric, values in per_relation.items()
    }


def load_seed(seed: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    run_dir = REPO_ROOT / "runs" / f"p3e_groupsup_seed{seed}"
    with np.load(run_dir / "test_predictions.npz") as source:
        arrays = {key: source[key] for key in source.files}
    report = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    if report["protocol_freeze"] != "Q12" or not report["test_accessed"]:
        raise ValueError(f"seed {seed} is not a completed frozen test run")
    return arrays, report


def main() -> int:
    loaded = [load_seed(seed) for seed in SEEDS]
    reference = loaded[0][0]
    for arrays, _report in loaded[1:]:
        for key in ("sample_id", "labels", "negative", "weak", "weak_coco_small"):
            if not np.array_equal(reference[key], arrays[key]):
                raise ValueError(f"bootstrap cluster alignment differs for {key}")
    sample_count = len(reference["sample_id"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    distributions = {
        f"{method}_minus_{baseline}": {
            metric: np.empty(REPLICATES, dtype=np.float64) for metric in METRICS
        }
        for method, baseline in COMPARISONS
    }
    for offset in range(0, REPLICATES, CHUNK_SIZE):
        size = min(CHUNK_SIZE, REPLICATES - offset)
        draws = rng.integers(0, sample_count, size=(size, sample_count))
        counts = np.zeros((size, sample_count), dtype=np.float64)
        for row, draw in enumerate(draws):
            counts[row] = np.bincount(draw, minlength=sample_count)
        seed_metrics: list[dict[str, dict[str, np.ndarray]]] = []
        for arrays, _report in loaded:
            seed_metrics.append(
                {method: bootstrap_method(arrays, method, counts) for method in METHODS}
            )
        for method, baseline in COMPARISONS:
            comparison = distributions[f"{method}_minus_{baseline}"]
            for metric in METRICS:
                comparison[metric][offset : offset + size] = np.mean(
                    [
                        metrics[method][metric] - metrics[baseline][metric]
                        for metrics in seed_metrics
                    ],
                    axis=0,
                )

    point_means = {
        method: {
            metric: float(
                np.mean([report["methods"][method][metric] for _arrays, report in loaded])
            )
            for metric in METRICS
        }
        for method in METHODS
    }
    rows: list[dict[str, object]] = []
    comparisons: dict[str, object] = {}
    for method, baseline in COMPARISONS:
        comparison_id = f"{method}_minus_{baseline}"
        comparisons[comparison_id] = {}
        for metric in METRICS:
            values = distributions[comparison_id][metric]
            result = {
                "point_difference": point_means[method][metric] - point_means[baseline][metric],
                "bootstrap_mean_difference": float(np.nanmean(values)),
                "ci95_low": float(np.nanpercentile(values, 2.5)),
                "ci95_high": float(np.nanpercentile(values, 97.5)),
            }
            comparisons[comparison_id][metric] = result
            rows.append(
                {
                    "comparison": comparison_id,
                    "metric": metric,
                    **result,
                }
            )
    report = {
        "schema_version": 1,
        "protocol_freeze": "Q12",
        "test_accessed": True,
        "bootstrap_unit": "official val2017 image",
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seed_aggregation": "metric per model seed, then arithmetic mean across seeds",
        "model_seeds": list(SEEDS),
        "point_estimates": point_means,
        "comparisons": comparisons,
    }
    output_json = REPO_ROOT / "reports/final_test_bootstrap.json"
    output_csv = REPO_ROOT / "reports/final_test_bootstrap.csv"
    if output_json.exists() or output_csv.exists():
        raise SystemExit("refusing to overwrite prior frozen bootstrap output")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
