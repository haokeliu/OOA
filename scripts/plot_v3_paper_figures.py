#!/usr/bin/env python3
"""Generate the main scientific figures for the V3 manuscript."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_v3a_cluster_robustness import cross_validated_prediction

OUTPUT_DIR = REPO_ROOT / "paper/figures"
COLORS = {
    "confidence": "#7A7A7A",
    "summary": "#0072B2",
    "view_logits": "#56B4E9",
    "LogitMLP": "#E69F00",
    "RCVI": "#009E73",
}
REPRESENTATIONS = ("confidence", "summary", "view_logits", "LogitMLP", "RCVI")


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_coco_prediction() -> None:
    report = json.loads(
        (REPO_ROOT / "reports/v3a_observability_law.json").read_text(encoding="utf-8")
    )
    rows = report["rows"]
    target = np.asarray([row["modelval_realized"] for row in rows])
    predictions = (
        cross_validated_prediction(rows, False),
        cross_validated_prediction(rows, True),
    )
    titles = (
        f"Oracle only ($R^2$={report['oracle_only']['r_squared']:.3f})",
        f"Oracle + observable ($R^2$={report['oracle_plus_observable']['r_squared']:.3f})",
    )
    low = min(target.min(), *(prediction.min() for prediction in predictions))
    high = max(target.max(), *(prediction.max() for prediction in predictions))
    padding = 0.04 * (high - low)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True)
    for axis, prediction, title in zip(axes, predictions, titles, strict=True):
        axis.scatter(prediction, target, s=11, alpha=0.28, color="#0072B2", linewidths=0)
        axis.plot([low, high], [low, high], color="#222222", linestyle="--", linewidth=1)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Cross-validated predicted gain")
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
    axes[0].set_ylabel("Independent realized gain")
    fig.suptitle("COCO: held-out observable opportunity predicts selector success", fontsize=11)
    fig.tight_layout()
    save(fig, "v3_coco_prediction")


def external_rows() -> dict[str, list[dict[str, object]]]:
    sources = {
        "VOC2007 test": (
            "reports/v3b_voc_external_confirmation.json",
            "test_realized",
        ),
        "VOC2012 val": (
            "reports/v3c_voc2012_external_confirmation.json",
            "voc2012_realized",
        ),
    }
    output = {}
    for dataset, (path, realized_name) in sources.items():
        report = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
        output[dataset] = [
            {**row, "realized": row[realized_name]} for row in report["rows"]
        ]
    return output


def plot_external_calibration() -> None:
    datasets = external_rows()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), sharex=True, sharey=True)
    for row_index, (dataset, rows) in enumerate(datasets.items()):
        for column, field in enumerate(("checkpoint_oracle", "checkpoint_observable")):
            axis = axes[row_index, column]
            for representation in REPRESENTATIONS:
                selected = [row for row in rows if row["representation"] == representation]
                axis.scatter(
                    [row[field] for row in selected],
                    [row["realized"] for row in selected],
                    label=representation,
                    color=COLORS[representation],
                    s=18,
                    alpha=0.62,
                    linewidths=0,
                )
            x = np.asarray([row[field] for row in rows])
            y = np.asarray([row["realized"] for row in rows])
            correlation = float(__import__("scipy").stats.spearmanr(x, y).statistic)
            low, high = min(x.min(), y.min()), max(x.max(), y.max())
            axis.plot([low, high], [low, high], color="#222222", linestyle="--", linewidth=0.9)
            axis.text(
                0.04,
                0.93,
                f"Spearman={correlation:.3f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
            )
            axis.grid(alpha=0.18, linewidth=0.6)
            axis.set_title(
                f"{dataset}: {'oracle' if column == 0 else 'observable'} checkpoint",
                fontsize=9.5,
            )
        axes[row_index, 0].set_ylabel("External realized gain")
    axes[1, 0].set_xlabel("Checkpoint opportunity")
    axes[1, 1].set_xlabel("Checkpoint opportunity")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=8)
    fig.suptitle("Frozen VOC checkpoints transfer across datasets", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "v3_voc_external_calibration")


def cluster_interval(rows: list[dict[str, object]], representation: str) -> tuple[float, float]:
    relations = sorted({row["pair_id"] for row in rows})
    values = np.asarray(
        [
            np.mean(
                [
                    row["recovered"]
                    for row in rows
                    if row["pair_id"] == relation
                    and row["representation"] == representation
                ]
            )
            for relation in relations
        ]
    )
    generator = np.random.default_rng(20260911)
    samples = values[
        generator.integers(0, len(values), size=(10_000, len(values)))
    ].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def plot_representation_recovery() -> None:
    datasets = external_rows()
    normalized = {}
    for dataset, rows in datasets.items():
        key = "test_recovered_oracle_fraction" if dataset.startswith("VOC2007") else (
            "voc2012_recovered_oracle_fraction"
        )
        normalized[dataset] = [{**row, "recovered": row[key]} for row in rows]
    x = np.arange(len(REPRESENTATIONS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.2, 3.5))
    for dataset_index, (dataset, rows) in enumerate(normalized.items()):
        means = [
            np.mean(
                [row["recovered"] for row in rows if row["representation"] == representation]
            )
            for representation in REPRESENTATIONS
        ]
        intervals = [cluster_interval(rows, representation) for representation in REPRESENTATIONS]
        errors = np.asarray(
            [[mean - low for mean, (low, _high) in zip(means, intervals, strict=True)],
             [high - mean for mean, (_low, high) in zip(means, intervals, strict=True)]]
        )
        positions = x + (dataset_index - 0.5) * width
        axis.bar(
            positions,
            means,
            width,
            label=dataset,
            color=("#0072B2" if dataset_index == 0 else "#E69F00"),
            alpha=0.88,
            yerr=errors,
            capsize=2.5,
            error_kw={"linewidth": 0.8},
        )
    axis.set_xticks(x, REPRESENTATIONS)
    axis.set_ylabel("Mean recovered oracle fraction")
    axis.set_ylim(0, 0.82)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False, fontsize=8.5)
    axis.set_title("Simple evidence summaries recover most observable candidate gain", fontsize=11)
    fig.tight_layout()
    save(fig, "v3_representation_recovery")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    plot_coco_prediction()
    plot_external_calibration()
    plot_representation_recovery()
    print(
        json.dumps(
            {
                "figures": [
                    "v3_coco_prediction",
                    "v3_voc_external_calibration",
                    "v3_representation_recovery",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
