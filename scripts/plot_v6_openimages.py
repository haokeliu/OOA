#!/usr/bin/env python3
"""Plot relation effects and frozen subgroup intervals for V6."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports/v6_openimages_test_confirmation.json"
ROBUSTNESS_PATH = REPO_ROOT / "reports/v6_openimages_robustness.json"
OUTPUT_ROOT = REPO_ROOT / "paper/figures"


def interval(summary: dict) -> tuple[float, float, float]:
    effect = summary["soft_minus_probability_hard"]
    return effect["mean"], *effect["relation_cluster_95_percentile_interval"]


def main() -> int:
    report = json.loads(REPORT_PATH.read_text())
    robustness = json.loads(ROBUSTNESS_PATH.read_text())
    relation_rows = robustness["relation_rows"]
    labels = [row["pair_id"].replace("_to_", " → ").replace("_", " ") for row in relation_rows]
    effects = np.asarray([row["mean_soft_minus_probability_hard"] for row in relation_rows])

    forest = [
        ("Overall", interval(report["primary"])),
        ("Exact-name relations", interval(report["exact_name_relation_sensitivity"])),
        ("Confidence", interval(report["by_representation"]["confidence"])),
        ("Summary", interval(report["by_representation"]["summary"])),
        ("View logits", interval(report["by_representation"]["view_logits"])),
        ("Seed 17", interval(report["by_seed"]["17"])),
        ("Seed 29", interval(report["by_seed"]["29"])),
        ("Seed 43", interval(report["by_seed"]["43"])),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.6), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(labels))
    colors = np.where(effects >= 0, "#277da1", "#d95f59")
    axes[0].barh(y, effects, color=colors, alpha=0.9)
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].set_xlabel("Mean test BCE gain: soft − probability-hard")
    axes[0].set_title("A. Relation-level effects (9 frozen tasks each)", loc="left", fontweight="bold")
    axes[0].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[0].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))

    forest_labels = [row[0] for row in forest]
    forest_values = np.asarray([row[1] for row in forest])
    fy = np.arange(len(forest_labels))[::-1]
    means = forest_values[:, 0]
    errors = np.vstack((means - forest_values[:, 1], forest_values[:, 2] - means))
    axes[1].errorbar(
        means,
        fy,
        xerr=errors,
        fmt="o",
        color="#277da1",
        ecolor="#5c677d",
        capsize=3,
        markersize=6,
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(fy, forest_labels)
    axes[1].set_xlabel("Soft − probability-hard BCE gain (95% cluster CI)")
    axes[1].set_title("B. Frozen and sensitivity summaries", loc="left", fontweight="bold")
    axes[1].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(-4, -4))

    figure.suptitle(
        "Prospective Open Images test confirmation of continuous uncertainty hedging",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "v6_openimages_external.png", dpi=220, bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "v6_openimages_external.pdf", bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
