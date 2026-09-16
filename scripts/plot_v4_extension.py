#!/usr/bin/env python3
"""Create the V4 sample-efficiency and soft-action mechanism figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "paper" / "figures"

COLORS = {
    "confidence": "#7A8CA5",
    "summary": "#D9822B",
    "view_logits": "#2878B5",
}
LABELS = {
    "confidence": "Confidence",
    "summary": "Summary",
    "view_logits": "View logits",
}


def main() -> int:
    sample = json.loads(
        (REPO_ROOT / "reports/v3e_tree_sample_efficiency.json").read_text(
            encoding="utf-8"
        )
    )
    mechanism = json.loads(
        (REPO_ROOT / "reports/v4d_soft_mechanism_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    fractions = sample["fractions"]
    counts = [
        sample["summary"][str(value)]["summary"]["train_count"]
        for value in fractions
    ]
    percentages = [100 * value for value in fractions]

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    for representation in ("confidence", "summary", "view_logits"):
        recovered = [
            100
            * sample["summary"][str(fraction)][representation][
                "mean_modelval_recovered_oracle_fraction"
            ]
            for fraction in fractions
        ]
        axes[0].plot(
            percentages,
            recovered,
            marker="o",
            linewidth=2,
            color=COLORS[representation],
            label=LABELS[representation],
        )
    axes[0].set_xticks(
        percentages,
        [f"{percentage:.0f}%\n{count:,}" for percentage, count in zip(percentages, counts)],
    )
    axes[0].set_xlabel("Gate training fraction and image count")
    axes[0].set_ylabel("Mean model-validation oracle recovery (%)")
    axes[0].set_title("a  Richer evidence is more sample-hungry", loc="left", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    comparisons = (
        ("probability_hard_minus_direct_gain_hard", "Probability hard − direct gain hard"),
        ("soft_minus_probability_hard", "Analytic soft − probability hard"),
    )
    positions = np.arange(3)
    width = 0.34
    for comparison_index, (key, label) in enumerate(comparisons):
        means = []
        lower = []
        upper = []
        for representation in ("confidence", "summary", "view_logits"):
            result = mechanism["by_representation"][representation][key]
            means.append(10_000 * result["mean"])
            interval = result["relation_cluster_95_percentile_interval"]
            lower.append(10_000 * (result["mean"] - interval[0]))
            upper.append(10_000 * (interval[1] - result["mean"]))
        offset = (comparison_index - 0.5) * width
        axes[1].bar(
            positions + offset,
            means,
            width,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            color=("#4C78A8", "#F58518")[comparison_index],
            label=label,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, ["Confidence", "Summary", "View logits"])
    axes[1].set_ylabel(r"Paired BCE gain ($\times 10^{-4}$)")
    axes[1].set_title("b  Both risk modeling and soft actions help", loc="left", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(OUTPUT_DIR / f"v4_extension.{extension}", bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
