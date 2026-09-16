#!/usr/bin/env python3
"""Plot V8 calibration metrics and matched baseline gains."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = REPO_ROOT / "reports/v8_calibration_baselines.json"
OUTPUT_ROOT = REPO_ROOT / "paper/figures"


def main() -> int:
    report = json.loads(REPORT_PATH.read_text())
    calibration = report["calibration_summary"]
    baselines = report["baseline_summary"]

    methods = ["raw", "temperature", "platt", "isotonic"]
    method_labels = ["Raw", "Temperature", "Platt", "Isotonic"]
    nll = np.asarray([calibration[name]["mean_nll"] for name in methods])
    ece = np.asarray([calibration[name]["mean_ece_10_bin"] for name in methods])

    baseline_names = [
        "nested_selected_constant_endpoint",
        "nested_selected_static_interpolation",
        "direct_gain_hard",
        "probability_hard_platt",
        "analytic_soft_platt",
        "revealed_label_endpoint_oracle",
        "logistic_endpoint_stacking",
    ]
    baseline_labels = [
        "Selected constant",
        "Static interpolation",
        "Direct-gain hard",
        "Probability-hard Platt",
        "Analytic-soft Platt",
        "Revealed-label oracle",
        "Logistic stacking",
    ]
    values = np.asarray([baselines[name]["gain"]["mean"] for name in baseline_names])
    intervals = np.asarray(
        [baselines[name]["gain"]["relation_cluster_95_percentile_interval"] for name in baseline_names]
    )
    colors = ["#4c78a8"] * 5 + ["#999999", "#e07a5f"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), gridspec_kw={"width_ratios": [0.9, 1.4]})

    x = np.arange(len(methods))
    bars = axes[0].bar(x, nll, color=["#7f8c8d", "#59a14f", "#4c78a8", "#e15759"])
    axes[0].set_xticks(x, method_labels, rotation=18, ha="right")
    axes[0].set_ylabel("Target-label negative log likelihood")
    axes[0].set_title("A. Nested calibration audit", loc="left", fontweight="bold")
    axes[0].set_ylim(0, max(nll) * 1.16)
    for bar, nll_value, ece_value in zip(bars, nll, ece, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            nll_value + 0.025,
            f"NLL {nll_value:.3f}\nECE {ece_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    y = np.arange(len(baseline_names))[::-1]
    errors = np.vstack((values - intervals[:, 0], intervals[:, 1] - values))
    for index, (value, color) in enumerate(zip(values, colors, strict=True)):
        axes[1].errorbar(
            value,
            y[index],
            xerr=errors[:, index : index + 1],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y, baseline_labels)
    axes[1].set_xlabel("BCE gain over full-image endpoint (95% relation-cluster CI)")
    axes[1].set_title("B. Matched baselines and scope controls", loc="left", fontweight="bold")
    axes[1].legend(
        handles=[
            Line2D([0], [0], marker="o", color="#4c78a8", label="Candidate-constrained", lw=0),
            Line2D([0], [0], marker="o", color="#999999", label="Label oracle", lw=0),
            Line2D([0], [0], marker="o", color="#e07a5f", label="Unconstrained", lw=0),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )

    figure.suptitle(
        "V8 development audit: calibration robustness and routing baselines",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_ROOT / "v8_calibration_baselines.png", dpi=220, bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "v8_calibration_baselines.pdf", bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
