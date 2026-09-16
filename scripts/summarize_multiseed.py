#!/usr/bin/env python3
"""Aggregate the frozen P2e/P3e development results across registered seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (17, 29, 43)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--candidate-prefix", default="p2e_seed")
    parser.add_argument("--gate-prefix", default="p3e_groupsup_seed")
    parser.add_argument("--output-stem", default="p2e_p3e_multiseed_summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        candidate_run = f"{args.candidate_prefix}{seed}"
        gate_run = f"{args.gate_prefix}{seed}"
        candidate = load_json(REPO_ROOT / "runs" / candidate_run / "metrics.json")
        gate = load_json(REPO_ROOT / "runs" / gate_run / "metrics.json")
        operating = load_json(
            REPO_ROOT / "runs" / gate_run / "modelval_operating_metrics.json"
        )
        probes = load_json(REPO_ROOT / "runs" / f"q11_probe_seed{seed}" / "metrics.json")
        if candidate["seed"] != seed or gate["seed"] != seed:
            raise ValueError(f"seed provenance mismatch for seed {seed}")
        if operating["test_accessed"] or gate["test_accessed"]:
            raise ValueError(f"test-access flag set in development run for seed {seed}")
        if probes["test_accessed"]:
            raise ValueError(f"test-access flag set in diagnostic probe for seed {seed}")

        opportunity = candidate["opportunity"]
        seed_rows.append(
            {
                "seed": seed,
                "candidate_run": candidate_run,
                "gate_run": gate_run,
                "oracle_relative_bce_improvement": opportunity["oracle_relative_improvement"],
                "beneficial_fraction": opportunity["beneficial_fraction"],
                "harmful_fraction": opportunity["harmful_fraction"],
                "opportunity_pass": opportunity["passes_preregistered_threshold"],
                "p3_pass": operating["p3_ap_pass"],
            }
        )
        for method, metrics in gate["methods"].items():
            training_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "feasible": metrics["feasible"],
                    "constraint_violation": metrics["constraint_violation"],
                    "training_selected_macro_ap": metrics["selected_macro_ap"],
                    "negative_margin": metrics["negative_risk"] - metrics["negative_limit"],
                    "weak_margin": metrics["weak_risk"] - metrics["weak_limit"],
                }
            )
        for method, metrics in operating["methods"].items():
            diagnostic = operating["gate_behavior_diagnostics"].get(method)
            operating_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "diagnostic_only": method in operating["diagnostic_only_methods"],
                    "selected_macro_ap": metrics["selected_macro_ap"],
                    "macro_cfpr": metrics["macro_cfpr"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_weak_recall": metrics["macro_weak_recall"],
                    "macro_coco_small_recall": metrics["macro_coco_small_recall"],
                    "gain_target_weighted_auroc": (
                        diagnostic["gain_target_weighted_auroc"] if diagnostic else None
                    ),
                }
            )
        for kind, metrics in probes["probes"].items():
            probe_rows.append(
                {
                    "seed": seed,
                    "probe": kind,
                    "macro_auroc": metrics["macro_auroc"],
                    "macro_average_precision": metrics["macro_average_precision"],
                }
            )

    methods = sorted({row["method"] for row in operating_rows})
    operating_summary: dict[str, Any] = {}
    for method in methods:
        rows = [row for row in operating_rows if row["method"] == method]
        operating_summary[method] = {
            field: summarize([float(row[field]) for row in rows])
            for field in (
                "selected_macro_ap",
                "macro_cfpr",
                "macro_recall",
                "macro_weak_recall",
                "macro_coco_small_recall",
            )
        }
        aurocs = [
            float(row["gain_target_weighted_auroc"])
            for row in rows
            if row["gain_target_weighted_auroc"] is not None
        ]
        if aurocs:
            operating_summary[method]["gain_target_weighted_auroc"] = summarize(aurocs)

    training_methods = sorted({row["method"] for row in training_rows})
    training_summary: dict[str, Any] = {}
    for method in training_methods:
        rows = [row for row in training_rows if row["method"] == method]
        training_summary[method] = {
            "feasible_seed_count": sum(bool(row["feasible"]) for row in rows),
            "constraint_violation": summarize(
                [float(row["constraint_violation"]) for row in rows]
            ),
            "negative_margin": summarize([float(row["negative_margin"]) for row in rows]),
            "weak_margin": summarize([float(row["weak_margin"]) for row in rows]),
        }

    b1_cfpr = operating_summary["B1"]["macro_cfpr"]["mean"]
    probe_summary = {
        kind: {
            field: summarize(
                [float(row[field]) for row in probe_rows if row["probe"] == kind]
            )
            for field in ("macro_auroc", "macro_average_precision")
        }
        for kind in sorted({row["probe"] for row in probe_rows})
    }
    report = {
        "schema_version": 1,
        "development_only": True,
        "test_accessed": False,
        "seeds": args.seeds,
        "seed_count": len(args.seeds),
        "all_seeds_pass_p2_opportunity": all(row["opportunity_pass"] for row in seed_rows),
        "any_seed_passes_p3": any(row["p3_pass"] for row in seed_rows),
        "candidate": {
            field: summarize([float(row[field]) for row in seed_rows])
            for field in (
                "oracle_relative_bce_improvement",
                "beneficial_fraction",
                "harmful_fraction",
            )
        },
        "operating_point": {
            "calibration_target_recall": 0.8,
            "methods": operating_summary,
            "relative_cfpr_reduction_vs_b1": {
                method: (b1_cfpr - summary["macro_cfpr"]["mean"]) / b1_cfpr
                for method, summary in operating_summary.items()
            },
        },
        "training_constraints": training_summary,
        "group_observability_probes": probe_summary,
        "seed_results": seed_rows,
    }

    report_dir = REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_json = report_dir / f"{args.output_stem}.json"
    output_operating_csv = report_dir / f"{args.output_stem}_operating.csv"
    output_training_csv = report_dir / f"{args.output_stem}_training.csv"
    output_probe_csv = report_dir / f"{args.output_stem}_probes.csv"
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path, rows in (
        (output_operating_csv, operating_rows),
        (output_training_csv, training_rows),
        (output_probe_csv, probe_rows),
    ):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
