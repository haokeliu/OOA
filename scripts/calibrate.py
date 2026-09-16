#!/usr/bin/env python3
"""Freeze recall-targeted thresholds using calib labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.calibration import calibrate_classes
from edcr.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", choices=["calib"], default="calib")
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    run_dir = REPO_ROOT / "runs" / args.run
    manifest = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (REPO_ROOT / "data/manifests/calib.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    output: dict[str, object] = {
        "schema_version": 1,
        "split": "calib",
        "target_recall": config["evaluation"]["calibration_recall"],
        "min_positives": config["evaluation"]["min_calibration_positives"],
        "prediction_rule": "score >= threshold",
        "methods": {},
    }
    for method in ("B0", "B1", "B2"):
        prediction_path = run_dir / f"predictions_{method}_calib.npz"
        with np.load(prediction_path) as predictions:
            sample_ids = predictions["sample_id"].astype(str)
            logits = predictions["logits"].astype(np.float64)
        labels = np.zeros_like(logits)
        for index, sample_id in enumerate(sample_ids):
            labels[index, manifest[sample_id]["labels"]] = 1
        thresholds = calibrate_classes(
            labels,
            logits,
            config["evaluation"]["calibration_recall"],
            config["evaluation"]["min_calibration_positives"],
        )
        output["methods"][method] = thresholds
    (run_dir / "thresholds.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
