#!/usr/bin/env python3
"""Run the single frozen official-test evaluation after the Q12 method freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluate_gate_modelval import load_candidate, load_split, method_scores, summarize_method

from edcr.config import load_config

FROZEN_METHODS = ("B0", "B1", "B2", "B3", "B4", "B4C", "B4G", "B5", "B9", "M1", "M2", "O1")


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--p3-run", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(REPO_ROOT / args.config)
    output_dir = REPO_ROOT / "runs" / args.p3_run
    training_path = output_dir / "metrics.json"
    thresholds_path = output_dir / "modelval_operating_metrics.json"
    gates_path = output_dir / "gates.pt"
    test_groups_path = REPO_ROOT / "data/groups/natural_test.npz"
    test_output_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.npz"
    required = (training_path, thresholds_path, gates_path, test_groups_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing frozen input(s): {missing}")
    if test_output_path.exists() or predictions_path.exists():
        raise SystemExit(f"refusing to overwrite prior final test output in {output_dir}")

    training = json.loads(training_path.read_text(encoding="utf-8"))
    calibration = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if training["seed"] not in config["training"]["seeds"]:
        raise SystemExit("run seed is not registered")
    if calibration["target_recall"] != config["evaluation"]["calibration_recall"]:
        raise SystemExit("stored threshold target differs from frozen configuration")
    if tuple(calibration["methods"].keys()) != FROZEN_METHODS:
        raise SystemExit("stored method matrix differs from Q12 freeze")
    if tuple(calibration["thresholds"].keys()) != FROZEN_METHODS:
        raise SystemExit("stored threshold matrix differs from Q12 freeze")
    dry_report = {
        "p3_run": args.p3_run,
        "seed": training["seed"],
        "candidate_run": training["candidate_run"],
        "frozen_methods": FROZEN_METHODS,
        "threshold_target_recall": calibration["target_recall"],
        "inputs_present": True,
        "prior_test_output": False,
    }
    if args.dry_run:
        print(json.dumps(dry_report, indent=2))
        return 0

    candidate_run = training["candidate_run"]
    candidate_path = REPO_ROOT / "runs" / candidate_run / "candidates.pt"
    candidate = load_candidate(REPO_ROOT / "runs" / candidate_run, config)
    test_ids, test_features, test_tensors = load_split("test", candidate)
    gate_states = torch.load(gates_path, map_location="cpu", weights_only=True)
    scores, _gate_values = method_scores(
        candidate,
        test_features,
        test_tensors,
        gate_states,
        int(config["training"]["gate_hidden"]),
        int(config["training"]["cached_batch_size"]),
    )
    if set(scores) != set(FROZEN_METHODS) or len(scores) != len(FROZEN_METHODS):
        raise RuntimeError("computed method matrix differs from Q12 freeze")
    with np.load(test_groups_path) as groups:
        if not np.array_equal(groups["sample_id"].astype(str), test_ids.astype(str)):
            raise ValueError("test natural-group sample order differs")
        negative = groups["negative"].astype(bool)
        weak = groups["weak"].astype(bool)
        weak_coco_small = groups["weak_coco_small"].astype(bool)
    labels = test_tensors.target_labels.numpy()
    thresholds = calibration["thresholds"]
    method_metrics = {
        method: summarize_method(
            labels, scores[method], thresholds[method], negative, weak, weak_coco_small
        )
        for method in FROZEN_METHODS
    }
    report = {
        "schema_version": 1,
        "protocol_freeze": "Q12",
        "p3_run": args.p3_run,
        "candidate_run": candidate_run,
        "seed": training["seed"],
        "calibration_split": "calib",
        "evaluation_split": "official_val2017_test",
        "test_accessed": True,
        "target_recall": calibration["target_recall"],
        "methods": method_metrics,
        "diagnostic_only_methods": ["O1"],
        "frozen_methods": list(FROZEN_METHODS),
    }
    test_output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    arrays: dict[str, np.ndarray] = {
        "sample_id": test_ids,
        "labels": labels.astype(np.uint8),
        "negative": negative.astype(np.uint8),
        "weak": weak.astype(np.uint8),
        "weak_coco_small": weak_coco_small.astype(np.uint8),
    }
    for method in FROZEN_METHODS:
        arrays[f"scores_{method}"] = scores[method].astype(np.float32)
        arrays[f"thresholds_{method}"] = np.asarray(thresholds[method], dtype=np.float32)
    np.savez_compressed(predictions_path, **arrays)
    provenance = {
        "candidate_checkpoint": candidate_path,
        "gates_checkpoint": gates_path,
        "calibrated_thresholds": thresholds_path,
        "test_cache_index": REPO_ROOT / "data/features/test/index.json",
        "test_manifest": REPO_ROOT / "data/manifests/test.jsonl",
        "test_groups": test_groups_path,
        "evaluator_code": Path(__file__).resolve(),
        "preregistration": REPO_ROOT / "docs/preregistration.md",
    }
    (output_dir / "test_input_hashes.json").write_text(
        json.dumps(
            {name: sha256(path) for name, path in provenance.items()}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "test_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "selected_macro_ap",
            "macro_cfpr",
            "macro_recall",
            "macro_weak_recall",
            "macro_coco_small_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for method, row in method_metrics.items():
            writer.writerow({"method": method, **{key: row[key] for key in fieldnames[1:]}})
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
