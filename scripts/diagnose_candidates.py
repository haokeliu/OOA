#!/usr/bin/env python3
"""Break a frozen P2 candidate result down by relation and label group."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def bce(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, logits) - labels * logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=["modelval"], default="modelval")
    args = parser.parse_args()
    run_dir = REPO_ROOT / "runs" / args.run
    relations = json.loads(
        (REPO_ROOT / "data/manifests/selected_relations.json").read_text(encoding="utf-8")
    )
    manifest_rows = {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in (REPO_ROOT / "data/manifests/modelval.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    arrays: dict[str, np.ndarray] = {}
    sample_ids: np.ndarray | None = None
    for method in ("B1", "B2"):
        with np.load(run_dir / f"predictions_{method}.npz") as prediction:
            current_ids = prediction["sample_id"].astype(str)
            if sample_ids is not None and not np.array_equal(sample_ids, current_ids):
                raise ValueError("prediction sample order differs between candidates")
            sample_ids = current_ids
            arrays[method] = prediction["logits"].astype(np.float64)
    assert sample_ids is not None
    labels = np.zeros_like(arrays["B1"])
    for index, sample_id in enumerate(sample_ids):
        labels[index, manifest_rows[sample_id]["labels"]] = 1.0

    epsilon = 1e-4
    rows: list[dict[str, object]] = []
    for relation in relations:
        source = int(relation["source_id"])
        target = int(relation["target_id"])
        target_labels = labels[:, target]
        base = arrays["B1"][:, target]
        context = arrays["B2"][:, target]
        difference = bce(target_labels, base) - bce(target_labels, context)
        source_labels = labels[:, source]
        base_loss = bce(target_labels, base)
        context_loss = bce(target_labels, context)
        use_context = context_loss.mean() < base_loss.mean()
        global_loss = context_loss if use_context else base_loss
        oracle_loss = np.minimum(base_loss, context_loss)
        groups: dict[str, object] = {}
        for source_value, target_value in ((1, 1), (1, 0), (0, 1), (0, 0)):
            mask = (source_labels == source_value) & (target_labels == target_value)
            key = f"n{source_value}{target_value}"
            groups[key] = {
                "count": int(mask.sum()),
                "mean_loss_difference": float(difference[mask].mean()) if mask.any() else None,
                "beneficial_fraction": float(np.mean(difference[mask] > epsilon))
                if mask.any()
                else None,
                "harmful_fraction": float(np.mean(difference[mask] < -epsilon))
                if mask.any()
                else None,
            }
        rows.append(
            {
                "pair_id": relation["pair_id"],
                "source_id": source,
                "target_id": target,
                "best_global_bce": float(global_loss.mean()),
                "oracle_bce": float(oracle_loss.mean()),
                "oracle_relative_improvement": float(
                    (global_loss.mean() - oracle_loss.mean()) / global_loss.mean()
                ),
                "beneficial_fraction": float(np.mean(difference > epsilon)),
                "harmful_fraction": float(np.mean(difference < -epsilon)),
                "ignored_fraction": float(np.mean(np.abs(difference) <= epsilon)),
                "residual_absolute_quantiles": {
                    str(q): float(np.quantile(np.abs(context - base), q))
                    for q in (0.5, 0.9, 0.99, 1.0)
                },
                "groups": groups,
            }
        )
    report = {"schema_version": 1, "run": args.run, "split": args.split, "relations": rows}
    output = run_dir / f"diagnostics_{args.split}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
