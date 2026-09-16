#!/usr/bin/env python3
"""Probe whether inference features separate frozen natural risk groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_gate import balanced_indices, load_candidate, load_groups, sample_balanced

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.gating import ConfidenceGate, GateTensors, build_gate_tensors, gate_values
from edcr.models import GateNetwork


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_probe(kind: str, tensors: GateTensors, hidden: int) -> nn.Module:
    if kind == "confidence":
        return ConfidenceGate(hidden)
    return GateNetwork(tensors.numeric.shape[-1], tensors.source_embeddings.shape[-1], hidden)


def probe_values(
    kind: str, probe: nn.Module, tensors: GateTensors, indices: torch.Tensor
) -> torch.Tensor:
    return gate_values("B4" if kind == "confidence" else "probe_full", probe, tensors, indices)


def train_probe(
    kind: str,
    tensors: GateTensors,
    negative: torch.Tensor,
    weak: torch.Tensor,
    config: dict[str, object],
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    probe = make_probe(kind, tensors, int(config["training"]["gate_hidden"]))
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(config["training"]["gate_lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["training"]["cached_batch_size"])
    relation_count = tensors.numeric.shape[1]
    per_group_relation = max(1, batch_size // (2 * relation_count))
    steps = math.ceil((int(negative.sum()) + int(weak.sum())) / batch_size)
    negative_pools = balanced_indices(negative)
    weak_pools = balanced_indices(weak)
    generator = torch.Generator().manual_seed(seed)
    for _epoch in range(20):
        probe.train()
        for _step in range(steps):
            negative_indices = sample_balanced(negative_pools, per_group_relation, generator)
            weak_indices = sample_balanced(weak_pools, per_group_relation, generator)
            indices = torch.cat((negative_indices, weak_indices))
            targets = torch.cat(
                (torch.zeros_like(negative_indices), torch.ones_like(weak_indices))
            ).float()
            order = torch.randperm(len(indices), generator=generator)
            values = probe_values(kind, probe, tensors, indices[order])
            loss = functional.binary_cross_entropy(values, targets[order])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return probe


def evaluate_probe(
    kind: str,
    probe: nn.Module,
    tensors: GateTensors,
    negative: torch.Tensor,
    weak: torch.Tensor,
) -> dict[str, object]:
    relation_count = tensors.numeric.shape[1]
    negative_pools = balanced_indices(negative)
    weak_pools = balanced_indices(weak)
    rows: list[dict[str, float | int]] = []
    probe.eval()
    with torch.inference_mode():
        for relation in range(relation_count):
            negative_indices = negative_pools[relation]
            weak_indices = weak_pools[relation]
            indices = torch.cat((negative_indices, weak_indices))
            targets = np.concatenate(
                (np.zeros(len(negative_indices)), np.ones(len(weak_indices)))
            )
            values = probe_values(kind, probe, tensors, indices).numpy()
            rows.append(
                {
                    "relation_index": relation,
                    "negative_support": len(negative_indices),
                    "weak_support": len(weak_indices),
                    "auroc": float(roc_auc_score(targets, values)),
                    "average_precision": float(average_precision_score(targets, values)),
                    "mean_negative_score": float(values[: len(negative_indices)].mean()),
                    "mean_weak_score": float(values[len(negative_indices) :].mean()),
                }
            )
    return {
        "macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "macro_average_precision": float(
            np.mean([row["average_precision"] for row in rows])
        ),
        "per_relation": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--candidate-run")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not preregistered")
    candidate_run = args.candidate_run or f"p2e_seed{args.seed}"
    run_id = args.run_id or f"q11_probe_seed{args.seed}"
    paths = {
        "gate_cache": REPO_ROOT / "data/features/gate",
        "modelval_cache": REPO_ROOT / "data/features/modelval",
        "gate_manifest": REPO_ROOT / "data/manifests/gate.jsonl",
        "modelval_manifest": REPO_ROOT / "data/manifests/modelval.jsonl",
        "gate_groups": REPO_ROOT / "data/groups/natural_gate.npz",
        "modelval_groups": REPO_ROOT / "data/groups/natural_modelval.npz",
    }
    candidate_path = REPO_ROOT / "runs" / candidate_run / "candidates.pt"
    candidate = load_candidate(REPO_ROOT / "runs" / candidate_run, config)
    tensors: dict[str, GateTensors] = {}
    masks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("gate", "modelval"):
        sample_ids, features, labels = load_feature_shards(
            paths[f"{split}_cache"], paths[f"{split}_manifest"], 80
        )
        tensors[split] = build_gate_tensors(
            candidate, torch.from_numpy(features), torch.from_numpy(labels)
        )
        masks[split] = load_groups(paths[f"{split}_groups"], sample_ids)

    output_dir = REPO_ROOT / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results: dict[str, object] = {}
    checkpoints: dict[str, object] = {}
    for kind in ("confidence", "full"):
        probe = train_probe(kind, tensors["gate"], *masks["gate"], config, args.seed)
        checkpoints[kind] = probe.state_dict()
        results[kind] = evaluate_probe(kind, probe, tensors["modelval"], *masks["modelval"])
    report = {
        "schema_version": 1,
        "diagnostic_only": True,
        "test_accessed": False,
        "training_split": "gate",
        "evaluation_split": "modelval",
        "fixed_epochs": 20,
        "seed": args.seed,
        "candidate_run": candidate_run,
        "elapsed_seconds": time.perf_counter() - started,
        "probes": results,
    }
    torch.save(checkpoints, output_dir / "probes.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = {
        "candidate_checkpoint": candidate_path,
        "config": REPO_ROOT / args.config,
        "gate_manifest": paths["gate_manifest"],
        "modelval_manifest": paths["modelval_manifest"],
        "gate_groups": paths["gate_groups"],
        "modelval_groups": paths["modelval_groups"],
        "probe_code": Path(__file__).resolve(),
    }
    (output_dir / "input_hashes.json").write_text(
        json.dumps(
            {name: sha256(path) for name, path in provenance.items()}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
