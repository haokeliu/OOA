#!/usr/bin/env python3
"""Train natural-only P3 gate baselines and constrained M1 on frozen P2d candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.gate_targets import update_dual
from edcr.gating import (
    ConfidenceGate,
    ConstantGate,
    GateTensors,
    build_gate_tensors,
    gate_values,
    predict_gate,
    target_balanced_bce,
)
from edcr.models import CandidateModel, GateNetwork, MultiViewHead, RelationIndex


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def load_candidate(run_dir: Path, config: dict[str, object]) -> CandidateModel:
    checkpoint = torch.load(run_dir / "candidates.pt", map_location="cpu", weights_only=True)
    first_weight = checkpoint["base_head"]["linear.weight"]
    feature_dim, class_count = first_weight.shape[1], first_weight.shape[0]
    base_head = MultiViewHead(feature_dim, class_count, config["features"]["pooling_temperature"])
    full_head = nn.Linear(feature_dim, class_count)
    relations = [RelationIndex(**row) for row in checkpoint["relations"]]
    model = CandidateModel(
        base_head,
        full_head,
        relations,
        config["training"]["residual_bound"],
        tuple(checkpoint["base_view_indices"]),
        torch.tensor(checkpoint["context_centers"]),
        torch.tensor(checkpoint["context_scales"]),
        checkpoint["context_signal"],
    )
    model.load_state_dict(checkpoint["candidate"])
    model.requires_grad_(False)
    model.eval()
    return model


def load_groups(path: Path, sample_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(path) as groups:
        group_ids = groups["sample_id"].astype(str)
        if not np.array_equal(group_ids, sample_ids.astype(str)):
            raise ValueError(f"group sample order differs: {path}")
        return torch.from_numpy(groups["negative"]), torch.from_numpy(groups["weak"])


def selected_macro_ap(labels: torch.Tensor, logits: torch.Tensor) -> float:
    return float(
        np.mean(
            [
                average_precision_score(labels[:, index].numpy(), logits[:, index].numpy())
                for index in range(labels.shape[1])
            ]
        )
    )


def gain_target_diagnostics(tensors: GateTensors, epsilon: float) -> dict[str, object]:
    """Audit whether B9's loss-difference target collapses to the class label."""
    base_loss = functional.binary_cross_entropy_with_logits(
        tensors.base, tensors.target_labels, reduction="none"
    )
    context_loss = functional.binary_cross_entropy_with_logits(
        tensors.base + tensors.residual, tensors.target_labels, reduction="none"
    )
    difference = base_loss - context_loss
    gain_target = difference > 0
    label = tensors.target_labels > 0.5

    def summarize(keep: torch.Tensor) -> dict[str, float | int | None]:
        retained = int(keep.sum())
        return {
            "retained": retained,
            "ignored_fraction": float((~keep).float().mean()),
            "label_agreement": float((gain_target[keep] == label[keep]).float().mean())
            if retained
            else None,
            "complement_agreement": float((gain_target[keep] == ~label[keep]).float().mean())
            if retained
            else None,
        }

    overall_keep = difference.abs() >= epsilon
    return {
        "epsilon": epsilon,
        "overall": summarize(overall_keep),
        "per_relation": [
            summarize(overall_keep[:, relation]) for relation in range(difference.shape[1])
        ],
    }


def balanced_indices(mask: torch.Tensor) -> list[torch.Tensor]:
    _sample_count, relation_count = mask.shape
    return [
        torch.nonzero(mask[:, relation], as_tuple=False).squeeze(1) * relation_count + relation
        for relation in range(relation_count)
    ]


def sample_balanced(
    pools: list[torch.Tensor], per_relation: int, generator: torch.Generator
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    for pool in pools:
        if len(pool) == 0:
            raise ValueError("balanced group pool has no support")
        choices = torch.randint(len(pool), (per_relation,), generator=generator)
        output.append(pool[choices])
    return torch.cat(output)


def evaluate(
    method: str,
    gate: nn.Module,
    tensors: GateTensors,
    negative: torch.Tensor,
    weak: torch.Tensor,
    batch_size: int,
    config: dict[str, object],
) -> dict[str, float | bool | list[float]]:
    values = predict_gate(method, gate, tensors, batch_size)
    logits = tensors.base + values * tensors.residual
    negative_risk = float(target_balanced_bce(logits, tensors.target_labels, negative))
    weak_risk = float(target_balanced_bce(logits, tensors.target_labels, weak))
    reference = tensors.base + tensors.residual
    negative_reference = float(target_balanced_bce(reference, tensors.target_labels, negative))
    weak_reference = float(target_balanced_bce(reference, tensors.target_labels, weak))
    negative_limit = config["training"]["negative_risk_ratio"] * negative_reference
    weak_limit = weak_reference + config["training"]["weak_risk_slack"]
    return {
        "selected_macro_ap": selected_macro_ap(tensors.target_labels, logits),
        "natural_bce": float(target_balanced_bce(logits, tensors.target_labels)),
        "negative_risk": negative_risk,
        "negative_reference": negative_reference,
        "negative_limit": negative_limit,
        "weak_risk": weak_risk,
        "weak_reference": weak_reference,
        "weak_limit": weak_limit,
        "constraint_violation": max(0.0, negative_risk - negative_limit)
        + max(0.0, weak_risk - weak_limit),
        "feasible": negative_risk <= negative_limit and weak_risk <= weak_limit,
        "gate_mean": float(values.mean()),
        "gate_quantiles": [float(torch.quantile(values, q)) for q in (0.1, 0.5, 0.9)],
    }


def make_gate(method: str, tensors: GateTensors, hidden: int) -> nn.Module:
    if method == "B3":
        return ConstantGate(tensors.numeric.shape[1])
    if method in {"B4", "B4C", "B4G"}:
        return ConfidenceGate(hidden)
    return GateNetwork(tensors.numeric.shape[-1], tensors.source_embeddings.shape[-1], hidden)


def train_one(
    method: str,
    train: GateTensors,
    validation: GateTensors,
    train_negative: torch.Tensor,
    train_weak: torch.Tensor,
    validation_negative: torch.Tensor,
    validation_weak: torch.Tensor,
    config: dict[str, object],
    seed: int,
    max_epochs: int,
    log_handle,
) -> tuple[nn.Module, dict[str, object]]:
    torch.manual_seed(seed)
    gate = make_gate(method, train, config["training"]["gate_hidden"])
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=config["training"]["gate_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    batch_size = int(config["training"]["cached_batch_size"])
    relation_count = train.numeric.shape[1]
    total = train.base.numel()
    natural_indices = torch.arange(total)
    negative_pools = balanced_indices(train_negative)
    weak_pools = balanced_indices(train_weak)
    generator = torch.Generator().manual_seed(seed)
    per_relation = max(1, batch_size // relation_count)
    reference_logits = train.base + train.residual
    negative_reference = float(
        target_balanced_bce(reference_logits, train.target_labels, train_negative)
    )
    weak_reference = float(target_balanced_bce(reference_logits, train.target_labels, train_weak))
    lambda_negative = 0.0
    lambda_weak = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float] | None = None
    stale = 0
    for epoch in range(max_epochs):
        gate.train()
        order = natural_indices[torch.randperm(total, generator=generator)]
        for offset in range(0, total, batch_size):
            indices = order[offset : offset + batch_size]
            relation_indices = indices % relation_count
            sample_indices = indices // relation_count
            q = gate_values(method, gate, train, indices)
            logits = (
                train.base[sample_indices, relation_indices]
                + q * train.residual[sample_indices, relation_indices]
            )
            labels = train.target_labels[sample_indices, relation_indices]
            if method == "B9":
                base_loss = functional.binary_cross_entropy_with_logits(
                    train.base[sample_indices, relation_indices], labels, reduction="none"
                )
                context_loss = functional.binary_cross_entropy_with_logits(
                    train.base[sample_indices, relation_indices]
                    + train.residual[sample_indices, relation_indices],
                    labels,
                    reduction="none",
                )
                difference = (base_loss - context_loss).detach()
                weights = difference.abs()
                targets = (difference > 0).float()
                keep = weights >= config["training"]["weak_delta_epsilon"]
                loss = (
                    functional.binary_cross_entropy(q[keep], targets[keep], reduction="none")
                    * weights[keep]
                ).sum() / weights[keep].sum().clamp_min(1e-12)
            else:
                loss = functional.binary_cross_entropy_with_logits(logits, labels)
            if method in {"B4C", "M1"}:
                negative_indices = sample_balanced(negative_pools, per_relation, generator)
                weak_indices = sample_balanced(weak_pools, per_relation, generator)
                negative_q = gate_values(method, gate, train, negative_indices)
                weak_q = gate_values(method, gate, train, weak_indices)
                neg_rel, neg_sample = (
                    negative_indices % relation_count,
                    negative_indices // relation_count,
                )
                weak_rel, weak_sample = (
                    weak_indices % relation_count,
                    weak_indices // relation_count,
                )
                negative_loss = functional.binary_cross_entropy_with_logits(
                    train.base[neg_sample, neg_rel]
                    + negative_q * train.residual[neg_sample, neg_rel],
                    train.target_labels[neg_sample, neg_rel],
                )
                weak_loss = functional.binary_cross_entropy_with_logits(
                    train.base[weak_sample, weak_rel]
                    + weak_q * train.residual[weak_sample, weak_rel],
                    train.target_labels[weak_sample, weak_rel],
                )
                loss = loss + lambda_negative * (
                    negative_loss - config["training"]["negative_risk_ratio"] * negative_reference
                )
                loss = loss + lambda_weak * (
                    weak_loss - weak_reference - config["training"]["weak_risk_slack"]
                )
            if method in {"B4G", "M2"}:
                negative_indices = sample_balanced(negative_pools, per_relation, generator)
                weak_indices = sample_balanced(weak_pools, per_relation, generator)
                negative_q = gate_values(method, gate, train, negative_indices)
                weak_q = gate_values(method, gate, train, weak_indices)
                group_gate_loss = 0.5 * (
                    functional.binary_cross_entropy(negative_q, torch.zeros_like(negative_q))
                    + functional.binary_cross_entropy(weak_q, torch.ones_like(weak_q))
                )
                loss = loss + group_gate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_metrics = evaluate(
            method,
            gate,
            train,
            train_negative,
            train_weak,
            batch_size,
            config,
        )
        if method in {"B4C", "M1"}:
            lambda_negative = update_dual(
                lambda_negative,
                train_metrics["negative_risk"]
                - config["training"]["negative_risk_ratio"] * negative_reference,
                config["training"]["dual_lr"],
            )
            lambda_weak = update_dual(
                lambda_weak,
                train_metrics["weak_risk"] - weak_reference - config["training"]["weak_risk_slack"],
                config["training"]["dual_lr"],
            )
        validation_metrics = evaluate(
            method,
            gate,
            validation,
            validation_negative,
            validation_weak,
            batch_size,
            config,
        )
        record = {
            "method": method,
            "epoch": epoch,
            "lambda_negative": lambda_negative,
            "lambda_weak": lambda_weak,
            "train": train_metrics,
            "modelval": validation_metrics,
        }
        log_handle.write(json.dumps(record, sort_keys=True) + "\n")
        log_handle.flush()
        if method in {"B4C", "B4G", "M1", "M2"}:
            key = (
                1.0 if validation_metrics["feasible"] else 0.0,
                validation_metrics["selected_macro_ap"]
                if validation_metrics["feasible"]
                else -validation_metrics["constraint_violation"],
            )
        else:
            key = (1.0, validation_metrics["selected_macro_ap"])
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(gate.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"{method} produced no checkpoint")
    gate.load_state_dict(best_state)
    return gate, evaluate(
        method,
        gate,
        validation,
        validation_negative,
        validation_weak,
        batch_size,
        config,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--candidate-run", default="p2d_seed17")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-id")
    parser.add_argument("--gate-cache", default="data/features/gate")
    parser.add_argument("--modelval-cache", default="data/features/modelval")
    parser.add_argument("--gate-groups", default="data/groups/natural_gate.npz")
    parser.add_argument("--modelval-groups", default="data/groups/natural_modelval.npz")
    parser.add_argument("--gate-manifest", default="data/manifests/gate.jsonl")
    parser.add_argument("--modelval-manifest", default="data/manifests/modelval.jsonl")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["B3", "B4", "B4C", "B4G", "B5", "B9", "M1", "M2"],
        default=None,
    )
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--dual-lr", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(REPO_ROOT / args.config)
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not preregistered")
    if args.max_epochs is not None:
        config["training"]["gate_epochs"] = args.max_epochs
    if args.patience is not None:
        config["training"]["patience"] = args.patience
    if args.dual_lr is not None:
        if args.dual_lr <= 0:
            raise SystemExit("dual learning rate must be positive")
        config["training"]["dual_lr"] = args.dual_lr
    run_id = args.run_id or f"p3_natural_seed{args.seed}"
    cache_paths = {
        "gate": REPO_ROOT / args.gate_cache,
        "modelval": REPO_ROOT / args.modelval_cache,
    }
    group_paths = {
        "gate": REPO_ROOT / args.gate_groups,
        "modelval": REPO_ROOT / args.modelval_groups,
    }
    manifest_paths = {
        "gate": REPO_ROOT / args.gate_manifest,
        "modelval": REPO_ROOT / args.modelval_manifest,
    }
    required = {split: (path / "index.json").is_file() for split, path in cache_paths.items()}
    if args.dry_run:
        print(json.dumps({"run_id": run_id, "feature_caches": required}, indent=2))
        return 0
    if not all(required.values()):
        raise SystemExit(f"feature caches incomplete: {required}")
    candidate = load_candidate(REPO_ROOT / "runs" / args.candidate_run, config)
    if any(parameter.requires_grad for parameter in candidate.parameters()):
        raise RuntimeError("candidate must remain frozen")
    split_tensors: dict[str, GateTensors] = {}
    split_masks: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("gate", "modelval"):
        sample_ids, features_np, labels_np = load_feature_shards(
            cache_paths[split],
            manifest_paths[split],
            80,
        )
        split_tensors[split] = build_gate_tensors(
            candidate, torch.from_numpy(features_np), torch.from_numpy(labels_np)
        )
        split_masks[split] = load_groups(group_paths[split], sample_ids)
    output_dir = REPO_ROOT / "runs" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoints: dict[str, object] = {}
    metrics: dict[str, object] = {
        "schema_version": 1,
        "candidate_run": args.candidate_run,
        "seed": args.seed,
        "training_data": "natural_gate_only",
        "test_accessed": False,
        "methods": {},
        "b9_gain_target_diagnostics": gain_target_diagnostics(
            split_tensors["gate"], float(config["training"]["weak_delta_epsilon"])
        ),
    }
    methods = args.methods or ["B3", "B4", "B4C", "B4G", "B5", "B9", "M1", "M2"]
    max_epochs = int(config["training"]["gate_epochs"])
    with (output_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_handle:
        for method in methods:
            gate, result = train_one(
                method,
                split_tensors["gate"],
                split_tensors["modelval"],
                *split_masks["gate"],
                *split_masks["modelval"],
                config,
                args.seed,
                max_epochs,
                log_handle,
            )
            checkpoints[method] = gate.state_dict()
            metrics["methods"][method] = result
    metrics["elapsed_seconds"] = time.perf_counter() - started
    torch.save(checkpoints, output_dir / "gates.pt")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config_resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cpu",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "git_commit.txt").write_text((git_commit() or "UNCOMMITTED") + "\n")
    provenance_paths = {
        "candidate_checkpoint": REPO_ROOT / "runs" / args.candidate_run / "candidates.pt",
        "config": REPO_ROOT / args.config,
        "gate_manifest": manifest_paths["gate"],
        "modelval_manifest": manifest_paths["modelval"],
        "gate_groups": group_paths["gate"],
        "modelval_groups": group_paths["modelval"],
        "train_gate_code": Path(__file__).resolve(),
        "gating_code": REPO_ROOT / "src" / "edcr" / "gating.py",
    }
    (output_dir / "input_hashes.json").write_text(
        json.dumps(
            {name: sha256(path) for name, path in provenance_paths.items()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
