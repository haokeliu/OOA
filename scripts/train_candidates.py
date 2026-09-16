#!/usr/bin/env python3
"""Train B0/B1 heads and the frozen single-relation candidate B2."""

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
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from edcr.cache import load_feature_shards
from edcr.config import load_config
from edcr.metrics import macro_average_precision
from edcr.models import CandidateModel, MultiViewHead, RelationIndex


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


def batches(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def predict_heads(
    base_head: MultiViewHead,
    full_head: nn.Linear,
    features: torch.Tensor,
    batch_size: int,
    base_view_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    base_outputs: list[torch.Tensor] = []
    full_outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for offset in range(0, len(features), batch_size):
            batch = features[offset : offset + batch_size]
            base_outputs.append(base_head(batch[:, base_view_indices]).cpu())
            full_outputs.append(full_head(batch[:, 0]).cpu())
    return torch.cat(base_outputs).numpy(), torch.cat(full_outputs).numpy()


def bce_numpy(labels: np.ndarray, logits: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, logits) - labels * logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--fit-cache", default="data/features/fit")
    parser.add_argument("--modelval-cache", default="data/features/modelval")
    parser.add_argument("--fit-manifest", default="data/manifests/fit.jsonl")
    parser.add_argument("--modelval-manifest", default="data/manifests/modelval.jsonl")
    parser.add_argument("--class-count", type=int, default=80)
    parser.add_argument(
        "--internal-headval-modulus",
        type=int,
        help="Use one deterministic fit fold for head early stopping instead of modelval",
    )
    parser.add_argument(
        "--relation-manifest", default="data/manifests/selected_relations.json"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-id")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--base-views", choices=["all", "corners"], default="all")
    parser.add_argument(
        "--context-selection", choices=["ap", "bce", "oracle_safe"], default="ap"
    )
    parser.add_argument("--candidate-global-bce-budget", type=float, default=0.02)
    parser.add_argument("--context-centering", choices=["zero", "fit_mean"], default="zero")
    parser.add_argument(
        "--context-signal", choices=["probability", "standardized_logit"], default="probability"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = REPO_ROOT / args.config
    config = load_config(config_path)
    if args.candidate_global_bce_budget < 0:
        raise SystemExit("candidate global BCE budget must be non-negative")
    if args.class_count <= 1:
        raise SystemExit("class count must exceed one")
    if args.internal_headval_modulus is not None and args.internal_headval_modulus < 2:
        raise SystemExit("internal head-validation modulus must be at least two")
    if args.seed not in config["training"]["seeds"]:
        raise SystemExit(f"seed {args.seed} is not preregistered")
    run_id = args.run_id or f"p2_seed{args.seed}"
    base_view_indices = (0, 1, 2, 3, 4) if args.base_views == "all" else (1, 2, 3, 4)
    run_dir = REPO_ROOT / "runs" / run_id
    relation_manifest = REPO_ROOT / args.relation_manifest
    selected_relations = json.loads(relation_manifest.read_text(encoding="utf-8"))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "fit_cache_exists": (REPO_ROOT / args.fit_cache / "index.json").is_file(),
                    "modelval_cache_exists": (
                        REPO_ROOT / args.modelval_cache / "index.json"
                    ).is_file(),
                    "relations": [row["pair_id"] for row in selected_relations],
                    "base_views": args.base_views,
                    "context_selection": args.context_selection,
                    "context_centering": args.context_centering,
                    "context_signal": args.context_signal,
                },
                indent=2,
            )
        )
        return 0

    fit_manifest = REPO_ROOT / args.fit_manifest
    modelval_manifest = REPO_ROOT / args.modelval_manifest
    fit_ids, fit_features_np, fit_labels_np = load_feature_shards(
        REPO_ROOT / args.fit_cache, fit_manifest, args.class_count
    )
    val_ids, val_features_np, val_labels_np = load_feature_shards(
        REPO_ROOT / args.modelval_cache,
        modelval_manifest,
        args.class_count,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    fit_features = torch.from_numpy(fit_features_np)
    fit_labels = torch.from_numpy(fit_labels_np)
    val_features = torch.from_numpy(val_features_np)
    headval_source = "modelval"
    head_train_features = fit_features
    head_train_labels = fit_labels
    head_val_features = val_features
    head_val_labels_np = val_labels_np
    if args.internal_headval_modulus is not None:
        internal_mask = np.asarray(
            [
                int.from_bytes(
                    hashlib.sha256(f"{args.seed}:{sample_id}".encode()).digest()[:8],
                    "big",
                )
                % args.internal_headval_modulus
                == 0
                for sample_id in fit_ids
            ],
            dtype=bool,
        )
        if not internal_mask.any() or internal_mask.all():
            raise RuntimeError("deterministic internal head-validation split is empty")
        headval_source = f"fit_hash_mod_{args.internal_headval_modulus}_equals_0"
        head_train_features = fit_features[~torch.from_numpy(internal_mask)]
        head_train_labels = fit_labels[~torch.from_numpy(internal_mask)]
        head_val_features = fit_features[torch.from_numpy(internal_mask)]
        head_val_labels_np = fit_labels_np[internal_mask]
    feature_dim = fit_features.shape[-1]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base_head = MultiViewHead(
        feature_dim, args.class_count, config["features"]["pooling_temperature"]
    )
    full_head = nn.Linear(feature_dim, args.class_count)
    optimizer = torch.optim.AdamW(
        list(base_head.parameters()) + list(full_head.parameters()),
        lr=config["training"]["heads_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    epochs = args.max_epochs or config["training"]["head_epochs"]
    patience = config["training"]["patience"]
    batch_size = config["training"]["cached_batch_size"]
    log_path = run_dir / "train_log.jsonl"
    best_score = float("-inf")
    best_epoch = -1
    best_heads: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
    stale = 0
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        for epoch in range(epochs):
            base_head.train()
            full_head.train()
            total_loss = 0.0
            sample_count = 0
            for feature_batch, label_batch in batches(
                head_train_features, head_train_labels, batch_size, args.seed + epoch
            ):
                optimizer.zero_grad(set_to_none=True)
                base_logits = base_head(feature_batch[:, base_view_indices])
                full_logits = full_head(feature_batch[:, 0])
                loss = functional.binary_cross_entropy_with_logits(base_logits, label_batch)
                loss = loss + functional.binary_cross_entropy_with_logits(full_logits, label_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.detach().item() * len(feature_batch)
                sample_count += len(feature_batch)
            base_head.eval()
            full_head.eval()
            val_base, val_full = predict_heads(
                base_head, full_head, head_val_features, batch_size, base_view_indices
            )
            score = macro_average_precision(head_val_labels_np, val_base)
            record = {
                "stage": "heads",
                "epoch": epoch,
                "train_loss": total_loss / sample_count,
                "headval_source": headval_source,
                "headval_b1_map": score,
                "headval_b0_map": macro_average_precision(head_val_labels_np, val_full),
            }
            log_handle.write(json.dumps(record, sort_keys=True) + "\n")
            log_handle.flush()
            if score is not None and score > best_score:
                best_score = score
                best_epoch = epoch
                best_heads = (
                    copy.deepcopy(base_head.state_dict()),
                    copy.deepcopy(full_head.state_dict()),
                )
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_heads is None:
            raise RuntimeError("head training produced no selectable checkpoint")
        base_head.load_state_dict(best_heads[0])
        full_head.load_state_dict(best_heads[1])

        relations = [
            RelationIndex(source=int(row["source_id"]), target=int(row["target_id"]))
            for row in selected_relations
        ]
        source_indices = [relation.source for relation in relations]
        with torch.inference_mode():
            fit_source_logits = full_head(fit_features[:, 0])[:, source_indices]
        if args.context_signal == "standardized_logit":
            context_centers = fit_source_logits.mean(dim=0)
            context_scales = fit_source_logits.std(dim=0).clamp_min(1e-3)
        elif args.context_centering == "fit_mean":
            context_centers = torch.sigmoid(fit_source_logits).mean(dim=0)
            context_scales = torch.ones(len(relations))
        else:
            context_centers = torch.zeros(len(relations))
            context_scales = torch.ones(len(relations))
        candidate = CandidateModel(
            base_head,
            full_head,
            relations,
            config["training"]["residual_bound"],
            base_view_indices,
            context_centers,
            context_scales,
            args.context_signal,
        )
        candidate.freeze_heads()
        targets = [relation.target for relation in relations]
        fit_oracle_diagnostics: list[dict[str, object]] | None = None
        best_context: dict[str, torch.Tensor] | None = None
        if args.context_selection == "oracle_safe":
            candidate.eval()
            with torch.inference_mode():
                fit_base = candidate.base_head(fit_features[:, base_view_indices])[:, targets]
                source_logits = candidate.full_head(fit_features[:, 0])[:, source_indices]
                if args.context_signal == "standardized_logit":
                    source_signal = torch.tanh(
                        (source_logits - candidate.context_centers) / candidate.context_scales
                    )
                else:
                    source_signal = torch.sigmoid(source_logits) - candidate.context_centers
            fit_targets = fit_labels[:, targets]
            coefficient_grid = torch.arange(-1.95, 1.951, 0.05)
            selected_coefficients: list[float] = []
            fit_oracle_diagnostics = []
            for relation_index, relation_row in enumerate(selected_relations):
                base = fit_base[:, relation_index]
                labels = fit_targets[:, relation_index]
                base_loss = functional.binary_cross_entropy_with_logits(
                    base, labels, reduction="none"
                )
                safe_limit = float(base_loss.mean()) * (1.0 + args.candidate_global_bce_budget)
                candidates: list[tuple[float, float, float, float]] = []
                for coefficient_tensor in coefficient_grid:
                    coefficient = float(coefficient_tensor)
                    context = base + coefficient_tensor * source_signal[:, relation_index]
                    context_loss = functional.binary_cross_entropy_with_logits(
                        context, labels, reduction="none"
                    )
                    context_bce = float(context_loss.mean())
                    if context_bce <= safe_limit:
                        oracle_bce = float(torch.minimum(base_loss, context_loss).mean())
                        candidates.append(
                            (oracle_bce, abs(coefficient), coefficient, context_bce)
                        )
                if not candidates:
                    raise RuntimeError(f"no safe candidate for {relation_row['pair_id']}")
                oracle_bce, _magnitude, coefficient, context_bce = min(candidates)
                selected_coefficients.append(coefficient)
                fit_oracle_diagnostics.append(
                    {
                        "pair_id": relation_row["pair_id"],
                        "selected_coefficient": coefficient,
                        "base_bce": float(base_loss.mean()),
                        "safe_bce_limit": safe_limit,
                        "always_on_bce": context_bce,
                        "oracle_bce": oracle_bce,
                        "safe_grid_count": len(candidates),
                    }
                )
            coefficients = torch.tensor(selected_coefficients, dtype=candidate.context_raw.dtype)
            raw = torch.atanh(coefficients / float(config["training"]["residual_bound"]))
            with torch.no_grad():
                candidate.context_raw.copy_(raw)
            best_context = copy.deepcopy(candidate.state_dict())
            log_handle.write(
                json.dumps(
                    {
                        "stage": "context_oracle_safe_grid",
                        "selection_metric": args.context_selection,
                        "candidate_global_bce_budget": args.candidate_global_bce_budget,
                        "grid_start": -1.95,
                        "grid_stop": 1.95,
                        "grid_step": 0.05,
                        "relations": fit_oracle_diagnostics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            log_handle.flush()
        else:
            context_optimizer = torch.optim.AdamW(
                [candidate.context_raw],
                lr=config["training"]["heads_lr"],
                weight_decay=0.0,
            )
            best_context_score = float("-inf")
            stale = 0
            for epoch in range(epochs):
                candidate.train()
                total_loss = 0.0
                sample_count = 0
                for feature_batch, label_batch in batches(
                    fit_features, fit_labels, batch_size, args.seed + 10_000 + epoch
                ):
                    context_optimizer.zero_grad(set_to_none=True)
                    logits = candidate(feature_batch)["full"][:, targets]
                    loss = functional.binary_cross_entropy_with_logits(
                        logits, label_batch[:, targets]
                    )
                    loss.backward()
                    context_optimizer.step()
                    total_loss += loss.detach().item() * len(feature_batch)
                    sample_count += len(feature_batch)
                candidate.eval()
                with torch.inference_mode():
                    outputs = candidate(val_features)
                context_score = macro_average_precision(
                    val_labels_np, outputs["full"].numpy(), targets
                )
                context_bce = float(
                    functional.binary_cross_entropy_with_logits(
                        outputs["full"][:, targets], torch.from_numpy(val_labels_np[:, targets])
                    )
                )
                selection_score = context_score if args.context_selection == "ap" else -context_bce
                log_handle.write(
                    json.dumps(
                        {
                            "stage": "context",
                            "epoch": epoch,
                            "train_loss": total_loss / sample_count,
                            "modelval_selected_macro_ap": context_score,
                            "modelval_selected_bce": context_bce,
                            "selection_metric": args.context_selection,
                            "context_raw": candidate.context_raw.detach().tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log_handle.flush()
                if selection_score is not None and selection_score > best_context_score:
                    best_context_score = selection_score
                    best_context = copy.deepcopy(candidate.state_dict())
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break
    if best_context is None:
        raise RuntimeError("context training produced no selectable checkpoint")
    candidate.load_state_dict(best_context)
    candidate.eval()
    with torch.inference_mode():
        outputs = candidate(val_features)
    b1_logits = outputs["base"].numpy()
    b2_logits = outputs["full"].numpy()
    _, b0_logits = predict_heads(base_head, full_head, val_features, batch_size, base_view_indices)
    target_ids = [relation.target for relation in relations]
    selected_labels = val_labels_np[:, target_ids]
    b1_selected = b1_logits[:, target_ids]
    b2_selected = b2_logits[:, target_ids]
    b1_losses = bce_numpy(selected_labels, b1_selected)
    b2_losses = bce_numpy(selected_labels, b2_selected)
    differences = b1_losses - b2_losses
    use_context_globally = b2_losses.mean(axis=0) < b1_losses.mean(axis=0)
    global_losses = np.where(use_context_globally[None, :], b2_losses, b1_losses)
    oracle_losses = np.minimum(b1_losses, b2_losses)
    oracle_logits = b1_logits.copy()
    oracle_logits[:, target_ids] = np.where(differences > 0.0, b2_selected, b1_selected)
    best_global_loss = float(global_losses.mean())
    oracle_loss = float(oracle_losses.mean())
    epsilon = float(config["training"]["weak_delta_epsilon"])
    metrics = {
        "split": "modelval",
        "method_version": (
            "p2e"
            if args.context_selection == "oracle_safe"
            else (
                "p2d"
                if args.context_signal == "standardized_logit"
                else (
                    "p2c"
                    if args.context_centering == "fit_mean"
                    else ("p2b" if args.base_views == "corners" else "p2_v1")
                )
            )
        ),
        "base_views": args.base_views,
        "context_selection": args.context_selection,
        "context_centering": args.context_centering,
        "context_signal": args.context_signal,
        "candidate_global_bce_budget": args.candidate_global_bce_budget
        if args.context_selection == "oracle_safe"
        else None,
        "fit_oracle_diagnostics": fit_oracle_diagnostics,
        "context_centers": {
            row["pair_id"]: float(candidate.context_centers[index])
            for index, row in enumerate(selected_relations)
        },
        "context_scales": {
            row["pair_id"]: float(candidate.context_scales[index])
            for index, row in enumerate(selected_relations)
        },
        "seed": args.seed,
        "class_count": args.class_count,
        "headval_source": headval_source,
        "head_train_count": len(head_train_features),
        "head_val_count": len(head_val_features),
        "b0_all_map": macro_average_precision(val_labels_np, b0_logits),
        "b1_all_map": macro_average_precision(val_labels_np, b1_logits),
        "b1_selected_macro_ap": macro_average_precision(val_labels_np, b1_logits, target_ids),
        "b2_all_map": macro_average_precision(val_labels_np, b2_logits),
        "b2_selected_macro_ap": macro_average_precision(val_labels_np, b2_logits, target_ids),
        "context_coefficients": {
            row["pair_id"]: float(
                config["training"]["residual_bound"]
                * torch.tanh(candidate.context_raw[index]).detach()
            )
            for index, row in enumerate(selected_relations)
        },
        "opportunity": {
            "best_global_bce": best_global_loss,
            "oracle_bce": oracle_loss,
            "oracle_relative_improvement": (
                (best_global_loss - oracle_loss) / best_global_loss
                if best_global_loss > 0
                else None
            ),
            "beneficial_fraction": float(np.mean(differences > epsilon)),
            "harmful_fraction": float(np.mean(differences < -epsilon)),
            "ignored_fraction": float(np.mean(np.abs(differences) <= epsilon)),
            "global_context_enabled": {
                selected_relations[index]["pair_id"]: bool(value)
                for index, value in enumerate(use_context_globally)
            },
            "passes_preregistered_threshold": bool(
                best_global_loss > 0
                and (best_global_loss - oracle_loss) / best_global_loss >= 0.02
                and np.mean(differences > epsilon) >= 0.10
                and np.mean(differences < -epsilon) >= 0.10
            ),
        },
        "best_head_epoch": best_epoch,
        "elapsed_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "base_head": base_head.state_dict(),
            "full_head": full_head.state_dict(),
            "candidate": candidate.state_dict(),
            "relations": [relation.__dict__ for relation in relations],
            "base_view_indices": list(base_view_indices),
            "context_selection": args.context_selection,
            "context_centering": args.context_centering,
            "context_centers": candidate.context_centers.detach().tolist(),
            "context_scales": candidate.context_scales.detach().tolist(),
            "context_signal": args.context_signal,
            "seed": args.seed,
        },
        run_dir / "candidates.pt",
    )
    for method, logits in (
        ("B0", b0_logits),
        ("B1", b1_logits),
        ("B2", b2_logits),
        ("O1", oracle_logits),
    ):
        np.savez_compressed(
            run_dir / f"predictions_{method}.npz",
            sample_id=val_ids,
            logits=logits.astype(np.float32),
            method=np.asarray(method),
        )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "config_resolved.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "environment.json").write_text(
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
    (run_dir / "git_commit.txt").write_text((git_commit() or "UNCOMMITTED") + "\n")
    (run_dir / "manifest_hashes.json").write_text(
        json.dumps(
            {
                "fit": sha256(fit_manifest),
                "modelval": sha256(modelval_manifest),
                "relations": sha256(relation_manifest),
            },
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
