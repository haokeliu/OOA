"""Inference-available gate features and target-balanced risk utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from edcr.models import CandidateModel


@dataclass(frozen=True)
class GateTensors:
    numeric: torch.Tensor
    source_embeddings: torch.Tensor
    target_embeddings: torch.Tensor
    base: torch.Tensor
    residual: torch.Tensor
    target_labels: torch.Tensor
    source_labels: torch.Tensor


class ConstantGate(nn.Module):
    """One learned soft gate value per relation."""

    def __init__(self, relation_count: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(relation_count))

    def forward(self, relation_indices: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits[relation_indices])


class ConfidenceGate(nn.Module):
    """Target-confidence-only baseline with the same hidden width as the full gate."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, confidence: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(confidence.unsqueeze(-1))).squeeze(-1)


def build_gate_tensors(
    model: CandidateModel,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> GateTensors:
    """Build protocol features without exposing labels to the gate inputs."""
    if features.ndim != 3 or labels.ndim != 2 or len(features) != len(labels):
        raise ValueError("features and labels must be [N,V,D] and [N,C]")
    with torch.inference_mode():
        view_logits = model.base_head.view_logits(features)
        full_logits = model.full_head(features[:, 0])
        outputs = model(features)
        numeric_rows: list[torch.Tensor] = []
        source_embeddings: list[torch.Tensor] = []
        target_embeddings: list[torch.Tensor] = []
        base_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        target_label_rows: list[torch.Tensor] = []
        source_label_rows: list[torch.Tensor] = []
        class_embeddings = functional.normalize(model.base_head.linear.weight, dim=-1)
        for relation in model.relations:
            target_views = view_logits[:, :, relation.target]
            numeric_rows.append(
                torch.stack(
                    (
                        target_views.mean(dim=1),
                        target_views.max(dim=1).values,
                        target_views.std(dim=1, unbiased=False),
                        torch.sigmoid(full_logits[:, relation.target]),
                        torch.sigmoid(full_logits[:, relation.source]),
                        outputs["residual"][:, relation.target],
                    ),
                    dim=-1,
                )
            )
            source_embeddings.append(class_embeddings[relation.source])
            target_embeddings.append(class_embeddings[relation.target])
            base_rows.append(outputs["base"][:, relation.target])
            residual_rows.append(outputs["residual"][:, relation.target])
            target_label_rows.append(labels[:, relation.target])
            source_label_rows.append(labels[:, relation.source])
    return GateTensors(
        numeric=torch.stack(numeric_rows, dim=1),
        source_embeddings=torch.stack(source_embeddings),
        target_embeddings=torch.stack(target_embeddings),
        base=torch.stack(base_rows, dim=1),
        residual=torch.stack(residual_rows, dim=1),
        target_labels=torch.stack(target_label_rows, dim=1),
        source_labels=torch.stack(source_label_rows, dim=1),
    )


def natural_group_masks(
    tensors: GateTensors, weak_presence: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if weak_presence.shape != tensors.target_labels.shape:
        raise ValueError("weak_presence must match [samples, relations]")
    negative = (tensors.target_labels == 0) & (tensors.source_labels == 1)
    weak = (tensors.target_labels == 1) & weak_presence.bool()
    return negative, weak


def target_balanced_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average per relation first, then average relations with complete support."""
    if logits.shape != labels.shape or logits.ndim != 2:
        raise ValueError("logits and labels must have equal [samples, relations] shape")
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    if mask.shape != labels.shape:
        raise ValueError("mask shape differs")
    losses = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    per_relation: list[torch.Tensor] = []
    for relation_index in range(logits.shape[1]):
        selected = mask[:, relation_index]
        if not selected.any():
            raise ValueError(f"relation {relation_index} has no group support")
        per_relation.append(losses[selected, relation_index].mean())
    return torch.stack(per_relation).mean()


def gate_values(
    method: str,
    gate: nn.Module,
    tensors: GateTensors,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one gate checkpoint without exposing group or label metadata."""
    relation_count = tensors.numeric.shape[1]
    relation_indices = flat_indices % relation_count
    if method == "B3":
        return gate(relation_indices)
    sample_indices = flat_indices // relation_count
    if method in {"B4", "B4C", "B4G"}:
        confidence = torch.sigmoid(tensors.base[sample_indices, relation_indices])
        return gate(confidence)
    return gate(
        tensors.numeric[sample_indices, relation_indices],
        tensors.source_embeddings[relation_indices],
        tensors.target_embeddings[relation_indices],
    )


def predict_gate(
    method: str,
    gate: nn.Module,
    tensors: GateTensors,
    batch_size: int,
) -> torch.Tensor:
    """Return an [images, relations] matrix of soft gate values."""
    total = tensors.base.numel()
    chunks: list[torch.Tensor] = []
    gate.eval()
    with torch.inference_mode():
        for offset in range(0, total, batch_size):
            indices = torch.arange(offset, min(offset + batch_size, total))
            chunks.append(gate_values(method, gate, tensors, indices))
    return torch.cat(chunks).reshape_as(tensors.base)
