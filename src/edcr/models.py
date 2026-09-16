"""Frozen-feature candidate and gate models from protocol v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def smooth_max_pool(view_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Pool BxVxC logits only after independent view scoring."""
    if view_logits.ndim != 3:
        raise ValueError("view_logits must have shape [batch, views, classes]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    view_count = view_logits.shape[1]
    return temperature * (
        torch.logsumexp(view_logits / temperature, dim=1)
        - torch.log(view_logits.new_tensor(float(view_count)))
    )


class MultiViewHead(nn.Module):
    def __init__(self, feature_dim: int, class_count: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, class_count)
        self.temperature = temperature

    def view_logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, views, feature_dim]")
        return self.linear(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return smooth_max_pool(self.view_logits(features), self.temperature)


@dataclass(frozen=True)
class RelationIndex:
    source: int
    target: int


class CandidateModel(nn.Module):
    """Compute frozen base logits and bounded directed context residuals."""

    def __init__(
        self,
        base_head: MultiViewHead,
        full_head: nn.Linear,
        relations: list[RelationIndex],
        residual_bound: float = 2.0,
        base_view_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
        context_centers: torch.Tensor | None = None,
        context_scales: torch.Tensor | None = None,
        context_signal: str = "probability",
    ) -> None:
        super().__init__()
        self.base_head = base_head
        self.full_head = full_head
        self.relations = tuple(relations)
        self.residual_bound = residual_bound
        if not base_view_indices:
            raise ValueError("base_view_indices must not be empty")
        self.base_view_indices = tuple(base_view_indices)
        centers = (
            torch.zeros(len(relations), dtype=torch.float32)
            if context_centers is None
            else context_centers.detach().to(dtype=torch.float32).clone()
        )
        if centers.shape != (len(relations),):
            raise ValueError("context_centers must have one value per relation")
        self.register_buffer("context_centers", centers, persistent=False)
        scales = (
            torch.ones(len(relations), dtype=torch.float32)
            if context_scales is None
            else context_scales.detach().to(dtype=torch.float32).clone()
        )
        if scales.shape != (len(relations),) or torch.any(scales <= 0):
            raise ValueError("context_scales must be positive with one value per relation")
        if context_signal not in {"probability", "standardized_logit"}:
            raise ValueError("unsupported context_signal")
        self.register_buffer("context_scales", scales, persistent=False)
        self.context_signal = context_signal
        self.context_raw = nn.Parameter(torch.zeros(len(relations)))

    def freeze_heads(self) -> None:
        self.base_head.requires_grad_(False)
        self.full_head.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.base_head(features[:, self.base_view_indices])
        full_logits = self.full_head(features[:, 0])
        residual = torch.zeros_like(base)
        for relation_index, relation in enumerate(self.relations):
            source_logit = full_logits[:, relation.source]
            if self.context_signal == "probability":
                source_signal = torch.sigmoid(source_logit) - self.context_centers[relation_index]
            else:
                source_signal = torch.tanh(
                    (source_logit - self.context_centers[relation_index])
                    / self.context_scales[relation_index]
                )
            correction = (
                self.residual_bound * torch.tanh(self.context_raw[relation_index]) * source_signal
            )
            residual[:, relation.target] = correction
        return {"base": base, "residual": residual, "full": base + residual}


class GateNetwork(nn.Module):
    """Shared two-layer soft gate; group metadata is never an input."""

    def __init__(self, numeric_dim: int, label_embedding_dim: int, hidden: int = 32) -> None:
        super().__init__()
        input_dim = numeric_dim + 2 * label_embedding_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((numeric_features, source_embedding, target_embedding), dim=-1)
        return torch.sigmoid(self.network(inputs)).squeeze(-1)


def apply_gate(base: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if base.shape != residual.shape:
        raise ValueError("base and residual shapes differ")
    if gate.ndim == 1:
        gate = gate.unsqueeze(-1)
    return base + gate * residual
