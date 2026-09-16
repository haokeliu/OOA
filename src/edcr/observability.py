"""Gain-regression models for observable-opportunity experiments."""

from __future__ import annotations

import torch
from torch import nn


class LogitGainMLP(nn.Module):
    """Shared gain regressor over standardized view-logit features."""

    def __init__(
        self, input_dim: int, relation_count: int, hidden: int = 64, relation_dim: int = 8
    ) -> None:
        super().__init__()
        self.relation_embedding = nn.Embedding(relation_count, relation_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim + relation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor, relation_indices: torch.Tensor) -> torch.Tensor:
        relation = self.relation_embedding(relation_indices)
        if features.ndim == 3 and relation.ndim == 2:
            relation = relation.unsqueeze(0).expand(features.shape[0], -1, -1)
        return self.network(torch.cat((features, relation), dim=-1)).squeeze(-1)


class RelationConditionedViewGain(nn.Module):
    """Predict gain from query-conditioned interactions with frozen image views."""

    def __init__(
        self,
        feature_dim: int,
        view_count: int,
        numeric_dim: int,
        projection_dim: int = 32,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.view_count = view_count
        self.view_projection = nn.Linear(feature_dim, projection_dim, bias=False)
        self.query_projection = nn.Linear(feature_dim, projection_dim, bias=False)
        interaction_dim = 2 * view_count * projection_dim
        self.network = nn.Sequential(
            nn.Linear(interaction_dim + numeric_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        views: torch.Tensor,
        numeric: torch.Tensor,
        source_queries: torch.Tensor,
        target_queries: torch.Tensor,
    ) -> torch.Tensor:
        if views.ndim != 3 or views.shape[1] != self.view_count:
            raise ValueError("views must have shape [batch, configured_views, feature_dim]")
        projected_views = self.view_projection(views)
        source = self.query_projection(source_queries)
        target = self.query_projection(target_queries)
        if numeric.ndim == 3 and source.ndim == 2:
            projected_views = projected_views.unsqueeze(1)
            source = source.unsqueeze(0).unsqueeze(2)
            target = target.unsqueeze(0).unsqueeze(2)
            interactions = torch.cat(
                (projected_views * source, projected_views * target), dim=2
            )
            interactions = interactions.flatten(start_dim=2)
        else:
            source = source.unsqueeze(1)
            target = target.unsqueeze(1)
            interactions = torch.cat(
                (projected_views * source, projected_views * target), dim=1
            ).flatten(start_dim=1)
        inputs = torch.cat((numeric, interactions), dim=-1)
        return self.network(inputs).squeeze(-1)


class DenseMapGain(nn.Module):
    """Predict gain from fixed source/target CLIP patch-similarity maps."""

    def __init__(
        self,
        relation_count: int,
        numeric_dim: int,
        channels: int = 16,
        hidden: int = 64,
        relation_dim: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.average_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.maximum_pool = nn.AdaptiveMaxPool2d((2, 2))
        self.relation_embedding = nn.Embedding(relation_count, relation_dim)
        spatial_dim = channels * 2 * 2 * 2
        self.network = nn.Sequential(
            nn.Linear(spatial_dim + numeric_dim + relation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        maps: torch.Tensor,
        numeric: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        if maps.ndim not in {4, 5} or maps.shape[-3] != 2:
            raise ValueError("maps must have shape [pairs,2,H,W] or [batch,relations,2,H,W]")
        relation = self.relation_embedding(relation_indices)
        if maps.ndim == 5:
            batch, relation_count = maps.shape[:2]
            encoded = self.encoder(maps.flatten(0, 1))
            pooled = torch.cat(
                (self.average_pool(encoded), self.maximum_pool(encoded)), dim=1
            ).flatten(start_dim=1)
            pooled = pooled.reshape(batch, relation_count, -1)
            relation = relation.unsqueeze(0).expand(batch, -1, -1)
        else:
            encoded = self.encoder(maps)
            pooled = torch.cat(
                (self.average_pool(encoded), self.maximum_pool(encoded)), dim=1
            ).flatten(start_dim=1)
        return self.network(torch.cat((pooled, numeric, relation), dim=-1)).squeeze(-1)
