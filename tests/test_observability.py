import pytest
import torch

from edcr.observability import DenseMapGain, LogitGainMLP, RelationConditionedViewGain


def test_logit_gain_mlp_returns_one_gain_per_pair() -> None:
    model = LogitGainMLP(input_dim=18, relation_count=5)
    output = model(torch.randn(7, 18), torch.tensor([0, 1, 2, 3, 4, 0, 1]))
    assert output.shape == (7,)


def test_relation_conditioned_view_gain_shapes() -> None:
    model = RelationConditionedViewGain(16, view_count=5, numeric_dim=6)
    output = model(
        torch.randn(7, 5, 16),
        torch.randn(7, 6),
        torch.randn(7, 16),
        torch.randn(7, 16),
    )
    assert output.shape == (7,)


def test_relation_conditioned_view_gain_rejects_wrong_view_count() -> None:
    model = RelationConditionedViewGain(16, view_count=5, numeric_dim=6)
    with pytest.raises(ValueError, match="configured_views"):
        model(
            torch.randn(7, 4, 16),
            torch.randn(7, 6),
            torch.randn(7, 16),
            torch.randn(7, 16),
        )


def test_gain_models_support_image_by_relation_batches() -> None:
    logit_model = LogitGainMLP(input_dim=18, relation_count=5)
    assert logit_model(torch.randn(7, 5, 18), torch.arange(5)).shape == (7, 5)
    view_model = RelationConditionedViewGain(16, view_count=5, numeric_dim=6)
    output = view_model(
        torch.randn(7, 5, 16),
        torch.randn(7, 5, 6),
        torch.randn(5, 16),
        torch.randn(5, 16),
    )
    assert output.shape == (7, 5)


def test_dense_map_gain_supports_image_by_relation_batches() -> None:
    model = DenseMapGain(relation_count=3, numeric_dim=6)
    output = model(torch.randn(4, 3, 2, 14, 14), torch.randn(4, 3, 6), torch.arange(3))
    assert output.shape == (4, 3)


def test_dense_map_gain_rejects_missing_source_target_channels() -> None:
    model = DenseMapGain(relation_count=3, numeric_dim=6)
    with pytest.raises(ValueError, match="maps must have shape"):
        model(torch.randn(4, 3, 14, 14), torch.randn(4, 3, 6), torch.arange(3))
