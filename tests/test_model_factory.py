"""Tests for Siamese model factory and genetics-only path."""
import pytest
import torch
from torch_geometric.data import Data, Batch

from src.models.siamese_gnn import (
    SiameseConfig,
    build_siamese_model,
)


def test_build_graph_default():
    cfg = SiameseConfig(
        in_channels=32, hidden_channels=16, prs_dim=0, model_type="graph"
    )
    m = build_siamese_model(cfg)
    assert m.__class__.__name__ == "SiameseBrainNet"


def test_build_fused_alias_matches_multimodal():
    cfg = SiameseConfig(
        in_channels=32,
        hidden_channels=16,
        prs_dim=8,
        prs_embed_dim=16,
        model_type="fused",
    )
    m = build_siamese_model(cfg)
    assert m.__class__.__name__ == "MultimodalSiameseBrainNet"


def test_build_multimodal_auto():
    cfg = SiameseConfig(
        in_channels=32,
        hidden_channels=16,
        prs_dim=8,
        prs_embed_dim=16,
        model_type="auto",
    )
    m = build_siamese_model(cfg)
    assert m.__class__.__name__ == "MultimodalSiameseBrainNet"


def _single_graph_prs(prs: torch.Tensor) -> Data:
    return Data(
        x=torch.zeros(1, 2),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        prs=prs,
    )


def test_build_genetics_only():
    cfg = SiameseConfig(
        in_channels=32,
        hidden_channels=16,
        prs_dim=8,
        prs_embed_dim=16,
        model_type="genetics_only",
    )
    m = build_siamese_model(cfg)
    assert m.__class__.__name__ == "GeneticsOnlySiameseNet"
    b = 4
    prs_a = torch.randn(b, 8)
    prs_b = torch.randn(b, 8)
    da = Batch.from_data_list(
        [_single_graph_prs(prs_a[i : i + 1]) for i in range(b)]
    )
    db = Batch.from_data_list(
        [_single_graph_prs(prs_b[i : i + 1]) for i in range(b)]
    )
    za, zb = m(da, db)
    assert za.shape[0] == b and zb.shape[0] == b
