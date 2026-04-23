"""Cross-modal MHA in BrainGNNEncoder (KCL P65-style)."""
import pytest
import torch
from torch_geometric.data import Data

from src.models.siamese_gnn import BrainGNNEncoder, ModalityCrossAttentionBlock, SiameseConfig


def test_modality_mha_block_shapes():
    blk = ModalityCrossAttentionBlock((4, 4, 8), d_model=32, n_heads=4, dropout=0.0)
    x = torch.randn(12, 16)  # 12 supervoxels
    y = blk(x, return_attn=False)
    assert y.shape == (12, 32)
    y2, w = blk(x, return_attn=True)
    assert y2.shape == (12, 32)
    assert w is not None
    assert w.shape[0] == 12  # per-node attention
    assert w.shape[1:] == (3, 3)  # modalities x modalities


def test_brain_gnn_encoder_cross_modal_forward():
    cfg = SiameseConfig(
        in_channels=16,
        modality_feature_dims=(4, 4, 8),
        use_cross_modal_attention=True,
        cross_modal_d_model=32,
        cross_modal_num_heads=4,
        hidden_channels=16,
        num_layers=2,
    )
    enc = BrainGNNEncoder(cfg)
    x = torch.randn(5, 16)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0]], dtype=torch.long
    )
    out = enc(x, edge_index)
    assert out.shape[0] == 1  # one graph
    assert out.shape[1] == enc.graph_embed_dim

    out2, att = enc(
        x, edge_index, return_modality_attn=True, return_node_features=False
    )
    assert out2.shape == out.shape
    assert att is not None and att.shape[0] == 5

    out3, nodes, att2 = enc(
        x, edge_index, return_modality_attn=True, return_node_features=True
    )
    assert nodes.shape[0] == 5
    assert att2 is not None


def test_brain_gnn_default_no_regression():
    """Single-block ``in_channels`` path matches previous behaviour (no MHA)."""
    cfg = SiameseConfig(
        in_channels=8,
        hidden_channels=4,
        num_layers=1,
    )
    enc = BrainGNNEncoder(cfg)
    assert enc.modality_mha is None
    x = torch.randn(3, 8)
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    o = enc(x, ei)
    assert o.shape == (1, enc.graph_embed_dim)
