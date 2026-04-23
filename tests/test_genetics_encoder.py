import numpy as np
import pytest
import torch

from src.models.genetics_encoder import GeneticsEncoder, GeneticsEncoderConfig


def test_genetics_encoder_forward_various_in_dim():
    for in_dim in (1, 7, 128):
        enc = GeneticsEncoder.from_dims(
            in_dim=in_dim, hidden_dim=32, out_dim=16, num_hidden_blocks=2
        )
        b = 5
        x = torch.randn(b, in_dim)
        y = enc(x)
        assert y.shape == (b, 16)


def test_genetics_encoder_rejects_mismatched_feature_dim():
    enc = GeneticsEncoder.from_dims(10, 16, 8, num_hidden_blocks=1)
    with pytest.raises(ValueError, match="Expected in_dim"):
        enc(torch.randn(3, 9))


def test_config_validation():
    with pytest.raises(ValueError):
        GeneticsEncoderConfig(in_dim=0, out_dim=8)


def test_batchnorm_training_eval():
    enc = GeneticsEncoder.from_dims(4, 8, 2, num_hidden_blocks=2)
    enc.train()
    x = torch.randn(8, 4)
    y1 = enc(x)
    enc.eval()
    y2 = enc(x)
    assert y1.shape == y2.shape == (8, 2)
    assert np.isfinite(y2.detach().numpy()).all()
