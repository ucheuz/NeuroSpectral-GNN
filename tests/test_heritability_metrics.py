import numpy as np
import torch

from src.analysis.heritability import twin_separation_metrics


def test_twin_separation_includes_unrel():
    n = 6
    z_a = torch.randn(n, 8)
    z_b = torch.randn(n, 8)
    zyg = ["MZ", "DZ", "UNREL", "MZ", "DZ", "UNREL"]
    m = twin_separation_metrics(z_a, z_b, zyg, distance_metric="cosine")
    assert "mean_unrel_distance" in m
    assert m["n_unrel"] == 2
    assert m["n_mz"] == 2
    assert m["n_dz"] == 2
    assert not np.isnan(m["mean_unrel_distance"])


def test_twin_separation_mz_dz_only():
    z = torch.eye(4)
    zyg = ["MZ", "DZ", "MZ", "DZ"]
    m = twin_separation_metrics(z, z, zyg)
    assert m["n_unrel"] == 0
    assert np.isnan(m["mean_unrel_distance"])
