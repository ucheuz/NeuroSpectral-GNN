"""Reverse-map, saliency, and bar-chart helpers (KCL P65)."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.models.siamese_gnn import SiameseBrainNet, SiameseConfig
from src.utils.saliency import (
    gradient_saliency_contrastive_pair,
    integrated_gradients_contrastive_pair,
)
from src.utils.visualization import (
    map_nodes_to_volume,
    plot_modality_importance_barchart,
    pooled_modality_query_importance,
)
from src.utils import get_device


def test_map_nodes_to_volume_and_roundtrip_nibabel():
    nib = pytest.importorskip("nibabel")
    a = np.zeros((8, 8, 8), dtype=np.int16)
    a[1:4, 1:4, 1:4] = 1
    a[5:7, 5:7, 5:7] = 2
    with tempfile.TemporaryDirectory() as td:
        lab_p = Path(td) / "lab.nii.gz"
        out_p = Path(td) / "heat.nii.gz"
        im = nib.Nifti1Image(a, np.eye(4))
        nib.save(im, str(lab_p))
        v = np.array([0.1, 0.9], dtype=np.float32)
        map_nodes_to_volume(v, lab_p, out_p, fill_background=0.0)
        o = np.asarray(nib.load(str(out_p)).dataobj, dtype=np.float32)
        assert o[a == 1].mean() == pytest.approx(0.1, abs=1e-3)
        assert o[a == 2].mean() == pytest.approx(0.9, abs=1e-3)


def test_pooled_modality_and_bar_chart():
    a = np.ones((3, 3, 3), dtype=np.float32) / 9.0
    p = pooled_modality_query_importance(a, pool="rowsum")
    assert p.shape == (3,) and p.sum() == pytest.approx(1.0, abs=0.01)
    with tempfile.TemporaryDirectory() as td:
        plot_modality_importance_barchart(
            p, p, ["A", "B", "C"], Path(td) / "m.png", title="t"
        )
        assert (Path(td) / "m.png").is_file()


def test_saliency_shapes_with_siamese():
    cfg = SiameseConfig(
        in_channels=8,
        modality_feature_dims=(4, 4),
        use_cross_modal_attention=True,
        cross_modal_d_model=16,
        cross_modal_num_heads=2,
        hidden_channels=8,
        num_layers=1,
    )
    m = SiameseBrainNet(cfg).eval()
    n = 5
    x = torch.randn(n, 8)
    ei = torch.tensor([[0, 1, 1, 2, 2], [1, 0, 2, 0, 1]], dtype=torch.long)
    a = Data(x=x, edge_index=ei, edge_attr=torch.zeros(ei.size(1)))
    b = a.clone()
    dev = get_device("cpu", verbose=False)
    lab = torch.zeros(1, dtype=torch.long)
    g = gradient_saliency_contrastive_pair(
        m, a, b, lab, dev, twin="a"
    )
    assert g.shape == (n,)
    ig = integrated_gradients_contrastive_pair(
        m, a, b, lab, dev, n_steps=4, twin="a"
    )
    assert ig.shape == (n,)


def test_encode_modality_attention_runs():
    cfg = SiameseConfig(
        in_channels=8,
        modality_feature_dims=(4, 4),
        use_cross_modal_attention=True,
        cross_modal_d_model=16,
        cross_modal_num_heads=2,
        hidden_channels=8,
        num_layers=1,
    )
    m = SiameseBrainNet(cfg).eval()
    n = 4
    x = torch.randn(n, 8)
    ei = torch.tensor(
        [[i, (i + 1) % n] for i in range(n)]
        + [[(i + 1) % n, i] for i in range(n)],
        dtype=torch.long,
    ).T.contiguous()
    d = Data(x=x, edge_index=ei)
    att = m.encode_modality_attention(d)
    assert att is not None
    assert att.shape == (n, 2, 2)
