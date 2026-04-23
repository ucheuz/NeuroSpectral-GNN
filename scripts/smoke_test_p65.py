#!/usr/bin/env python
"""Smoke test for KCL P65: cross-modal MHA encoder + 3D SLIC (synthetic voxels only).

  python scripts/smoke_test_p65.py

Requires: torch, torch-geometric, scikit-image (for SLIC). NIfTI not required.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _test_cross_modal_siamese() -> None:
    import torch
    from torch_geometric.data import Data

    from src.models.siamese_gnn import SiameseBrainNet, SiameseConfig

    cfg = SiameseConfig(
        in_channels=16,
        modality_feature_dims=(4, 4, 8),
        use_cross_modal_attention=True,
        cross_modal_d_model=32,
        cross_modal_num_heads=4,
        hidden_channels=16,
        num_layers=2,
    )
    m = SiameseBrainNet(cfg)
    m.eval()  # single-graph batch of 1 — BatchNorm needs eval or batch>1
    n = 6
    x = torch.randn(n, 16)
    edge_index = torch.tensor(
        [[i, (i + 1) % n] for i in range(n)]
        + [[(i + 1) % n, i] for i in range(n)],
        dtype=torch.long,
    ).T.contiguous()
    a = Data(x=x, edge_index=edge_index)
    b = Data(x=x, edge_index=edge_index)
    za, zb = m(a, b)
    assert za.shape[0] == 1 and zb.shape[0] == 1
    enc = m.encoder
    _, att = enc(x, edge_index, return_modality_attn=True)
    assert att is not None and att.shape[0] == n
    print("  cross-modal Siamese forward + modality attention: OK")


def _test_slic_3d() -> None:
    import numpy as np

    from src.preprocessing.slic_supervoxels import SlicNiftiConfig, run_slic_on_volume

    rs = np.random.RandomState(0)
    # Small 4D stack (Z, Y, X, C) — e.g. FA, MD, T1-like, FLAIR-like channels
    vol = rs.standard_normal(size=(32, 32, 32, 4)).astype(np.float32)
    lab = run_slic_on_volume(
        vol,
        SlicNiftiConfig(
            n_segments=100,
            compactness=0.1,
            enforce_connectivity=True,
        ),
    )
    assert lab.shape == (32, 32, 32)
    n_labels = int(lab.max())
    assert n_labels >= 1 and int(lab.min()) >= 0
    print(f"  SLIC 3D label map: shape {lab.shape}, n_seg≈1..{n_labels}: OK")


def main() -> int:
    print("P65 smoke test")
    _test_cross_modal_siamese()
    _test_slic_3d()
    print("All P65 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
