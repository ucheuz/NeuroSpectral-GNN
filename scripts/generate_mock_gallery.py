#!/usr/bin/env python
"""Dissertation-style mock gallery: three twin-pair scenarios + saliency + NIfTIs.

Builds **synthetic** graphs and a toy label NIfTI, runs gradient saliency and
the reverse-mapper, and saves a comparison PNG plus per-case scalar NIfTIs.

Scenarios (illustrative, not inferential):
  - **mz_high**: MZ pair, nearly identical node features + shared PRS signal
  - **dz_low**: DZ pair, partial sharing
  - **unrelated**: high distance, uncorrelated features

MPS-safe: standard torch + numpy I/O.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.siamese_gnn import ContrastiveLoss, SiameseConfig, build_siamese_model
from src.utils.device import get_device
from src.utils.saliency import gradient_saliency_contrastive_pair
from src.utils.synthetic_atlas import generate_synthetic_atlas_nifti
from src.utils.visualization import map_nodes_to_volume


@dataclass
class Scenario:
    key: str
    title: str
    label: int  # 0 = MZ (pull together), 1 = negative (DZ / UNREL)


def _ring_graph(n: int, feat_dim: int, rng: np.random.Generator) -> Data:
    idx = np.arange(n)
    src = np.concatenate([idx, (idx + 1) % n])
    dst = np.concatenate([(idx + 1) % n, idx])
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    x = torch.tensor(
        rng.standard_normal((n, feat_dim)), dtype=torch.float32
    )
    return Data(x=x, edge_index=edge_index)


def _make_pair(
    scenario: Scenario,
    n_nodes: int,
    feat_dim: int,
    prs_dim: int,
    rng: np.random.Generator,
) -> tuple[Data, Data]:
    base = _ring_graph(n_nodes, feat_dim, rng)
    if scenario.key == "mz_high":
        noise = torch.tensor(rng.standard_normal((n_nodes, feat_dim)) * 0.05, dtype=torch.float32)
        x_b = base.x + noise
        # Emphasise "genetic target" nodes for saliency / story
        tgt = min(4, n_nodes)
        x_b[:tgt, :] = base.x[:tgt, :] + noise[:tgt] * 0.02
        prs_a = torch.tensor(rng.standard_normal(prs_dim), dtype=torch.float32).view(1, -1)
        prs_b = prs_a + torch.tensor(rng.standard_normal(prs_dim) * 0.03, dtype=torch.float32).view(1, -1)
    elif scenario.key == "dz_low":
        x_b = 0.55 * base.x + torch.tensor(
            rng.standard_normal((n_nodes, feat_dim)) * 0.5, dtype=torch.float32
        )
        prs_a = torch.tensor(rng.standard_normal(prs_dim), dtype=torch.float32).view(1, -1)
        prs_b = 0.5 * prs_a + torch.tensor(
            rng.standard_normal(prs_dim) * 0.7, dtype=torch.float32
        ).view(1, -1)
    else:  # unrelated
        x_b = torch.tensor(
            rng.standard_normal((n_nodes, feat_dim)), dtype=torch.float32
        )
        prs_a = torch.tensor(rng.standard_normal(prs_dim), dtype=torch.float32).view(1, -1)
        prs_b = torch.tensor(rng.standard_normal(prs_dim), dtype=torch.float32).view(1, -1)

    da = Data(x=base.x, edge_index=base.edge_index, prs=prs_a)
    db = Data(x=x_b, edge_index=base.edge_index, prs=prs_b)
    return da, db


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=Path("runs/mock_gallery"))
    ap.add_argument("--n-nodes", type=int, default=32)
    ap.add_argument("--feat-dim", type=int, default=16)
    ap.add_argument("--prs-dim", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = get_device(args.device)

    vol_shape = (32, 32, 32)
    label_path = generate_synthetic_atlas_nifti(
        args.n_nodes, vol_shape, out / "mock_node_labels.nii.gz"
    )

    cfg = SiameseConfig(
        in_channels=args.feat_dim,
        hidden_channels=32,
        num_layers=2,
        dropout=0.1,
        prs_dim=args.prs_dim,
        prs_embed_dim=32,
        prs_hidden=32,
        model_type="multimodal",
        projection_dim=16,
        projection_hidden=32,
    )
    model = build_siamese_model(cfg).to(device)
    # eval: single-pair steps use batch-1 on ``prs``; BatchNorm1d in GeneticsEncoder
    # errors in training mode (needs >1 sample per channel). Saliency also expects eval.
    model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ContrastiveLoss(margin=1.0, metric="cosine")
    scenarios = [
        Scenario("mz_high", "MZ (high similarity)", 0),
        Scenario("dz_low", "DZ (partial)", 1),
        Scenario("unrelated", "Unrelated", 1),
    ]

    # Quick adaptation so saliency is non-degenerate
    for _ in range(12):
        for sc in scenarios:
            da, db = _make_pair(sc, args.n_nodes, args.feat_dim, args.prs_dim, rng)
            label = torch.tensor([sc.label], dtype=torch.long)
            za, zb = model(
                da.to(device).clone(),
                db.to(device).clone(),
            )
            loss = loss_fn(za, zb, label.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, sc in zip(axes, scenarios):
        da, db = _make_pair(sc, args.n_nodes, args.feat_dim, args.prs_dim, rng)
        label = torch.tensor([sc.label], dtype=torch.long)
        sal = gradient_saliency_contrastive_pair(
            model,  # type: ignore[arg-type]
            da,
            db,
            label,
            device,
            loss_fn=ContrastiveLoss(margin=1.0, metric="cosine"),
            twin="a",
        )
        nii_path = out / f"mock_saliency_{sc.key}.nii.gz"
        map_nodes_to_volume(sal, label_path, nii_path, fill_background=0.0)
        ax.bar(np.arange(len(sal)), sal, width=0.85, color="steelblue")
        ax.set_title(sc.title)
        ax.set_xlabel("node (supervoxel)")
        ax.set_ylabel("|grad|")
    fig.suptitle("Mock saliency (twin A) — illustrative", fontsize=11)
    fig.tight_layout()
    png = out / "mock_gallery_saliency.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    print("Wrote:", label_path)
    print("Wrote:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
