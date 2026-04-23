#!/usr/bin/env python
"""End-to-end smoke test: synthetic cohort -> DataLoader -> SiameseBrainNet
-> ContrastiveLoss -> backward pass on the best available device.

Validates the full training-loop plumbing before real data arrives.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.models import (  # noqa: E402
    ContrastiveLoss,
    HeritabilityAuxLoss,
    SiameseConfig,
    build_siamese_model,
)
from src.preprocessing.graph import GraphBuildConfig  # noqa: E402
from src.preprocessing.synthetic import SyntheticCohortConfig, save_synthetic_cohort  # noqa: E402
from src.utils import TwinBrainDataset, get_device, set_seed, twin_collate  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("smoke_test_model")
    set_seed(0)
    device = get_device()

    with tempfile.TemporaryDirectory(prefix="nsgnn_model_smoke_") as tmp:
        tmp = Path(tmp)
        log.info("Generating tiny synthetic cohort...")
        cfg = SyntheticCohortConfig(
            n_mz_pairs=8,
            n_dz_pairs=8,
            n_rois=64,
            heritability=0.7,
            seed=0,
            graph_config=GraphBuildConfig(
                sparsify_strategy="proportional",
                keep_top_fraction=0.2,
                node_feature_mode="profile",
            ),
        )
        save_synthetic_cohort(cfg, tmp)

        ds = TwinBrainDataset(tmp, include_zygosities={"MZ", "DZ"}, preload=True)
        loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=twin_collate)
        log.info("Dataset: %d pairs, %d batches", len(ds), len(loader))

        mcfg = SiameseConfig(
            in_channels=cfg.n_rois,
            hidden_channels=32,
            num_layers=3,
            projection_dim=16,
            projection_hidden=32,
        )
        model = build_siamese_model(mcfg).to(device)
        loss_fn = ContrastiveLoss(margin=1.0, metric="cosine")
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            z_a, z_b = model(batch.data_a, batch.data_b)
            loss = loss_fn(z_a, z_b, batch.label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += float(loss.detach().cpu())
        log.info("Mean train loss over smoke pass: %.4f", total / len(loader))

        # Quick sanity: MZ distances should be <= DZ distances on average even
        # after just the random init, because the synthetic generator shares
        # the additive-genetic component perfectly for MZ. Let's check.
        model.eval()
        with torch.no_grad():
            mz_d: list[float] = []
            dz_d: list[float] = []
            for batch in DataLoader(ds, batch_size=32, collate_fn=twin_collate):
                batch = batch.to(device)
                z_a = model.encode(batch.data_a)
                z_b = model.encode(batch.data_b)
                dists = loss_fn.distance(z_a, z_b).cpu().tolist()
                for d, zyg in zip(dists, batch.zygosities):
                    (mz_d if zyg == "MZ" else dz_d).append(d)
        log.info(
            "Mean cosine distance: MZ=%.4f | DZ=%.4f | (MZ<DZ? %s)",
            sum(mz_d) / max(len(mz_d), 1),
            sum(dz_d) / max(len(dz_d), 1),
            (sum(mz_d) / max(len(mz_d), 1)) < (sum(dz_d) / max(len(dz_d), 1)),
        )

    # ------------------------------------------------------------------
    # Phase 2: multimodal PRS + heritability-aux-loss smoke test
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="nsgnn_mm_smoke_") as tmp2:
        tmp2 = Path(tmp2)
        log.info("Generating multimodal synthetic cohort (with PRS)...")
        mm_cfg = SyntheticCohortConfig(
            n_mz_pairs=8,
            n_dz_pairs=8,
            n_rois=64,
            heritability=0.7,
            prs_dim=16,
            prs_informativeness=1.0,
            seed=1,
            graph_config=GraphBuildConfig(
                sparsify_strategy="proportional",
                keep_top_fraction=0.2,
                node_feature_mode="profile",
            ),
        )
        save_synthetic_cohort(mm_cfg, tmp2)

        ds_mm = TwinBrainDataset(tmp2, include_zygosities={"MZ", "DZ"}, preload=True)
        loader_mm = DataLoader(
            ds_mm, batch_size=4, shuffle=True, collate_fn=twin_collate
        )

        mm_mcfg = SiameseConfig(
            in_channels=mm_cfg.n_rois,
            hidden_channels=32,
            num_layers=3,
            projection_dim=16,
            projection_hidden=32,
            prs_dim=mm_cfg.prs_dim,
            prs_hidden=32,
            prs_embed_dim=32,
            prs_fusion="concat",
        )
        mm_model = build_siamese_model(mm_mcfg).to(device)
        log.info("Multimodal model: %s (params=%d)",
                 type(mm_model).__name__,
                 sum(p.numel() for p in mm_model.parameters()))
        aux_loss_fn = HeritabilityAuxLoss(target_h2=mm_cfg.heritability).to(device)
        aux_weight = 0.1
        optim = torch.optim.AdamW(mm_model.parameters(), lr=1e-3)

        mm_model.train()
        total_c, total_a, n_valid = 0.0, 0.0, 0
        for batch in loader_mm:
            batch = batch.to(device)
            z_a_enc = mm_model.encode(batch.data_a)
            z_b_enc = mm_model.encode(batch.data_b)
            z_a_proj = mm_model.project(z_a_enc)
            z_b_proj = mm_model.project(z_b_enc)
            contrastive = loss_fn(z_a_proj, z_b_proj, batch.label)
            aux, valid = aux_loss_fn(z_a_enc, z_b_enc, batch.zygosities)
            combined = contrastive + aux_weight * aux
            optim.zero_grad()
            combined.backward()
            optim.step()
            total_c += float(contrastive.detach().cpu())
            if valid:
                total_a += float(aux.detach().cpu())
                n_valid += 1
        log.info(
            "Multimodal smoke pass: contrastive=%.4f | aux_h2=%.4f over %d valid batches",
            total_c / len(loader_mm),
            total_a / max(n_valid, 1),
            n_valid,
        )

    log.info("Model smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
