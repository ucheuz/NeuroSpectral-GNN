"""Core training loop: per-fold trainer + cross-validation driver.

Decoupled from the CLI so it can be unit-tested and reused by the h^2 sweep
script (``scripts/run_h2_sweep.py``) without spawning subprocesses.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.analysis.heritability import (
    HeritabilityEstimate,
    falconer_h2,
    pair_similarities_from_embeddings,
    twin_separation_metrics,
)
from src.analysis.splits import family_stratified_kfold, summarise_splits
from src.models import (
    ContrastiveLoss,
    HeritabilityAuxLoss,
    SiameseConfig,
    build_siamese_model,
)
from src.utils import TwinBrainDataset, get_device, set_seed, twin_collate

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception:  # pragma: no cover
    _TB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All knobs exposed to the CLI and the sweep driver."""

    # Data
    data_root: Path = Path("data/synthetic_h060")
    include_zygosities: tuple[str, ...] = ("MZ", "DZ")

    # Model
    in_channels: int = 100
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    projection_dim: int = 32
    projection_hidden: int = 64
    use_edge_weight: bool = True
    pooling: str = "mean+add"
    normalize_embeddings: bool = True

    # Loss
    contrastive_margin: float = 1.0
    contrastive_metric: str = "cosine"

    # Heritability auxiliary loss (synthetic data only; set h^2_target to enable)
    heritability_aux_weight: float = 0.0
    heritability_aux_target: Optional[float] = None

    # Multimodal PRS fusion (synthetic generator must have been run with prs_dim > 0)
    prs_dim: int = 0
    prs_hidden: int = 64
    prs_embed_dim: int = 64
    prs_dropout: float = 0.1
    prs_fusion: str = "concat"
    genetics_mlp_num_hidden_blocks: int = 2
    # auto | graph | multimodal | genetics_only | fused (see SiameseConfig.model_type)
    model_type: str = "auto"

    # KCL P65: SLIC supervoxel node layout + cross-modal attention (optional)
    modality_feature_dims: Optional[tuple[int, ...]] = None
    use_cross_modal_attention: bool = False
    skip_graph_conv: bool = False
    cross_modal_d_model: int = 64
    cross_modal_num_heads: int = 4
    cross_modal_dropout: float = 0.1
    modality_names: tuple[str, ...] = ()

    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 60
    lr_min_ratio: float = 0.01  # eta_min = lr * lr_min_ratio
    grad_clip: float = 1.0

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # CV
    n_splits: int = 5

    # I/O
    output_dir: Path = Path("runs/exp")
    tensorboard: bool = True
    save_checkpoints: bool = True

    # Reproducibility
    seed: int = 42
    # torch device: "auto" | "mps" | "cuda" | "cpu" (see ``get_device``)
    device_preference: str = "auto"

    def to_siamese(self) -> SiameseConfig:
        return SiameseConfig(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_edge_weight=self.use_edge_weight,
            pooling=self.pooling,  # type: ignore[arg-type]
            projection_dim=self.projection_dim,
            projection_hidden=self.projection_hidden,
            normalize_embeddings=self.normalize_embeddings,
            modality_feature_dims=self.modality_feature_dims,
            use_cross_modal_attention=self.use_cross_modal_attention,
            skip_graph_conv=self.skip_graph_conv,
            cross_modal_d_model=self.cross_modal_d_model,
            cross_modal_num_heads=self.cross_modal_num_heads,
            cross_modal_dropout=self.cross_modal_dropout,
            modality_names=self.modality_names,
            prs_dim=self.prs_dim,
            prs_hidden=self.prs_hidden,
            prs_embed_dim=self.prs_embed_dim,
            prs_dropout=self.prs_dropout,
            prs_fusion=self.prs_fusion,  # type: ignore[arg-type]
            genetics_mlp_num_hidden_blocks=self.genetics_mlp_num_hidden_blocks,
            model_type=self.model_type,  # type: ignore[arg-type]
        )


@dataclass
class FoldResult:
    fold: int
    best_val_loss: float
    best_epoch: int
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    val_metrics_history: list[dict] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)
    heritability: Optional[dict] = None
    checkpoint_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def step(self, value: float, epoch: int) -> bool:
        """Return True if this epoch is a new best."""
        improved = value < self.best - self.min_delta
        if improved:
            self.best = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# ---------------------------------------------------------------------------
# Epoch runners
# ---------------------------------------------------------------------------


def _run_train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: ContrastiveLoss,
    optim: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    aux_loss_fn: Optional[HeritabilityAuxLoss] = None,
    aux_weight: float = 0.0,
) -> dict[str, float]:
    """Run one training epoch. Returns a dict of mean losses for logging."""
    model.train()
    total_contrastive = 0.0
    total_aux = 0.0
    total_combined = 0.0
    n_batches = 0
    n_aux_valid = 0
    for batch in loader:
        batch = batch.to(device)
        # Compute encoder embeddings once; both the contrastive (via projector)
        # and the aux head (directly on encoder output) can share them.
        z_a_enc = model.encode(batch.data_a)
        z_b_enc = model.encode(batch.data_b)
        z_a_proj = model.project(z_a_enc)
        z_b_proj = model.project(z_b_enc)

        contrastive = loss_fn(z_a_proj, z_b_proj, batch.label)

        aux = z_a_enc.new_zeros(())
        aux_is_valid = False
        if aux_loss_fn is not None and aux_weight > 0:
            aux, aux_is_valid = aux_loss_fn(z_a_enc, z_b_enc, batch.zygosities)
        combined = contrastive + aux_weight * aux

        optim.zero_grad()
        combined.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        total_contrastive += float(contrastive.detach().cpu())
        total_aux += float(aux.detach().cpu()) if aux_is_valid else 0.0
        total_combined += float(combined.detach().cpu())
        n_batches += 1
        if aux_is_valid:
            n_aux_valid += 1
    denom = max(n_batches, 1)
    aux_denom = max(n_aux_valid, 1)
    return {
        "loss": total_combined / denom,
        "loss_contrastive": total_contrastive / denom,
        "loss_aux": total_aux / aux_denom,
        "n_aux_valid": n_aux_valid,
        "n_batches": n_batches,
    }


@torch.no_grad()
def _run_eval_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: ContrastiveLoss,
    device: torch.device,
) -> tuple[float, dict, torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    total = 0.0
    n_batches = 0
    all_z_a = []
    all_z_b = []
    all_zyg: list[str] = []
    for batch in loader:
        batch = batch.to(device)
        # Use encoder outputs (not projection) for metric reporting - these are
        # what we'd keep for downstream analysis per SimCLR convention.
        z_a_enc = model.encode(batch.data_a)
        z_b_enc = model.encode(batch.data_b)
        # Training loss uses projector outputs to stay consistent with train.
        z_a_p = model.project(z_a_enc)
        z_b_p = model.project(z_b_enc)
        loss = loss_fn(z_a_p, z_b_p, batch.label)
        total += float(loss.detach().cpu())
        n_batches += 1
        all_z_a.append(z_a_enc.detach().cpu())
        all_z_b.append(z_b_enc.detach().cpu())
        all_zyg.extend(batch.zygosities)
    z_a_cat = torch.cat(all_z_a, dim=0) if all_z_a else torch.empty(0)
    z_b_cat = torch.cat(all_z_b, dim=0) if all_z_b else torch.empty(0)
    metrics = twin_separation_metrics(z_a_cat, z_b_cat, all_zyg, distance_metric="cosine")
    return total / max(n_batches, 1), metrics, z_a_cat, z_b_cat, all_zyg


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------


def train_single_fold(
    fold: int,
    train_dataset,
    val_dataset,
    cfg: TrainConfig,
    device: torch.device,
) -> FoldResult:
    fold_dir = cfg.output_dir / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if cfg.tensorboard and _TB_AVAILABLE:
        writer = SummaryWriter(log_dir=str(fold_dir / "tb"))

    # drop_last=True keeps BatchNorm stable when the last minibatch has size 1.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=twin_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=twin_collate,
    )

    model = build_siamese_model(cfg.to_siamese()).to(device)
    loss_fn = ContrastiveLoss(
        margin=cfg.contrastive_margin, metric=cfg.contrastive_metric  # type: ignore[arg-type]
    )

    aux_loss_fn: Optional[HeritabilityAuxLoss] = None
    if cfg.heritability_aux_weight > 0 and cfg.heritability_aux_target is not None:
        aux_loss_fn = HeritabilityAuxLoss(
            target_h2=cfg.heritability_aux_target
        ).to(device)
        logger.info(
            "[fold %d] Aux h^2 loss enabled (target=%.3f, weight=%.3f)",
            fold, cfg.heritability_aux_target, cfg.heritability_aux_weight,
        )

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=cfg.max_epochs, eta_min=cfg.lr * cfg.lr_min_ratio
    )

    early = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    result = FoldResult(fold=fold, best_val_loss=float("inf"), best_epoch=-1)

    best_state = None
    for epoch in range(cfg.max_epochs):
        t0 = time.perf_counter()
        train_stats = _run_train_epoch(
            model, train_loader, loss_fn, optim, device, cfg.grad_clip,
            aux_loss_fn=aux_loss_fn,
            aux_weight=cfg.heritability_aux_weight,
        )
        train_loss = train_stats["loss"]
        val_loss, val_metrics, z_a_val, z_b_val, zyg_val = _run_eval_epoch(
            model, val_loader, loss_fn, device
        )

        # Heritability estimate on the validation fold (encoder space).
        sims = pair_similarities_from_embeddings(
            z_a_val, z_b_val, zyg_val, metric="cosine"
        )
        h2_est = falconer_h2(
            sims.get("MZ", torch.empty(0)),
            sims.get("DZ", torch.empty(0)),
            bootstrap=0,
            method="cosine_encoder",
        )

        sched.step()
        result.train_loss_history.append(train_loss)
        result.val_loss_history.append(val_loss)
        result.val_metrics_history.append({**val_metrics, "h2": h2_est.h2})

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/train_contrastive",
                              train_stats["loss_contrastive"], epoch)
            if aux_loss_fn is not None:
                writer.add_scalar("loss/train_aux_h2",
                                  train_stats["loss_aux"], epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("lr", sched.get_last_lr()[0], epoch)
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v) and not k.startswith("n_"):
                    writer.add_scalar(f"val/{k}", v, epoch)
            if not np.isnan(h2_est.h2):
                writer.add_scalar("val/h2", h2_est.h2, epoch)

        improved = early.step(val_loss, epoch)
        if improved and cfg.save_checkpoints:
            best_state = {
                "model_state_dict": {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                },
                "epoch": epoch,
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "h2": h2_est.as_dict(),
            }

        dt = time.perf_counter() - t0
        u_extra = (
            f" UNREL={val_metrics['mean_unrel_distance']:.3f}"
            if int(val_metrics.get("n_unrel", 0)) > 0
            and not np.isnan(val_metrics.get("mean_unrel_distance", float("nan")))
            else ""
        )
        logger.info(
            "[fold %d] epoch %03d | train=%.4f val=%.4f | MZ=%.3f DZ=%.3f%s gap=%.3f "
            "AUC=%.3f h2=%.3f lr=%.2e | %.1fs%s",
            fold, epoch, train_loss, val_loss,
            val_metrics["mean_mz_distance"],
            val_metrics["mean_dz_distance"],
            u_extra,
            val_metrics["distance_gap"],
            val_metrics["auc"],
            h2_est.h2,
            sched.get_last_lr()[0],
            dt,
            " *" if improved else "",
        )

        if early.should_stop:
            logger.info(
                "[fold %d] early stopping at epoch %d (best=%d, val_loss=%.4f)",
                fold, epoch, early.best_epoch, early.best,
            )
            break

    result.best_val_loss = early.best
    result.best_epoch = early.best_epoch

    # Reload best state and compute final metrics + bootstrapped h^2 CI.
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    _, val_metrics, z_a_val, z_b_val, zyg_val = _run_eval_epoch(
        model, val_loader, loss_fn, device
    )
    sims = pair_similarities_from_embeddings(z_a_val, z_b_val, zyg_val, metric="cosine")
    h2_final = falconer_h2(
        sims.get("MZ", torch.empty(0)),
        sims.get("DZ", torch.empty(0)),
        bootstrap=1000,
        rng_seed=cfg.seed + fold,
        method="cosine_encoder",
    )
    result.final_metrics = val_metrics
    result.heritability = h2_final.as_dict()

    if cfg.save_checkpoints and best_state is not None:
        ckpt_path = fold_dir / "best.pt"
        torch.save(best_state, ckpt_path)
        result.checkpoint_path = ckpt_path

    (fold_dir / "fold_result.json").write_text(
        json.dumps(
            {
                "fold": result.fold,
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "final_metrics": result.final_metrics,
                "heritability": result.heritability,
                "train_loss_history": result.train_loss_history,
                "val_loss_history": result.val_loss_history,
            },
            indent=2,
            default=str,
        )
    )

    if writer is not None:
        writer.close()

    return result


# ---------------------------------------------------------------------------
# Cross-validation driver
# ---------------------------------------------------------------------------


def run_cross_validation(cfg: TrainConfig) -> dict:
    """Run family-stratified K-fold CV and return an aggregated report."""
    set_seed(cfg.seed)
    device = get_device(cfg.device_preference)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, default=str)
    )

    base_dataset = TwinBrainDataset(
        cfg.data_root,
        include_zygosities=set(cfg.include_zygosities),
        preload=True,
    )
    pairs_df = pd.DataFrame(base_dataset.pairs)
    logger.info(
        "Loaded %d pairs (%s)",
        len(pairs_df),
        pairs_df["zygosity"].value_counts().to_dict(),
    )

    splits = list(
        family_stratified_kfold(
            pairs_df, n_splits=cfg.n_splits, shuffle=True, seed=cfg.seed
        )
    )
    splits_summary = summarise_splits(pairs_df, splits)
    splits_summary.to_csv(cfg.output_dir / "splits_summary.csv", index=False)
    logger.info("CV split summary:\n%s", splits_summary.to_string(index=False))

    fold_results: list[FoldResult] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        logger.info(
            "======== Fold %d/%d (n_train=%d n_val=%d) ========",
            fold_idx + 1, len(splits), len(tr_idx), len(va_idx),
        )
        train_ds = Subset(base_dataset, tr_idx.tolist())
        val_ds = Subset(base_dataset, va_idx.tolist())
        result = train_single_fold(fold_idx, train_ds, val_ds, cfg, device)
        fold_results.append(result)

    # Cross-fold aggregation
    h2_list = [
        fr.heritability["h2"] for fr in fold_results
        if fr.heritability is not None and not np.isnan(fr.heritability["h2"])
    ]
    gap_list = [
        fr.final_metrics.get("distance_gap", float("nan")) for fr in fold_results
    ]
    auc_list = [fr.final_metrics.get("auc", float("nan")) for fr in fold_results]
    acc_list = [
        fr.final_metrics.get("pair_accuracy", float("nan")) for fr in fold_results
    ]

    summary = {
        "n_folds": len(fold_results),
        "mean_h2": float(np.mean(h2_list)) if h2_list else float("nan"),
        "std_h2": float(np.std(h2_list)) if h2_list else float("nan"),
        "mean_distance_gap": float(np.nanmean(gap_list)) if gap_list else float("nan"),
        "mean_auc": float(np.nanmean(auc_list)) if auc_list else float("nan"),
        "mean_pair_accuracy": float(np.nanmean(acc_list)) if acc_list else float("nan"),
        "per_fold": [
            {
                "fold": fr.fold,
                "best_epoch": fr.best_epoch,
                "best_val_loss": fr.best_val_loss,
                "h2": fr.heritability["h2"] if fr.heritability else float("nan"),
                "distance_gap": fr.final_metrics.get("distance_gap", float("nan")),
                "auc": fr.final_metrics.get("auc", float("nan")),
                "pair_accuracy": fr.final_metrics.get("pair_accuracy", float("nan")),
            }
            for fr in fold_results
        ],
    }
    (cfg.output_dir / "cv_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    logger.info(
        "===== CV summary =====  h2 = %.3f +/- %.3f  |  AUC = %.3f  |  acc = %.3f  |  gap = %.3f",
        summary["mean_h2"], summary["std_h2"],
        summary["mean_auc"],
        summary["mean_pair_accuracy"],
        summary["mean_distance_gap"],
    )
    return summary
