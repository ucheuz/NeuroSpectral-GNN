#!/usr/bin/env python
"""Train the Siamese BrainGNN with family-stratified K-fold cross-validation.

Example
-------
    python scripts/train.py \
        --data-root data/synthetic_h060 \
        --output-dir runs/synthetic_h060 \
        --n-splits 5 --max-epochs 60 --batch-size 8 \
        --in-channels 100 --hidden-channels 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training import TrainConfig, run_cross_validation  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Data
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--include-zygosities", nargs="+", default=["MZ", "DZ"])

    # Model
    p.add_argument("--in-channels", type=int, default=100)
    p.add_argument("--hidden-channels", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--projection-dim", type=int, default=32)
    p.add_argument("--projection-hidden", type=int, default=64)
    p.add_argument("--pooling", default="mean+add",
                   choices=["mean", "add", "mean+add"])
    p.add_argument("--no-edge-weight", action="store_true")
    p.add_argument("--no-normalize", action="store_true")

    # Loss
    p.add_argument("--contrastive-margin", type=float, default=1.0)
    p.add_argument("--contrastive-metric", default="cosine",
                   choices=["cosine", "euclidean"])

    # Heritability auxiliary loss (synthetic-only)
    p.add_argument("--heritability-aux-weight", type=float, default=0.0,
                   help="lambda weight for MSE(h^2_batch, h^2_target). 0 disables.")
    p.add_argument("--heritability-aux-target", type=float, default=None,
                   help="Ground-truth h^2 to anchor the auxiliary loss to. "
                        "Only valid on synthetic data.")

    # Multimodal PRS fusion
    p.add_argument("--prs-dim", type=int, default=0,
                   help="Dimensionality of PRS input. 0 disables the modality.")
    p.add_argument("--prs-hidden", type=int, default=64)
    p.add_argument("--prs-embed-dim", type=int, default=64)
    p.add_argument("--prs-dropout", type=float, default=0.1)
    p.add_argument("--prs-fusion", default="concat",
                   choices=["concat", "gated"])
    p.add_argument(
        "--genetics-mlp-num-hidden-blocks",
        type=int,
        default=2,
        help="PRS MLP: number of (Linear+BN+ReLU+Dropout) blocks before the final projection.",
    )
    p.add_argument(
        "--model-type",
        default="auto",
        choices=["auto", "graph", "multimodal", "fused", "genetics_only"],
        help="Encoder: graph-only (GNN), multimodal/fused (GNN+PRS), genetics_only (PRS ablation), "
             "or auto (graph if prs_dim=0 else multimodal).",
    )

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--lr-min-ratio", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Early stopping
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)

    # CV
    p.add_argument("--n-splits", type=int, default=5)

    # I/O
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-checkpoints", action="store_true")

    # Misc
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Torch device preference (see src.utils.device.get_device).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        include_zygosities=tuple(args.include_zygosities),
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        projection_dim=args.projection_dim,
        projection_hidden=args.projection_hidden,
        use_edge_weight=not args.no_edge_weight,
        pooling=args.pooling,
        normalize_embeddings=not args.no_normalize,
        contrastive_margin=args.contrastive_margin,
        contrastive_metric=args.contrastive_metric,
        heritability_aux_weight=args.heritability_aux_weight,
        heritability_aux_target=args.heritability_aux_target,
        prs_dim=args.prs_dim,
        prs_hidden=args.prs_hidden,
        prs_embed_dim=args.prs_embed_dim,
        prs_dropout=args.prs_dropout,
        prs_fusion=args.prs_fusion,
        genetics_mlp_num_hidden_blocks=args.genetics_mlp_num_hidden_blocks,
        model_type=args.model_type,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr_min_ratio=args.lr_min_ratio,
        grad_clip=args.grad_clip,
        patience=args.patience,
        min_delta=args.min_delta,
        n_splits=args.n_splits,
        tensorboard=not args.no_tensorboard,
        save_checkpoints=not args.no_checkpoints,
        seed=args.seed,
        device_preference=args.device,
    )

    summary = run_cross_validation(cfg)
    log = logging.getLogger("train")
    log.info("Final h^2 = %.3f +/- %.3f  |  AUC = %.3f",
             summary["mean_h2"], summary["std_h2"], summary["mean_auc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
