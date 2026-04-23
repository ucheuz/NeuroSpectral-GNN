#!/usr/bin/env python
"""Heritability-recovery sweep: for a range of ground-truth h^2 values,
generate a synthetic twin cohort (with optional PRS) and train one or more
model variants on it, plotting the recovered h^2 and MZ/DZ AUC curves.

This is the headline figure for the 30k Seed Award proposal and for the
dissertation's methodology-validation chapter. By default we compare:

    * baseline                : graph-only Siamese GNN, no auxiliary loss
    * multimodal+aux          : PRS fusion + heritability aux loss

The multimodal+aux variant demonstrates the two architectural contributions
of Project 65 (PRS fusion + heritability-aware training) while the baseline
provides the reference curve.

Outputs
-------
    {out_dir}/cohorts/h_{ii}/subjects/*.pt
    {out_dir}/runs/{variant}/h_{ii}/cv_summary.json
    {out_dir}/sweep_results.csv
    {out_dir}/h2_recovery.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe for CI / sandboxes / HPC
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.graph import GraphBuildConfig  # noqa: E402
from src.preprocessing.synthetic import (  # noqa: E402
    SyntheticCohortConfig,
    empirical_heritability_from_connectivities,
    generate_cohort,
    save_synthetic_cohort,
)
from src.training import TrainConfig, run_cross_validation  # noqa: E402
from src.utils import set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Model variants definition
# ---------------------------------------------------------------------------

VARIANT_STYLES = {
    "baseline": {
        "display": "GNN baseline (graph-only)",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    "multimodal+aux": {
        "display": "GNN + PRS + h$^2$ aux loss",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-",
    },
}


def _variant_train_config(
    variant: str,
    cohort_dir: Path,
    run_dir: Path,
    args: argparse.Namespace,
    h2_true: float,
) -> TrainConfig:
    zyg: tuple[str, ...] = (
        ("MZ", "DZ", "UNREL") if getattr(args, "n_unrelated", 0) > 0 else ("MZ", "DZ")
    )
    base = dict(
        data_root=cohort_dir,
        output_dir=run_dir,
        include_zygosities=zyg,
        in_channels=args.n_rois,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        projection_dim=args.projection_dim,
        projection_hidden=args.projection_hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        n_splits=args.n_splits,
        patience=args.patience,
        tensorboard=not args.no_tensorboard,
        save_checkpoints=args.save_checkpoints,
        seed=args.seed,
    )
    if variant == "baseline":
        return TrainConfig(**base)
    if variant == "multimodal+aux":
        return TrainConfig(
            **base,
            prs_dim=args.prs_dim,
            prs_hidden=args.prs_hidden,
            prs_embed_dim=args.prs_embed_dim,
            prs_fusion=args.prs_fusion,
            heritability_aux_weight=args.aux_weight,
            heritability_aux_target=h2_true,
        )
    raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Driver pieces
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--h2-values", type=float, nargs="+",
                   default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--variants", nargs="+",
                   default=list(VARIANT_STYLES.keys()),
                   choices=list(VARIANT_STYLES.keys()))
    p.add_argument("--n-mz", type=int, default=40)
    p.add_argument("--n-dz", type=int, default=40)
    p.add_argument(
        "--n-unrelated",
        type=int,
        default=0,
        help="Synthetic UNREL pairs (label=1, max margin); included in training if >0.",
    )
    p.add_argument("--n-rois", type=int, default=64)
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--projection-dim", type=int, default=16)
    p.add_argument("--projection-hidden", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=25)
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    # PRS + aux-loss (used by the multimodal+aux variant)
    p.add_argument("--prs-dim", type=int, default=16)
    p.add_argument("--prs-hidden", type=int, default=32)
    p.add_argument("--prs-embed-dim", type=int, default=32)
    p.add_argument("--prs-fusion", default="concat", choices=["concat", "gated"])
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--save-checkpoints", action="store_true",
                   help="Persist best-epoch checkpoints (used for the latent-space figure).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _generate_and_baseline(
    h2_true: float, cohort_dir: Path, args: argparse.Namespace
) -> float:
    """Generate a multimodal-ready cohort (PRS included so both variants can
    read from the same directory) and compute the raw-matrix Falconer baseline.
    """
    cfg = SyntheticCohortConfig(
        n_mz_pairs=args.n_mz,
        n_dz_pairs=args.n_dz,
        n_unrelated_pairs=getattr(args, "n_unrelated", 0),
        n_rois=args.n_rois,
        heritability=h2_true,
        prs_dim=args.prs_dim,
        seed=args.seed,
        graph_config=GraphBuildConfig(
            sparsify_strategy="proportional",
            keep_top_fraction=0.2,
            node_feature_mode="profile",
        ),
    )
    save_synthetic_cohort(cfg, cohort_dir)
    conns, pairs, _ = generate_cohort(cfg)
    return float(empirical_heritability_from_connectivities(conns, pairs))


def _plot_sweep(df: pd.DataFrame, variants: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), dpi=120)
    axA, axB = axes

    # Truth + Falconer baseline (same across variants -> pick first variant's rows)
    ref = df[df["variant"] == variants[0]].sort_values("h2_true")
    x_ref = ref["h2_true"].to_numpy()

    axA.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5,
             label="Ground truth ($y=x$)")
    axA.plot(
        x_ref, ref["h2_raw_falconer"], "s--", color="#d62728", lw=1.8,
        markersize=6, label="Raw-matrix Falconer baseline",
    )
    axB.axhline(0.5, color="gray", linestyle=":", lw=1.2,
                label="Chance (AUC = 0.5)")

    for variant in variants:
        style = VARIANT_STYLES[variant]
        sub = df[df["variant"] == variant].sort_values("h2_true")
        x = sub["h2_true"].to_numpy()
        axA.errorbar(
            x, sub["h2_gnn_mean"], yerr=sub["h2_gnn_std"],
            fmt=f"{style['marker']}{style['linestyle']}",
            color=style["color"], lw=2.0, capsize=4, markersize=7,
            label=style["display"],
        )
        axB.plot(
            x, sub["auc_mean"],
            f"{style['marker']}{style['linestyle']}",
            color=style["color"], lw=2.2, markersize=7,
            label=style["display"],
        )

    axA.set_xlabel("Ground-truth heritability ($h^2$)", fontsize=12)
    axA.set_ylabel("Estimated $\\hat{h}^2$", fontsize=12)
    axA.set_xlim(-0.05, 1.05)
    axA.set_ylim(-0.15, 1.15)
    axA.set_title("A. Heritability point estimate", fontsize=12)
    axA.grid(alpha=0.3)
    axA.legend(loc="upper left", frameon=True, fontsize=9)

    axB.set_xlabel("Ground-truth heritability ($h^2$)", fontsize=12)
    axB.set_ylabel("Validation AUC", fontsize=12)
    axB.set_xlim(-0.05, 1.05)
    axB.set_ylim(0.3, 1.05)
    axB.set_title("B. MZ vs. DZ discriminability", fontsize=12)
    axB.grid(alpha=0.3)
    axB.legend(loc="lower right", frameon=True, fontsize=9)

    fig.suptitle(
        "Project 65 - Heritability Recovery on Synthetic Twin Cohorts",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("h2_sweep")
    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    rows: list[dict] = []
    for h2_true in args.h2_values:
        label = f"{int(round(h2_true * 100)):03d}"
        cohort_dir = args.output_dir / "cohorts" / f"h_{label}"

        log.info("===== h^2_true = %.2f =====", h2_true)
        h2_raw = _generate_and_baseline(h2_true, cohort_dir, args)
        log.info("Raw-matrix Falconer baseline: %.3f", h2_raw)

        for variant in args.variants:
            run_dir = args.output_dir / "runs" / variant / f"h_{label}"
            log.info("  -- variant: %s", variant)
            cfg = _variant_train_config(variant, cohort_dir, run_dir, args, h2_true)
            summary = run_cross_validation(cfg)
            per_fold = [f["h2"] for f in summary["per_fold"]]
            log.info(
                "    h^2: %.3f +/- %.3f | AUC=%.3f | per-fold=%s",
                summary["mean_h2"], summary["std_h2"], summary["mean_auc"],
                [f"{x:.2f}" for x in per_fold],
            )
            rows.append({
                "h2_true": h2_true,
                "variant": variant,
                "h2_raw_falconer": h2_raw,
                "h2_gnn_mean": summary["mean_h2"],
                "h2_gnn_std": summary["std_h2"],
                "auc_mean": summary["mean_auc"],
                "distance_gap_mean": summary["mean_distance_gap"],
                "per_fold_h2": per_fold,
            })

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved sweep results to %s", csv_path)

    plot_path = args.output_dir / "h2_recovery.png"
    _plot_sweep(df, args.variants, plot_path)
    log.info("Saved figure to %s", plot_path)

    print("\n" + df.drop(columns=["per_fold_h2"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
