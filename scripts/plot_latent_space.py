#!/usr/bin/env python
"""Encode every subject with a trained Siamese model and project the learned
embeddings to 2D via UMAP (preferred) or t-SNE (fallback), colouring points by
family so that MZ pairs appear as tight dots, DZ pairs as loose dots, and
unrelated subjects as scattered dust.

This is the Month 3 tracker item ("latent space figure") and one of the
strongest grant-proposal visuals: it lets reviewers *see* that the model
has learned a genetically-informed manifold.

Usage
-----
    python scripts/plot_latent_space.py \
        --run-dir data/h2_sweep/runs/multimodal+aux/h_100 \
        --cohort-dir data/h2_sweep/cohorts/h_100 \
        --fold 1 \
        --output data/h2_sweep/latent_space.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import SiameseConfig, build_siamese_model  # noqa: E402
from src.utils import TwinBrainDataset, get_device, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Per-h^2 run directory containing config.json and fold_XX/best.pt")
    p.add_argument("--cohort-dir", type=Path, required=True,
                   help="Cohort directory containing subjects/*.pt and pairs.csv")
    p.add_argument("--fold", type=int, default=None,
                   help="Which fold checkpoint to load. Defaults to the fold with "
                        "the best validation loss.")
    p.add_argument("--method", default="auto",
                   choices=["auto", "umap", "tsne"],
                   help="Dim-reduction method. 'auto' prefers UMAP if available.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--highlight-n-families", type=int, default=4,
                   help="How many (MZ, DZ) families to highlight with lines "
                        "connecting co-twins. 0 disables.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config + checkpoint loading
# ---------------------------------------------------------------------------


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")
    return json.loads(cfg_path.read_text())


def _select_fold(run_dir: Path, requested: int | None) -> Path:
    cv = json.loads((run_dir / "cv_summary.json").read_text())
    per_fold = cv["per_fold"]
    if requested is None:
        best = min(per_fold, key=lambda f: f["best_val_loss"])
        fold_idx = best["fold"]
    else:
        fold_idx = requested
    fold_dir = run_dir / f"fold_{fold_idx:02d}"
    ckpt = fold_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoint {ckpt}. Was the sweep run with --save-checkpoints?"
        )
    return ckpt


def _build_model_from_config(cfg: dict, device: torch.device) -> torch.nn.Module:
    siamese_cfg = SiameseConfig(
        in_channels=cfg["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        use_edge_weight=cfg["use_edge_weight"],
        pooling=cfg["pooling"],  # type: ignore[arg-type]
        projection_dim=cfg["projection_dim"],
        projection_hidden=cfg["projection_hidden"],
        normalize_embeddings=cfg["normalize_embeddings"],
        prs_dim=cfg.get("prs_dim", 0),
        prs_hidden=cfg.get("prs_hidden", 64),
        prs_embed_dim=cfg.get("prs_embed_dim", 64),
        prs_dropout=cfg.get("prs_dropout", 0.1),
        prs_fusion=cfg.get("prs_fusion", "concat"),
    )
    return build_siamese_model(siamese_cfg).to(device).eval()


# ---------------------------------------------------------------------------
# Encoding the whole cohort
# ---------------------------------------------------------------------------


def _encode_cohort(
    model: torch.nn.Module,
    ds: TwinBrainDataset,
    device: torch.device,
) -> tuple[np.ndarray, list[dict]]:
    """Encode every (unique) subject in the dataset. Returns (embeddings, metadata)."""
    seen: set[str] = set()
    metas: list[dict] = []
    tensors: list[torch.Tensor] = []
    with torch.no_grad():
        for idx in range(len(ds)):
            data_a, data_b, meta = ds[idx]
            for data, subj, twin_id in (
                (data_a, meta["subject_a"], "A"),
                (data_b, meta["subject_b"], "B"),
            ):
                if subj in seen:
                    continue
                seen.add(subj)
                z = model.encode(data.to(device)).squeeze(0)
                tensors.append(z.cpu())
                metas.append({
                    "subject_id": subj,
                    "family_id": meta["family_id"],
                    "twin_id": twin_id,
                    "zygosity": meta["zygosity"],
                })
    z_all = torch.stack(tensors).numpy()
    return z_all, metas


# ---------------------------------------------------------------------------
# Dim reduction
# ---------------------------------------------------------------------------


def _reduce(z: np.ndarray, method: str, seed: int) -> tuple[np.ndarray, str]:
    if method in ("auto", "umap"):
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_neighbors=min(15, z.shape[0] - 1),
                min_dist=0.15,
                metric="euclidean",
                random_state=seed,
            )
            return reducer.fit_transform(z), "UMAP"
        except ImportError:
            if method == "umap":
                raise
    # Fallback to t-SNE (always available via scikit-learn)
    from sklearn.manifold import TSNE
    perplexity = min(30.0, max(5.0, (z.shape[0] - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(z), f"t-SNE (perplexity={perplexity:.0f})"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


ZYG_COLORS = {
    "MZ": "#d62728",      # red - identical
    "DZ": "#1f77b4",      # blue - siblings
    "UNREL": "#7f7f7f",   # gray
}


def _plot_latent(
    coords: np.ndarray,
    metas: list[dict],
    method_name: str,
    highlight_n: int,
    output: Path,
    seed: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)

    # Group family_id -> (idx_A, idx_B)
    family_to_idxs: dict[str, dict[str, int]] = {}
    for i, m in enumerate(metas):
        family_to_idxs.setdefault(m["family_id"], {})[m["twin_id"]] = i

    # ---- Scatter all points, color by zygosity -------------------------
    for zyg, color in ZYG_COLORS.items():
        idxs = [i for i, m in enumerate(metas) if m["zygosity"] == zyg]
        if not idxs:
            continue
        ax.scatter(
            coords[idxs, 0], coords[idxs, 1],
            s=55, c=color, alpha=0.75,
            edgecolors="white", linewidths=0.8,
            label=f"{zyg} (n={len(idxs) // 2} pairs)",
            zorder=3,
        )

    # ---- Connect co-twins with lines to make pair-tightness visible ----
    for fid, sides in family_to_idxs.items():
        if "A" not in sides or "B" not in sides:
            continue
        i_a, i_b = sides["A"], sides["B"]
        zyg = metas[i_a]["zygosity"]
        color = ZYG_COLORS.get(zyg, "#000000")
        ax.plot(
            [coords[i_a, 0], coords[i_b, 0]],
            [coords[i_a, 1], coords[i_b, 1]],
            color=color, alpha=0.35, lw=1.0, zorder=2,
        )

    # ---- Highlight a handful of pairs with family labels ---------------
    if highlight_n > 0:
        rng = np.random.default_rng(seed)
        mz_families = [f for f, s in family_to_idxs.items()
                       if "A" in s and "B" in s
                       and metas[s["A"]]["zygosity"] == "MZ"]
        dz_families = [f for f, s in family_to_idxs.items()
                       if "A" in s and "B" in s
                       and metas[s["A"]]["zygosity"] == "DZ"]
        picks: list[str] = []
        picks += list(rng.choice(mz_families,
                                 size=min(highlight_n, len(mz_families)),
                                 replace=False)) if mz_families else []
        picks += list(rng.choice(dz_families,
                                 size=min(highlight_n, len(dz_families)),
                                 replace=False)) if dz_families else []
        for fid in picks:
            sides = family_to_idxs[fid]
            i_a, i_b = sides["A"], sides["B"]
            cx = (coords[i_a, 0] + coords[i_b, 0]) / 2
            cy = (coords[i_a, 1] + coords[i_b, 1]) / 2
            ax.annotate(
                fid.replace("FAM_", ""),
                xy=(cx, cy),
                fontsize=8, color="black", alpha=0.7,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="gray", alpha=0.85, lw=0.6),
                zorder=4,
            )

    ax.set_xlabel(f"{method_name} dim 1", fontsize=11)
    ax.set_ylabel(f"{method_name} dim 2", fontsize=11)
    ax.set_title(
        "Learned twin-brain manifold\n"
        "MZ pairs collapse into tight dots; DZ pairs sit close but not identical",
        fontsize=12,
    )
    ax.grid(alpha=0.25)

    # Composite legend: zygosity markers + line style explanation
    handles, _ = ax.get_legend_handles_labels()
    handles += [
        mpatches.Patch(color="none", label=""),
        mpatches.Patch(color="gray", alpha=0.35, label="line = co-twin link"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=10)

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("plot_latent_space")
    set_seed(args.seed)
    device = get_device()

    cfg = _load_config(args.run_dir)
    ckpt_path = _select_fold(args.run_dir, args.fold)
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model = _build_model_from_config(cfg, device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Model restored: %s (params=%d)",
             type(model).__name__,
             sum(p.numel() for p in model.parameters()))

    zygosities = tuple(cfg.get("include_zygosities", ["MZ", "DZ"]))
    ds = TwinBrainDataset(
        args.cohort_dir,
        include_zygosities=set(zygosities),
        preload=True,
    )
    log.info("Encoding %d subjects from cohort %s", 2 * len(ds), args.cohort_dir)
    z_all, metas = _encode_cohort(model, ds, device)
    log.info("Embedding matrix: %s", z_all.shape)

    coords, method_name = _reduce(z_all, args.method, args.seed)
    log.info("Reduced to 2D via %s", method_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_latent(
        coords=coords,
        metas=metas,
        method_name=method_name,
        highlight_n=args.highlight_n_families,
        output=args.output,
        seed=args.seed,
    )
    log.info("Saved latent-space figure to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
