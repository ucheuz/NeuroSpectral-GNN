#!/usr/bin/env python
"""3D connectome / glass-brain style figure (grant / supplementary slide).

Takes a square connectivity matrix (Fisher-z or correlation) and parcellation
coordinates from the Schaefer atlas when ``n_rois`` in {100, 200, 400};
otherwise uses an illustrative 3D layout (not anatomically registered).

**Not** part of the training pipeline—export-only visualization.

Usage
-----
    # Random symmetric matrix matching Schaefer-100
    python scripts/plot_brain_3d.py --n-rois 100 --output figures/connectome_3d.png

    # From a NumPy .npy file
    python scripts/plot_brain_3d.py --matrix path/to/conn.npy --output fig.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_coords_schaefer(n_rois: int) -> np.ndarray | None:
    try:
        from nilearn import datasets
        from nilearn.plotting import find_parcellation_cut_coords
    except ImportError as e:  # pragma: no cover
        raise SystemExit("nilearn is required. pip install nilearn") from e
    if n_rois not in (100, 200, 400):
        return None
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=7, resolution_mm=1, verbose=0
    )
    maps = atlas["maps"]
    # Nilearn >= 0.14: (labels_img, background_label=0). Older: background_img=...
    try:
        return find_parcellation_cut_coords(maps, background_label=0)
    except TypeError:
        return find_parcellation_cut_coords(  # type: ignore[call-arg]
            maps,
            background_img=maps,
            label_separator="-",
        )


def _fallback_coords(n: int, seed: int) -> np.ndarray:
    """Illustrative MNI-like positions on a scale similar to nilearn."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(-50, 50, size=(n, 3)).astype(np.float32))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-rois", type=int, default=100, help="Matrix size (100/200/400 for Schaefer coords).")
    p.add_argument("--matrix", type=Path, default=None, help="N×N .npy connectivity (optional).")
    p.add_argument("--edge-threshold", type=float, default=0.4, help="Fisher-z absolute threshold (if applicable).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.matrix is not None:
        c = np.load(str(args.matrix))
    else:
        rng = np.random.default_rng(args.seed)
        a = rng.standard_normal((args.n_rois, args.n_rois))
        c = (a + a.T) / 2.0
        np.fill_diagonal(c, 0.0)

    n = c.shape[0]
    if c.shape[1] != n:
        raise SystemExit("Matrix must be square")
    np.fill_diagonal(c, 0.0)

    coords = _load_coords_schaefer(n)
    coord_note = "Schaefer MNI (nilearn)"
    if coords is None:
        coords = _fallback_coords(n, args.seed)
        coord_note = f"synthetic layout (N={n} not 100/200/400 — illustrative only)"
    else:
        coords = np.asarray(coords, dtype=np.float32)
        if len(coords) > n:
            coords = coords[:n]
        elif len(coords) < n:
            raise SystemExit(
                f"Atlas has {len(coords)} coords but matrix is {n}×{n}."
            )

    import matplotlib
    matplotlib.use("Agg")
    from nilearn import plotting  # type: ignore

    display = plotting.plot_connectome(
        c,
        coords,
        edge_threshold=args.edge_threshold,
        node_size=20,
        colorbar=True,
        display_mode="z",
        node_color="steelblue",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib import pyplot as plt

    if hasattr(display, "savefig"):
        display.savefig(str(args.output), dpi=150, bbox_inches="tight")
    else:
        fig = getattr(display, "frame", None) or plt.gcf()
        fig.suptitle(f"3D parcellation connectome ({coord_note})", fontsize=10, y=0.98)
        fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"Saved {args.output} — {coord_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
