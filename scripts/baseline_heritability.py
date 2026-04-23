#!/usr/bin/env python
"""Classical (Falconer) narrow-sense heritability on raw connectivity (KCL P65 baseline).

For each **node** (or optional **edge** in the upper triangle), we define a scalar
phenotype per subject from the Fisher-``z`` connectome, then compute twin–twin
Pearson correlations for MZ and DZ:

    h^2 = 2 ( r_MZ − r_DZ )

with ``r`` the across-pair Pearson ``r`` for that phenotype. Values are clamped to
``[0, 1]`` and written as ``baseline_falconer_atlas.nii.gz`` via
:func:`map_nodes_to_volume`. Task 2 is numpy/scipy-friendly (no MPS required).

Optionally compare to GNN per-node **saliency** (1D vector, same order as
supervoxels) with a simple scatter plot.

Output layout
-------------

Default: ``--output-dir`` or ``<run_dir>/baseline_heritability/`` (fits under
``runs/`` next to your training run).

    baseline_falconer_atlas.nii.gz
    baseline_falconer_h2.npy
    gnn_saliency_vs_falconer_h2.png  (if --saliency-npy is set)
    baseline_falconer_edges.npy  (if --export-edges)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.heritability import per_feature_falconer_h2
from src.utils.visualization import map_nodes_to_volume

PhenotypeMode = Literal["row_mean", "row_sum", "frob_row"]


def _connectivity_from_pt(path: Path) -> np.ndarray:
    d = torch.load(path, map_location="cpu", weights_only=False)
    c = getattr(d, "connectivity", None)
    if c is None:
        raise ValueError(
            f"{path} has no 'connectivity' tensor. "
            "Re-preprocess with graph.build that stores the dense matrix."
        )
    return np.asarray(c, dtype=np.float32)


def _phenotype_vector(c: np.ndarray, mode: PhenotypeMode) -> np.ndarray:
    n = c.shape[0]
    c = c.astype(np.float64, copy=True)
    np.fill_diagonal(c, 0.0)
    if mode == "row_mean":
        return (c.sum(axis=1) / max(n - 1, 1)).astype(np.float32)
    if mode == "row_sum":
        return c.sum(axis=1).astype(np.float32)
    if mode == "frob_row":
        return np.sqrt((c**2).sum(axis=1)).astype(np.float32)
    raise ValueError(f"Unknown phenotype mode: {mode!r}")


def _vectorize_upper(c: np.ndarray) -> np.ndarray:
    c = c.astype(np.float64, copy=False)
    n = c.shape[0]
    iu = np.triu_indices(n, k=1)
    return c[iu[0], iu[1]].astype(np.float32)


def _stacks(
    pair_rows: Sequence[dict],
    conns: dict[str, np.ndarray],
    mode: PhenotypeMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mz: list[Tuple[str, str]] = []
    dz: list[Tuple[str, str]] = []
    for r in pair_rows:
        a, b = str(r["subject_a"]), str(r["subject_b"])
        zy = str(r["zygosity"])
        if zy == "MZ":
            mz.append((a, b))
        elif zy == "DZ":
            dz.append((a, b))
    if len(mz) < 2 or len(dz) < 2:
        raise ValueError(
            f"Need at least 2 MZ and 2 DZ pairs for stable Falconer; got MZ={len(mz)} DZ={len(dz)}"
        )
    a_mz = np.stack([_phenotype_vector(conns[sa], mode) for sa, _ in mz], axis=0)
    b_mz = np.stack([_phenotype_vector(conns[sb], mode) for _, sb in mz], axis=0)
    a_dz = np.stack([_phenotype_vector(conns[sa], mode) for sa, _ in dz], axis=0)
    b_dz = np.stack([_phenotype_vector(conns[sb], mode) for _, sb in dz], axis=0)
    if a_mz.shape[1] != b_mz.shape[1]:
        raise ValueError("Inconsistent number of nodes across MZ stack.")
    return a_mz, b_mz, a_dz, b_dz


def _stacks_edge(
    pair_rows: Sequence[dict],
    conns: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mz, dz = [], []
    for r in pair_rows:
        a, b = str(r["subject_a"]), str(r["subject_b"])
        zy = str(r["zygosity"])
        if zy == "MZ":
            mz.append((a, b))
        elif zy == "DZ":
            dz.append((a, b))
    if len(mz) < 2 or len(dz) < 2:
        raise ValueError(
            f"Need at least 2 MZ and 2 DZ pairs; got MZ={len(mz)} DZ={len(dz)}"
        )
    a_mz = np.stack([_vectorize_upper(conns[sa]) for sa, _ in mz], axis=0)
    b_mz = np.stack([_vectorize_upper(conns[sb]) for _, sb in mz], axis=0)
    a_dz = np.stack([_vectorize_upper(conns[sa]) for sa, _ in dz], axis=0)
    b_dz = np.stack([_vectorize_upper(conns[sb]) for _, sb in dz], axis=0)
    return a_mz, b_mz, a_dz, b_dz


def _saliency_scatter_png(
    h2: np.ndarray,
    sal: np.ndarray,
    out: Path,
    *,
    title: str = "GNN saliency vs. Falconer h² (per node)",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h2 = np.asarray(h2, dtype=np.float64).ravel()
    sal = np.asarray(sal, dtype=np.float64).ravel()
    m = min(h2.size, sal.size)
    h2, sal = h2[:m], sal[:m]
    ok = np.isfinite(h2) & np.isfinite(sal)
    h2, sal = h2[ok], sal[ok]
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ax.scatter(h2, sal, s=8, alpha=0.5, c="tab:blue", edgecolors="none")
    ax.set_xlabel("Falconer h² (classical, per node)")
    ax.set_ylabel("GNN saliency (|∂L/∂x| or saved vector)")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    if np.any(ok):
        r = float(np.corrcoef(h2, sal)[0, 1]) if h2.size > 1 else float("nan")
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\nn = {h2.size}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    out.parent.mkdir(parents=True, exist=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--cohort-root",
        type=Path,
        required=True,
        help="Cohort with subjects/*.pt and pairs.csv (synthetic or preprocessed).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="If set, default --output-dir is <run_dir>/baseline_heritability",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write NIfTI, npy, and plots. Default: <run-dir>/... or "
        "runs/baseline_heritability under cwd.",
    )
    p.add_argument(
        "--slic-labels",
        type=Path,
        required=True,
        help="Reference SLIC label NIfTI (node order = sorted label IDs 1..K).",
    )
    p.add_argument(
        "--pairs-csv",
        type=str,
        default="pairs.csv",
        help="Relative to --cohort-root if not absolute.",
    )
    p.add_argument(
        "--phenotype",
        type=str,
        default="row_mean",
        choices=["row_mean", "row_sum", "frob_row"],
        help="Per-node summary of the connectome before twin correlations.",
    )
    p.add_argument(
        "--saliency-npy",
        type=Path,
        default=None,
        help="1D float32 per-node GNN saliency (length = number of supervoxels).",
    )
    p.add_argument(
        "--export-edges",
        action="store_true",
        help="Also save per-upper-triangle-edge Falconer h² (large .npy).",
    )
    p.add_argument(
        "--include-zygosities",
        type=str,
        default="MZ,DZ",
        help="Comma-separated; must include MZ and DZ for Falconer.",
    )
    args = p.parse_args()
    out = args.output_dir
    if out is None:
        if args.run_dir is not None:
            out = args.run_dir / "baseline_heritability"
        else:
            out = Path("runs") / "baseline_heritability"
    out = Path(out).resolve()
    out.mkdir(parents=True, exist=True)

    cohort = Path(args.cohort_root).expanduser().resolve()
    pairs_path = Path(args.pairs_csv) if Path(args.pairs_csv).is_absolute() else cohort / args.pairs_csv
    if not pairs_path.is_file():
        print(f"Missing {pairs_path}", file=sys.stderr)
        return 1
    zyg = {x.strip() for x in args.include_zygosities.split(",") if x.strip()}
    df = pd.read_csv(pairs_path)
    df = df[df["zygosity"].isin(zyg)].reset_index(drop=True)
    if len(df) == 0:
        print("No pairs after zygosity filter.", file=sys.stderr)
        return 1

    subjects_dir = cohort / "subjects"
    conns: dict[str, np.ndarray] = {}
    for sid in pd.unique(
        np.concatenate(
            [df["subject_a"].astype(str), df["subject_b"].astype(str)]
        )
    ):
        pth = subjects_dir / f"{sid}.pt"
        if not pth.is_file():
            print(f"Missing subject: {pth}", file=sys.stderr)
            return 1
        conns[str(sid)] = _connectivity_from_pt(pth)

    n0 = int(next(iter(conns.values())).shape[0])
    for sid, c in conns.items():
        if c.shape != (n0, n0):
            print(f"Shape mismatch for {sid}: {c.shape}", file=sys.stderr)
            return 1

    pair_rows = df.to_dict(orient="records")
    mode = args.phenotype  # type: ignore[assignment]
    a_mz, b_mz, a_dz, b_dz = _stacks(pair_rows, conns, mode)
    h2_node = per_feature_falconer_h2(a_mz, b_mz, a_dz, b_dz)
    np.save(out / "baseline_falconer_h2.npy", h2_node)
    nii = out / "baseline_falconer_atlas.nii.gz"
    map_nodes_to_volume(
        np.asarray(h2_node, dtype=np.float32),
        args.slic_labels,
        nii,
        fill_background=0.0,
        dtype=np.float32,
    )
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "cohort_root": str(cohort),
                "phenotype": args.phenotype,
                "n_nodes": int(h2_node.size),
                "n_mz_pairs": int(a_mz.shape[0]),
                "n_dz_pairs": int(a_dz.shape[0]),
                "nii": str(nii),
            },
            indent=2,
        )
    )
    if args.saliency_npy is not None and Path(args.saliency_npy).is_file():
        sal = np.load(args.saliency_npy, allow_pickle=False)
        png = out / "gnn_saliency_vs_falconer_h2.png"
        _saliency_scatter_png(
            h2_node,
            np.asarray(sal, dtype=np.float32),
            png,
        )
        print("Wrote", png)
    elif args.saliency_npy is not None:
        print(f"Skip scatter: {args.saliency_npy} not found", file=sys.stderr)

    if args.export_edges:
        e_mz_a, e_mz_b, e_dz_a, e_dz_b = _stacks_edge(pair_rows, conns)
        h2_edge = per_feature_falconer_h2(e_mz_a, e_mz_b, e_dz_a, e_dz_b)
        np.save(out / "baseline_falconer_edges.npy", h2_edge)
        print("Wrote", out / "baseline_falconer_edges.npy", f"len={h2_edge.size}")
    print("Wrote", nii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
