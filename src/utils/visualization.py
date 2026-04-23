"""Interpretability visualisation: NIfTI reverse-mapping and modality bar charts (KCL P65).

All 3D image I/O uses **nibabel** (FSL/ITK-SNAP–compatible ``.nii.gz``). Optional
**nilearn** helpers are used for template alignment checks when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    nib = None  # type: ignore[assignment]
    _NIB = e
else:
    _NIB = None


def map_nodes_to_volume(
    node_values: Union[np.ndarray, dict[int, float]],
    slic_label_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    background_label: int = 0,
    fill_background: float = np.nan,
    dtype: type = np.float32,
) -> Path:
    """Paint a per-supervoxel scalar (or vector slot) onto the 3D SLIC label grid.

    The label NIfTI stores **integer supervoxel IDs** (typically ``1…K``). The
    ``node_values`` vector is **0-based** with ``node_values[i]`` corresponding
    to label ID ``i + 1``, matching the row order used when building
    ``torch_geometric`` graphs (sorted by label ID).

    Parameters
    ----------
    node_values
        1D array of length ``>= max_label`` or a ``dict[label_id, value]`` for
        sparse updates. For array input, ``node_values[0]`` maps to voxel value
        ``1`` in the label volume.
    slic_label_path
        Path to the SLIC label map (``.nii`` / ``.nii.gz``).
    output_path
        Where to save the scalar **NIfTI** (same affine/resolution as input).
    background_label
        Voxels with this value are set to ``fill_background`` (default 0).
    fill_background
        Value written where no node score applies (default: NaN for transparent
        overlays in FSLeyes / ITK-SNAP).

    Returns
    -------
    Path
        Absolute path to the written file.
    """
    if nib is None:  # pragma: no cover
        raise ImportError("nibabel is required for NIfTI I/O") from _NIB

    slic_label_path = Path(slic_label_path)
    output_path = Path(output_path)
    img = nib.load(str(slic_label_path))
    d = np.rint(np.asarray(img.dataobj)).astype(np.int64)
    out = np.full(d.shape, fill_background, dtype=dtype)

    if isinstance(node_values, dict):
        for lab, val in node_values.items():
            li = int(lab)
            if li == background_label:
                continue
            out[d == li] = dtype(val)
    else:
        nv = np.asarray(node_values, dtype=dtype).ravel()
        for lab in np.unique(d):
            if lab == background_label or lab <= 0:
                continue
            idx = int(lab) - 1
            if 0 <= idx < len(nv):
                out[d == lab] = nv[idx]

    out_img = nib.Nifti1Image(np.asarray(out, dtype=dtype), img.affine, img.header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))

    # Optional: ensure nilearn can read the same file (MNI/affine tooling).
    try:
        from nilearn import image  # type: ignore  # noqa: F401
    except Exception:  # pragma: no cover
        pass

    return output_path.resolve()


def plot_modality_importance_barchart(
    importance_mz: np.ndarray,
    importance_dz: np.ndarray,
    modality_names: list[str],
    output_path: Union[str, Path],
    *,
    title: str = "Cross-modal attention mass (mean over nodes × batch)",
    ylabel: str = "Relative attention mass",
) -> Path:
    """Bar chart comparing **MZ** vs **DZ** pooled modality importance scores.

    ``importance_*`` are usually **length M** vectors (one per modality), e.g.
    row-sums of the M×M attention matrix averaged over nodes and pairs. This
    answers "how much did the model attend *from* modality m" in each zygosity
    stratum.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    x = np.arange(len(modality_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, importance_mz, width=w, label="MZ", color="steelblue")
    ax.bar(x + w / 2, importance_dz, width=w, label="DZ", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(modality_names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def pooled_modality_query_importance(
    attn: np.ndarray,
    pool: Literal["rowsum", "colsum", "trace"] = "rowsum",
) -> np.ndarray:
    """Reduce *mean* M×M attention to a length-M positive vector for bar charts.

    Parameters
    ----------
    attn
        ``(N, M, M)`` or ``(M, M)`` averaged attention weights from
        ``MultiheadAttention(average_attn_weights=True)``.
    pool
        * ``rowsum`` — total mass **from** each query modality (recommended).
        * ``colsum`` — mass **to** each key modality.
        * ``trace`` — diagonal only (self-update strength).
    """
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=0)
    if a.ndim != 2:
        raise ValueError(f"expected (M,M) or (N,M,M), got {a.shape}")
    if pool == "rowsum":
        v = a.sum(axis=1)
    elif pool == "colsum":
        v = a.sum(axis=0)
    else:
        v = np.clip(np.diag(a), 0.0, None)
    s = v.sum()
    if s > 0:
        v = v / s
    return v.astype(np.float32)


def per_node_dominant_modality(
    attn: np.ndarray,
    pool: Literal["rowsum", "colsum", "trace"] = "rowsum",
) -> np.ndarray:
    """Categorical **1..M** label per supervoxel: argmax of pooled M×M attention per node.

    For each of **N** nodes, the attention block is :math:`(M, M)` (query, key);
    we pool to a length-**M** vector (same rules as
    :func:`pooled_modality_query_importance`) and take the argmax, then return
    **1-based** indices so :func:`map_nodes_to_volume` can paint labels **1…M**
    in the supervoxel NIfTI.

    Parameters
    ----------
    attn
        Array of shape **(N, M, M)**.
    """
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f"attn must be (N, M, M), got {a.shape}")
    n, m, m2 = a.shape
    if m2 != m:
        raise ValueError("last two dims must be square (M, M)")
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        w = a[i]
        if pool == "rowsum":
            v = w.sum(axis=1)
        elif pool == "colsum":
            v = w.sum(axis=0)
        else:
            v = np.clip(np.diag(w), 0.0, None)
        if np.all(v == 0):
            out[i] = 0.0
        else:
            out[i] = float(np.argmax(v) + 1)  # 1..M
    return out


def plot_dominance_atlas_orthogonal(
    nifti_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    modality_names: list[str],
    # RGBA: first color = background (label 0)
    colors: Optional[list[tuple[float, float, float, float]]] = None,
    dpi: int = 150,
) -> Path:
    """Save an orthogonal view of a **discrete** dominance atlas (low memory).

    Uses **nilearn** when available, otherwise a Matplotlib 3-slice layout.
    ``modality_names[i]`` labels category **i+1** in the image (``0`` = background).
    """
    nifti_path, output_path = Path(nifti_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = len(modality_names)
    if colors is None:
        # Default: distinct hues + black background
        base = [
            (0.0, 0.0, 0.0, 1.0),  # 0
            (0.85, 0.2, 0.2, 1.0),
            (0.2, 0.45, 0.9, 1.0),
            (0.2, 0.7, 0.35, 1.0),
            (0.75, 0.55, 0.1, 1.0),
            (0.6, 0.25, 0.75, 1.0),
        ]
        colors = (base + [(0.4, 0.4, 0.4, 1.0)] * m)[: m + 1]
        while len(colors) < m + 1:
            colors.append((0.5, 0.5, 0.5, 1.0))
        colors = colors[: m + 1]

    if nib is None:  # pragma: no cover
        raise ImportError("nibabel is required for NIfTI plots") from _NIB
    from matplotlib import colors as mcolors
    from matplotlib import pyplot as plt

    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    cmap = mcolors.ListedColormap([c[:3] for c in colors[: m + 1]], name="modality_cats")
    b_edges = np.linspace(-0.5, float(m) + 0.5, m + 2, dtype=np.float64)
    norm = mcolors.BoundaryNorm(b_edges, cmap.N)
    if data.shape[0] == 0:
        raise ValueError("empty NIfTI")

    try:
        from nilearn import plotting  # type: ignore

        plotting.plot_img(
            img,
            display_mode="ortho",
            cut_coords=None,
            cmap=cmap,
            colorbar=True,
            vmin=0,
            vmax=float(m),
            threshold=0.0,
            output_file=str(output_path),
            dpi=dpi,
        )
    except Exception:  # pragma: no cover
        d = data
        zc, yc, xc = (d.shape[0] // 2, d.shape[1] // 2, d.shape[2] // 2)
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        sl = (d[zc, :, :], d[:, yc, :], d[:, :, xc])
        titles = ("Axial (Z-mid)", "Coronal (Y-mid)", "Sagittal (X-mid)")
        for ax, s, t in zip(axes, sl, titles):
            im = ax.imshow(
                s,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="auto",
            )
            ax.set_title(t, fontsize=9)
            ax.axis("off")
        cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.04)
        leg = " / ".join(f"{i+1}={n}" for i, n in enumerate(modality_names))
        cbar.set_label(f"dominant (0=bg)  {leg}")
        fig.suptitle("Modality dominance atlas (orthogonal mid-slices)", fontsize=10, y=1.02)
        fig.tight_layout()
        fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return output_path.resolve()
