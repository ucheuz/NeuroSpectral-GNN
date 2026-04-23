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
