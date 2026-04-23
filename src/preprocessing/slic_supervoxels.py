"""SLIC supervoxels for 3D NIfTI volumes (KCL Project 65 node definition).

Replaces *predefined* atlas parcels with *data-adaptive* supervoxels. Segments
are obtained by running 3D SLIC (Achanta et al., as implemented in scikit-image)
on a multi-parametric 3D image stack (e.g. registered FA, MD, T1 morphometry,
and/or T2-FLAIR lesion map). Each **connected supervoxel** becomes a graph node;
node features are **aggregate statistics** (mean, std, etc.) of each modality
**within** the label, plus optional spatial or shape descriptors.

**Typical P65 feature layout (example)** — you must match ``modality_feature_dims``
in :class:`src.models.siamese_gnn.SiameseConfig` to your engineered vectors::

    (FA scalars) + (MD scalars) + (T1 morphology) + (FLAIR lesion) = in_channels

For instance: ``(1, 1, 8, 2)`` for scalar FA, scalar MD, 8 T1 features, 2 FLAIR
features, stacked along the *last* axis of ``Data.x`` per supervoxel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from nibabel.affines import apply_affine
except ImportError as e:  # pragma: no cover
    apply_affine = None
    _NIB_ERR = e
else:
    _NIB_ERR = None

try:
    from skimage.segmentation import slic
except ImportError as e:  # pragma: no cover
    slic = None
    _SK_ERR = e
else:
    _SK_ERR = None


@dataclass
class SlicNiftiConfig:
    n_segments: int = 2000
    """Target number of supervoxels (soft target; not exact)."""
    compactness: float = 0.1
    """SLIC trade-off: higher → more box-like, lower → follows intensity edges."""
    sigma: float = 0.0
    """Pre-smoothing sigma (in voxel units) before SLIC; 0 = none."""
    enforce_connectivity: bool = True
    start_label: int = 1
    """Voxel value for first region (0 reserved for background in masks)."""
    multichannel: bool = True
    """If True, last axis of ``image`` is treated as feature channels."""


def run_slic_on_volume(
    image: np.ndarray,
    config: SlicNiftiConfig = SlicNiftiConfig(),
    *,
    channel_axis: int = -1,
) -> np.ndarray:
    """Run 3D SLIC on a numpy array.

    Parameters
    ----------
    image
        4D array ``(I, J, K, C)`` (multi-parametric) with ``channel_axis=-1``,
        or 3D ``(I, J, K)`` single-channel. Intensities should be in comparable
        scale per channel (e.g. z-score **within brain mask** before stacking).
    config
        Hyperparameters. ``compactness`` in multi-channel 3D often needs
        hand-tuning (typical *starting* range **0.01–0.2**; higher for smoother blobs).
    channel_axis
        Axis of feature channels. Default ``-1`` matches
        stack_modalities_4d.
    """
    if _SK_ERR is not None or slic is None:  # pragma: no cover
        raise ImportError("scikit-image is required. pip install scikit-image") from _SK_ERR

    img = image
    is_multichannel = img.ndim == 4
    if is_multichannel != config.multichannel:
        raise ValueError("image ndim and SlicNiftiConfig.multichannel are inconsistent")
    if is_multichannel and channel_axis not in (-1, img.ndim - 1):
        img = np.moveaxis(img, channel_axis, -1)

    kw: dict = dict(
        n_segments=config.n_segments,
        compactness=config.compactness,
        sigma=config.sigma,
        enforce_connectivity=config.enforce_connectivity,
        start_label=config.start_label,
    )
    if is_multichannel:
        # scikit-image >=0.19: channel axis for 4D (Z, Y, X, C) volumes
        kw["channel_axis"] = -1
    return slic(img, **kw)


def stack_modalities_4d(
    volumes: list[np.ndarray],
) -> np.ndarray:
    """Stack *aligned* 3D volumes along a channel axis, shape (Z, Y, X, C)."""
    if not volumes:
        raise ValueError("volumes must be non-empty")
    ref = volumes[0].shape
    for v in volumes:
        if v.shape != ref:
            raise ValueError(f"all volumes must share shape {ref}, got {v.shape}")
    return np.stack(volumes, axis=-1).astype(np.float32, copy=False)


def slic_labels_from_nifti_stack(
    paths: list[str],
    config: SlicNiftiConfig = SlicNiftiConfig(),
) -> np.ndarray:
    """Load several registered NIfTIs, stack channels, and return SLIC label map.

    All paths must point to 3D volumes with the **same** affine and shape
    (co-registered; same voxel grid). Uses nibabel; paths order defines channel
    order (e.g. ``[fa.nii, md.nii, t1.nii, flair.nii]``).
    """
    if _NIB_ERR is not None:  # pragma: no cover
        raise ImportError("nibabel is required for NIfTI I/O") from _NIB_ERR
    import nibabel as nib

    arrs: list[np.ndarray] = []
    for p in paths:
        im = nib.load(p)
        arrs.append(np.asanyarray(im.dataobj))
    vol4 = stack_modalities_4d(arrs)
    return run_slic_on_volume(vol4, config=config, channel_axis=-1)


def label_centroid_mni(
    label_img: "Any",
    seg: np.ndarray,
    label_id: int,
) -> np.ndarray:
    """Physical-space centroid (mm) of voxels with ``seg == label_id`` using affine."""
    if _NIB_ERR is not None:  # pragma: no cover
        raise ImportError("nibabel is required") from _NIB_ERR
    import nibabel as nib

    d = np.asarray(seg)
    mask = d == int(label_id)
    if not np.any(mask):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)
    ii, jj, kk = np.where(mask)
    ijk = np.column_stack([ii, jj, kk]).astype(np.float64)
    if hasattr(label_img, "affine"):
        aff = label_img.affine
    else:
        aff = nib.load(label_img).affine  # type: ignore[assignment]
    return apply_affine(aff, ijk).mean(axis=0).astype(np.float32)
