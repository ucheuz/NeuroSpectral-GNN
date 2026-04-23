"""Timeseries extraction + Fisher-z correlation connectivity.

Math primer (for the dissertation methods section):

1. The parcellated timeseries is a matrix T in R^{T x N} where T = number of
   fMRI timepoints and N = number of ROIs (graph nodes).

2. Pearson correlation between ROIs i, j is:
       r_ij = cov(T[:, i], T[:, j]) / (std(T[:, i]) * std(T[:, j]))
   This produces an N x N symmetric matrix with 1s on the diagonal.

3. Because r is bounded in [-1, 1], its sampling distribution is skewed for
   |r| -> 1. Fisher's z-transform,
       z_ij = arctanh(r_ij) = 0.5 * ln((1 + r_ij) / (1 - r_ij)),
   maps it onto R with approximately-normal sampling variance ~ 1/(T - 3).
   This is the canonical stabilizer used in connectomics and plays well with
   downstream losses that assume unbounded, roughly-Gaussian inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from nilearn.maskers import NiftiLabelsMasker

from src.preprocessing.atlas import AtlasBundle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeseriesConfig:
    """Bandpass + nuisance-regression settings for NiftiLabelsMasker.

    Defaults follow the resting-state fMRI community consensus
    (Power et al. 2014; Satterthwaite et al. 2013). Override per-dataset as
    needed.
    """

    standardize: str = "zscore_sample"
    detrend: bool = True
    low_pass: float = 0.10  # Hz (retain slow BOLD fluctuations)
    high_pass: float = 0.01  # Hz (remove scanner drift)
    t_r: Optional[float] = None  # repetition time in seconds; REQUIRED if filtering
    smoothing_fwhm: Optional[float] = None  # mm; None = no spatial smoothing
    memory_level: int = 1


def extract_timeseries(
    nii_path: str | Path,
    atlas: AtlasBundle,
    ts_config: TimeseriesConfig,
    confounds_path: Optional[str | Path] = None,
    cache_dir: Optional[str | Path] = None,
) -> np.ndarray:
    """Parcellate a 4D fMRI NIfTI into an (T x N) timeseries matrix.

    Parameters
    ----------
    nii_path : path
        4D functional NIfTI (.nii or .nii.gz).
    atlas : AtlasBundle
        Schaefer atlas handle from :func:`load_schaefer_atlas`.
    ts_config : TimeseriesConfig
        Filtering / standardization options.
    confounds_path : path, optional
        fmriprep-style confounds TSV (motion, aCompCor, etc.). If None, no
        nuisance regression is performed.
    cache_dir : path, optional
        nilearn joblib cache directory. Speeds up re-runs considerably.

    Returns
    -------
    np.ndarray, shape (T, N), dtype float32
    """
    nii_path = str(Path(nii_path).expanduser())
    if confounds_path is not None:
        confounds_path = str(Path(confounds_path).expanduser())

    if (ts_config.low_pass is not None or ts_config.high_pass is not None) and ts_config.t_r is None:
        logger.warning(
            "Bandpass filtering requested but t_r is None; nilearn will skip "
            "filtering. Provide t_r to enable low_pass/high_pass."
        )

    masker = NiftiLabelsMasker(
        labels_img=atlas.maps_img,
        standardize=ts_config.standardize,
        detrend=ts_config.detrend,
        low_pass=ts_config.low_pass,
        high_pass=ts_config.high_pass,
        t_r=ts_config.t_r,
        smoothing_fwhm=ts_config.smoothing_fwhm,
        memory=str(cache_dir) if cache_dir is not None else None,
        memory_level=ts_config.memory_level,
        verbose=0,
    )

    timeseries = masker.fit_transform(nii_path, confounds=confounds_path)
    # masker returns float64 by default; downcast to save memory on M1.
    timeseries = np.asarray(timeseries, dtype=np.float32)

    if timeseries.shape[1] != atlas.n_rois:
        logger.warning(
            "Timeseries has %d ROIs but atlas declares %d. Check atlas/image alignment.",
            timeseries.shape[1],
            atlas.n_rois,
        )
    return timeseries


def compute_fisher_z_correlation(
    timeseries: np.ndarray,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Convert (T x N) timeseries -> (N x N) Fisher-z connectivity matrix.

    Parameters
    ----------
    timeseries : np.ndarray, shape (T, N)
        Parcellated BOLD signals (one column per ROI).
    clip_eps : float
        Values of |r| are clipped to ``1 - clip_eps`` before arctanh to avoid
        infinity on the diagonal.

    Returns
    -------
    np.ndarray, shape (N, N), dtype float32
        Symmetric matrix with zeroed diagonal (self-connectivity carries no
        information after Fisher-z; arctanh(1) = inf).
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected 2D timeseries, got shape {timeseries.shape}")

    # Numpy's corrcoef expects (variables, observations) when rowvar=False is
    # set; we pass the raw (T, N) and let it orient.
    r = np.corrcoef(timeseries, rowvar=False)
    r = np.clip(r, -1.0 + clip_eps, 1.0 - clip_eps)
    z = np.arctanh(r).astype(np.float32, copy=False)
    np.fill_diagonal(z, 0.0)
    return z
