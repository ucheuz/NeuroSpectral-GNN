"""Schaefer atlas loader with caching.

The Schaefer 2018 atlas (Schaefer et al., Cerebral Cortex 2018) provides
functionally-informed cortical parcellations at multiple resolutions (100, 200,
400, ... ROIs) and multiple network granularities (7 or 17 canonical networks,
Yeo 2011). For Project 65 we default to 100 ROIs / 7 networks, which gives a
100-node graph - a sweet spot between anatomical fidelity and fitting in 16GB
unified memory during GNN training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from nilearn import datasets

logger = logging.getLogger(__name__)

SchaeferROIs = Literal[100, 200, 300, 400, 500, 600, 800, 1000]
SchaeferNetworks = Literal[7, 17]


@dataclass(frozen=True)
class AtlasBundle:
    """Handle to a fetched parcellation atlas.

    Attributes
    ----------
    name : str
        Human-readable atlas identifier (e.g. ``'Schaefer2018_100Parcels_7Networks'``).
    maps_img : str
        Filesystem path to the 3D NIfTI label image.
    labels : Sequence[str]
        Names of the ROIs, ordered to match the integer labels in ``maps_img``.
        Length equals ``n_rois``.
    n_rois : int
        Number of ROIs (graph nodes).
    """

    name: str
    maps_img: str
    labels: Sequence[str]
    n_rois: int


def load_schaefer_atlas(
    n_rois: SchaeferROIs = 100,
    yeo_networks: SchaeferNetworks = 7,
    resolution_mm: int = 2,
    data_dir: str | Path | None = None,
) -> AtlasBundle:
    """Fetch and cache the Schaefer 2018 atlas.

    Nilearn caches the atlas under ``~/nilearn_data`` by default; pass
    ``data_dir`` to override (useful for HPC scratch directories).

    Parameters
    ----------
    n_rois : int
        One of {100, 200, 300, 400, 500, 600, 800, 1000}.
    yeo_networks : int
        Canonical network granularity: 7 (coarser) or 17 (finer).
    resolution_mm : int
        Voxel resolution of the atlas volume (1 or 2 mm MNI152). 2mm is
        plenty for standard resting-state fMRI and uses 8x less memory.
    data_dir : str or Path, optional
        Override for the nilearn cache directory.

    Returns
    -------
    AtlasBundle
    """
    if data_dir is not None:
        data_dir = str(Path(data_dir).expanduser().resolve())

    logger.info(
        "Loading Schaefer 2018 atlas: %d ROIs, %d networks, %dmm resolution",
        n_rois,
        yeo_networks,
        resolution_mm,
    )

    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois,
        yeo_networks=yeo_networks,
        resolution_mm=resolution_mm,
        data_dir=data_dir,
    )

    # nilearn returns label names as bytes in older versions; normalize to str.
    raw_labels = list(atlas.labels)
    labels: list[str] = [
        lab.decode("utf-8") if isinstance(lab, bytes) else str(lab)
        for lab in raw_labels
    ]

    # Some nilearn versions include a 'Background' label at index 0; drop it
    # so len(labels) == n_rois.
    if len(labels) == n_rois + 1 and labels[0].lower().startswith("background"):
        labels = labels[1:]

    if len(labels) != n_rois:
        logger.warning(
            "Expected %d labels but got %d from Schaefer atlas; "
            "downstream indexing may be off.",
            n_rois,
            len(labels),
        )

    name = f"Schaefer2018_{n_rois}Parcels_{yeo_networks}Networks"
    return AtlasBundle(
        name=name,
        maps_img=str(atlas.maps),
        labels=tuple(labels),
        n_rois=n_rois,
    )
