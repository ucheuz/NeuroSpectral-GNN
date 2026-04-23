"""End-to-end per-subject preprocessing: NIfTI -> Fisher-z -> PyG Data -> .pt"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.preprocessing.atlas import AtlasBundle
from src.preprocessing.connectivity import (
    TimeseriesConfig,
    compute_fisher_z_correlation,
    extract_timeseries,
)
from src.preprocessing.graph import GraphBuildConfig, connectivity_to_pyg_data
from src.preprocessing.manifest import SubjectRecord

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Top-level orchestration config passed to :func:`preprocess_subject`."""

    output_dir: Path
    atlas: AtlasBundle
    ts_config: TimeseriesConfig = field(default_factory=TimeseriesConfig)
    graph_config: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    nilearn_cache_dir: Optional[Path] = None
    overwrite: bool = False


def _subject_output_path(config: PreprocessConfig, subject_id: str) -> Path:
    subjects_dir = config.output_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)
    return subjects_dir / f"{subject_id}.pt"


def preprocess_subject(
    record: SubjectRecord,
    config: PreprocessConfig,
) -> dict:
    """Run the full preprocessing pipeline for a single subject.

    Returns a status dict suitable for tabulating across a cohort:
        {
            'subject_id': ...,
            'status': 'ok' | 'skipped' | 'error',
            'output_path': str,
            'elapsed_s': float,
            'n_edges': int,
            'error': str (only if status=='error'),
        }
    """
    t0 = time.perf_counter()
    out_path = _subject_output_path(config, record.subject_id)

    if out_path.exists() and not config.overwrite:
        return {
            "subject_id": record.subject_id,
            "status": "skipped",
            "output_path": str(out_path),
            "elapsed_s": 0.0,
            "n_edges": None,
        }

    # Per-subject t_r override takes precedence over global default.
    ts_config = config.ts_config
    if record.t_r is not None and record.t_r != ts_config.t_r:
        ts_config = TimeseriesConfig(
            standardize=ts_config.standardize,
            detrend=ts_config.detrend,
            low_pass=ts_config.low_pass,
            high_pass=ts_config.high_pass,
            t_r=record.t_r,
            smoothing_fwhm=ts_config.smoothing_fwhm,
            memory_level=ts_config.memory_level,
        )

    try:
        timeseries = extract_timeseries(
            nii_path=record.nii_path,
            atlas=config.atlas,
            ts_config=ts_config,
            confounds_path=record.confounds_path,
            cache_dir=config.nilearn_cache_dir,
        )
        connectivity = compute_fisher_z_correlation(timeseries)
        data = connectivity_to_pyg_data(
            connectivity,
            config.graph_config,
            metadata={
                "subject_id": record.subject_id,
                "family_id": record.family_id,
                "twin_id": record.twin_id,
                "zygosity": record.zygosity,
                "atlas_name": config.atlas.name,
            },
        )

        torch.save(data, out_path)

        elapsed = time.perf_counter() - t0
        n_edges = int(data.edge_index.size(1))
        logger.info(
            "[%s] ok | nodes=%d edges=%d t=%.2fs",
            record.subject_id,
            data.num_nodes,
            n_edges,
            elapsed,
        )
        return {
            "subject_id": record.subject_id,
            "status": "ok",
            "output_path": str(out_path),
            "elapsed_s": elapsed,
            "n_edges": n_edges,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "[%s] preprocessing failed after %.2fs", record.subject_id, elapsed
        )
        return {
            "subject_id": record.subject_id,
            "status": "error",
            "output_path": str(out_path),
            "elapsed_s": elapsed,
            "n_edges": None,
            "error": repr(exc),
        }
