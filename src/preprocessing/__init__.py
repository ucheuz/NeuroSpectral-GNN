"""Preprocessing pipeline: fMRI NIfTI -> parcellated timeseries -> sparse PyG graph."""

from src.preprocessing.atlas import AtlasBundle, load_schaefer_atlas
from src.preprocessing.connectivity import (
    compute_fisher_z_correlation,
    extract_timeseries,
)
from src.preprocessing.graph import (
    GraphBuildConfig,
    connectivity_to_pyg_data,
)
from src.preprocessing.manifest import (
    SubjectRecord,
    TwinPair,
    build_twin_pairs,
    load_manifest,
)
from src.preprocessing.pipeline import (
    PreprocessConfig,
    preprocess_subject,
)
from src.preprocessing.synthetic import (
    SyntheticCohortConfig,
    empirical_heritability_from_connectivities,
    generate_cohort,
    save_synthetic_cohort,
)

__all__ = [
    "AtlasBundle",
    "GraphBuildConfig",
    "PreprocessConfig",
    "SubjectRecord",
    "SyntheticCohortConfig",
    "TwinPair",
    "build_twin_pairs",
    "compute_fisher_z_correlation",
    "connectivity_to_pyg_data",
    "empirical_heritability_from_connectivities",
    "extract_timeseries",
    "generate_cohort",
    "load_manifest",
    "load_schaefer_atlas",
    "preprocess_subject",
    "save_synthetic_cohort",
]
