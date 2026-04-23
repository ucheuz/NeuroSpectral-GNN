"""Synthetic twin connectome generator with tunable ground-truth heritability.

This module lets us prototype and unit-test the full Siamese GNN + contrastive
loss pipeline *before* real twin fMRI data is available, and - more
importantly - gives us a controlled benchmark for the grant proposal: a
scientist can set ground-truth h^2 and check that our GNN recovers it.

Generative model
----------------
Following the classical ACE decomposition (Falconer 1960; Neale & Cardon 1992)
applied entrywise to the upper triangle of the connectivity matrix:

        C_i = mu + A_i + E_i

    A_i ~ N(0, sigma_a^2)   (additive genetic component)
    E_i ~ N(0, sigma_e^2)   (unique environment component)
    mu                       (population-mean connectivity template)

Twin sharing rule:

    MZ:     A_{j} = A_{i}                                (identical)
    DZ:     A_{j} = rho * A_{i} + sqrt(1 - rho^2) * A'   (rho = 0.5)
    UNREL:  A_{j} independent of A_{i}

Ground-truth narrow-sense heritability is

        h^2 = sigma_a^2 / (sigma_a^2 + sigma_e^2).

We scale sigma_a^2 + sigma_e^2 = 1 without loss of generality and then
apply arctanh to match the Fisher-z distribution produced by the real
preprocessing pipeline - so downstream code is agnostic to the data source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch

from src.preprocessing.atlas import AtlasBundle
from src.preprocessing.graph import GraphBuildConfig, connectivity_to_pyg_data
from src.preprocessing.manifest import TwinPair, ZYGOSITY_TO_LABEL

logger = logging.getLogger(__name__)

Zygosity = Literal["MZ", "DZ", "UNREL"]


@dataclass
class SyntheticCohortConfig:
    """Controls the composition and statistics of the synthetic cohort."""

    n_mz_pairs: int = 40
    n_dz_pairs: int = 40
    n_unrelated_pairs: int = 0  # extra negatives, optional

    n_rois: int = 100
    heritability: float = 0.6  # h^2 in [0, 1]
    total_variance: float = 1.0  # sigma_a^2 + sigma_e^2
    dz_genetic_correlation: float = 0.5  # expected for DZ twins

    # Population-mean connectivity template: 'zero' | 'block' | np.ndarray
    # 'block' creates a stochastic-block-model-like structure with 7 modules
    # to crudely mimic the canonical Yeo networks.
    mean_template: str = "block"
    n_modules: int = 7
    within_module_mean: float = 0.5
    between_module_mean: float = 0.05

    # Fisher-z style: clip correlation to |r| < 1 before arctanh so tails
    # aren't pathological. Set to >= 1 to skip the arctanh step entirely.
    apply_arctanh: bool = True
    clip_eps: float = 1e-3

    # Polygenic risk score (PRS) modality - emulated genetic summary vector
    # shared between twins with the same correlation structure as the
    # additive-genetic connectivity component.
    prs_dim: int = 0  # 0 disables PRS generation
    prs_informativeness: float = 1.0  # scales signal-to-noise of PRS
    prs_noise_scale: float = 0.25  # env/measurement noise added per subject

    seed: int = 42

    # Graph construction and output
    graph_config: GraphBuildConfig = field(default_factory=GraphBuildConfig)


def _build_mean_template(cfg: SyntheticCohortConfig, rng: np.random.Generator) -> np.ndarray:
    """Construct the population-mean connectivity matrix mu."""
    n = cfg.n_rois
    if cfg.mean_template == "zero":
        return np.zeros((n, n), dtype=np.float32)
    if cfg.mean_template == "block":
        # Assign each ROI to one of n_modules communities (roughly equal).
        module_assign = rng.integers(0, cfg.n_modules, size=n)
        same = module_assign[:, None] == module_assign[None, :]
        mu = np.where(same, cfg.within_module_mean, cfg.between_module_mean).astype(np.float32)
        np.fill_diagonal(mu, 0.0)
        return mu
    raise ValueError(f"Unknown mean_template: {cfg.mean_template!r}")


def _sample_symmetric_gaussian(
    n: int, scale: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample an N x N symmetric zero-mean matrix with i.i.d. upper-triangle entries."""
    noise = rng.normal(scale=scale, size=(n, n)).astype(np.float32)
    sym = 0.5 * (noise + noise.T)
    np.fill_diagonal(sym, 0.0)
    return sym


def _arctanh_clip(matrix: np.ndarray, eps: float) -> np.ndarray:
    """Clip to |r| < 1 - eps then apply Fisher z-transform."""
    clipped = np.clip(matrix, -1.0 + eps, 1.0 - eps)
    return np.arctanh(clipped).astype(np.float32, copy=False)


def _sample_prs_pair(
    cfg: SyntheticCohortConfig,
    shared_genetic_correlation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a twin-pair's PRS vectors with given genetic correlation.

    Implementation:
        g_a ~ N(0, informativeness)
        g_b = rho * g_a + sqrt(1 - rho^2) * g'  (same construction as A_i)
        plus independent measurement noise on each subject.
    """
    d = cfg.prs_dim
    signal_scale = cfg.prs_informativeness
    g_a = rng.normal(scale=signal_scale, size=d).astype(np.float32)
    g_prime = rng.normal(scale=signal_scale, size=d).astype(np.float32)
    rho = shared_genetic_correlation
    g_b = rho * g_a + np.sqrt(max(1.0 - rho**2, 0.0)) * g_prime
    # Independent noise per subject (measurement error / non-genetic effects)
    g_a = g_a + rng.normal(scale=cfg.prs_noise_scale, size=d).astype(np.float32)
    g_b = g_b + rng.normal(scale=cfg.prs_noise_scale, size=d).astype(np.float32)
    return g_a.astype(np.float32), g_b.astype(np.float32)


def generate_cohort(
    cfg: SyntheticCohortConfig,
) -> tuple[dict[str, np.ndarray], list[TwinPair], dict[str, np.ndarray]]:
    """Generate synthetic connectivity matrices (+ optional PRS vectors).

    Returns
    -------
    connectivities : dict[subject_id -> np.ndarray]
        Per-subject N x N connectivity matrices (Fisher-z'd if configured).
    pairs : list[TwinPair]
        The twin-pair metadata table.
    prs_vectors : dict[subject_id -> np.ndarray]
        Per-subject PRS vectors (shape ``[prs_dim]``). Empty dict if
        ``cfg.prs_dim == 0``.
    """
    if not 0.0 <= cfg.heritability <= 1.0:
        raise ValueError(f"heritability must be in [0, 1], got {cfg.heritability}")
    if cfg.total_variance <= 0:
        raise ValueError("total_variance must be positive")

    rng = np.random.default_rng(cfg.seed)
    sigma_a = float(np.sqrt(cfg.heritability * cfg.total_variance))
    sigma_e = float(np.sqrt((1.0 - cfg.heritability) * cfg.total_variance))
    rho_dz = cfg.dz_genetic_correlation

    mu = _build_mean_template(cfg, rng)

    connectivities: dict[str, np.ndarray] = {}
    prs_vectors: dict[str, np.ndarray] = {}
    pairs: list[TwinPair] = []
    n = cfg.n_rois

    def _finalize(raw_corr: np.ndarray) -> np.ndarray:
        if cfg.apply_arctanh:
            return _arctanh_clip(raw_corr, cfg.clip_eps)
        out = raw_corr.astype(np.float32, copy=False)
        np.fill_diagonal(out, 0.0)
        return out

    family_counter = 0

    def _emit_pair(zyg: Zygosity, A_a: np.ndarray, A_b: np.ndarray,
                   prs_rho: float) -> None:
        nonlocal family_counter
        E_a = _sample_symmetric_gaussian(n, sigma_e, rng)
        E_b = _sample_symmetric_gaussian(n, sigma_e, rng)
        raw_a = mu + A_a + E_a
        raw_b = mu + A_b + E_b
        fid = f"FAM_{zyg}_{family_counter:04d}"
        family_counter += 1
        sa = f"{fid}_A"
        sb = f"{fid}_B"
        connectivities[sa] = _finalize(raw_a)
        connectivities[sb] = _finalize(raw_b)
        if cfg.prs_dim > 0:
            g_a, g_b = _sample_prs_pair(cfg, prs_rho, rng)
            prs_vectors[sa] = g_a
            prs_vectors[sb] = g_b
        pairs.append(
            TwinPair(
                family_id=fid,
                subject_a=sa,
                subject_b=sb,
                zygosity=zyg,
                label=ZYGOSITY_TO_LABEL[zyg],
            )
        )

    for _ in range(cfg.n_mz_pairs):
        A = _sample_symmetric_gaussian(n, sigma_a, rng)
        _emit_pair("MZ", A, A.copy(), prs_rho=1.0)

    for _ in range(cfg.n_dz_pairs):
        A = _sample_symmetric_gaussian(n, sigma_a, rng)
        A_prime = _sample_symmetric_gaussian(n, sigma_a, rng)
        A_b = rho_dz * A + np.sqrt(max(1.0 - rho_dz**2, 0.0)) * A_prime
        _emit_pair("DZ", A, A_b.astype(np.float32, copy=False), prs_rho=rho_dz)

    for _ in range(cfg.n_unrelated_pairs):
        A1 = _sample_symmetric_gaussian(n, sigma_a, rng)
        A2 = _sample_symmetric_gaussian(n, sigma_a, rng)
        _emit_pair("UNREL", A1, A2, prs_rho=0.0)

    logger.info(
        "Generated synthetic cohort: %d subjects, %d pairs "
        "(MZ=%d, DZ=%d, UNREL=%d), h^2=%.2f, sigma_a=%.3f, sigma_e=%.3f, prs_dim=%d",
        len(connectivities),
        len(pairs),
        cfg.n_mz_pairs,
        cfg.n_dz_pairs,
        cfg.n_unrelated_pairs,
        cfg.heritability,
        sigma_a,
        sigma_e,
        cfg.prs_dim,
    )
    return connectivities, pairs, prs_vectors


def _synthetic_atlas(n_rois: int) -> AtlasBundle:
    """Build a dummy AtlasBundle matching the synthetic n_rois. No file IO."""
    return AtlasBundle(
        name=f"Synthetic_{n_rois}ROIs",
        maps_img="<synthetic>",
        labels=tuple(f"ROI_{i:04d}" for i in range(n_rois)),
        n_rois=n_rois,
    )


def save_synthetic_cohort(
    cfg: SyntheticCohortConfig,
    output_dir: str | Path,
    atlas: Optional[AtlasBundle] = None,
) -> dict[str, Path]:
    """Generate a synthetic cohort and persist it in the same layout as the
    real preprocessing pipeline, so it can be consumed by ``TwinBrainDataset``
    without code changes.

    Outputs
    -------
        {output_dir}/subjects/{subject_id}.pt
        {output_dir}/pairs.csv
        {output_dir}/config.json
    """
    import json
    from dataclasses import asdict

    output_dir = Path(output_dir).expanduser()
    subjects_dir = output_dir / "subjects"
    subjects_dir.mkdir(parents=True, exist_ok=True)

    atlas = atlas or _synthetic_atlas(cfg.n_rois)

    connectivities, pairs, prs_vectors = generate_cohort(cfg)

    subject_to_pair: dict[str, TwinPair] = {}
    for p in pairs:
        subject_to_pair[p.subject_a] = p
        subject_to_pair[p.subject_b] = p

    for subject_id, conn in connectivities.items():
        pair = subject_to_pair[subject_id]
        twin_id = "A" if subject_id.endswith("_A") else "B"
        data = connectivity_to_pyg_data(
            conn,
            cfg.graph_config,
            metadata={
                "subject_id": subject_id,
                "family_id": pair.family_id,
                "twin_id": twin_id,
                "zygosity": pair.zygosity,
                "atlas_name": atlas.name,
                "synthetic_heritability": cfg.heritability,
            },
        )
        if cfg.prs_dim > 0 and subject_id in prs_vectors:
            # Stored as [1, prs_dim] so PyG's Batch concatenates cleanly into
            # [batch_size, prs_dim] across the first dim.
            data.prs = torch.from_numpy(prs_vectors[subject_id]).unsqueeze(0)
        torch.save(data, subjects_dir / f"{subject_id}.pt")

    pairs_df = pd.DataFrame(
        [
            {
                "family_id": p.family_id,
                "subject_a": p.subject_a,
                "subject_b": p.subject_b,
                "zygosity": p.zygosity,
                "label": p.label,
            }
            for p in pairs
        ]
    )
    pairs_df.to_csv(output_dir / "pairs.csv", index=False)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "cohort": asdict(cfg),
                "atlas": atlas.name,
            },
            indent=2,
            default=str,
        )
    )

    logger.info(
        "Saved %d synthetic subjects and %d pairs to %s",
        len(connectivities),
        len(pairs),
        output_dir,
    )
    return {
        "subjects_dir": subjects_dir,
        "pairs_csv": output_dir / "pairs.csv",
        "config_json": output_dir / "config.json",
    }


def empirical_heritability_from_connectivities(
    connectivities: dict[str, np.ndarray],
    pairs: list[TwinPair],
) -> float:
    """Falconer's h^2 estimate directly on raw connectivity matrices.

    Given twin-pair ICCs r_MZ and r_DZ, Falconer's formula is:

        h^2 = 2 * (r_MZ - r_DZ)

    We compute each pair's ICC as the Pearson correlation between the
    vectorised upper triangles of (A, B), then average within zygosity.

    Returns
    -------
    float
        Estimated h^2 in [0, 1] (can be slightly negative due to noise).
    """
    iu = np.triu_indices_from(next(iter(connectivities.values())), k=1)

    def _pair_icc(a: np.ndarray, b: np.ndarray) -> float:
        va = a[iu]
        vb = b[iu]
        if va.std() == 0 or vb.std() == 0:
            return 0.0
        return float(np.corrcoef(va, vb)[0, 1])

    mz_iccs = [
        _pair_icc(connectivities[p.subject_a], connectivities[p.subject_b])
        for p in pairs if p.zygosity == "MZ"
    ]
    dz_iccs = [
        _pair_icc(connectivities[p.subject_a], connectivities[p.subject_b])
        for p in pairs if p.zygosity == "DZ"
    ]
    if not mz_iccs or not dz_iccs:
        logger.warning("Need both MZ and DZ pairs to estimate h^2; returning NaN")
        return float("nan")
    return 2.0 * (float(np.mean(mz_iccs)) - float(np.mean(dz_iccs)))
