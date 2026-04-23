"""Heritability estimation from twin pair embeddings.

We extend the classical Falconer estimator

        h^2_Falconer = 2 * (r_MZ - r_DZ)

from phenotype correlations to learned embedding similarities:

        s_ab = cos(z_a, z_b) in [-1, 1]
        h^2_GNN = clamp(2 * (mean(s_MZ) - mean(s_DZ)), 0, 1)

This is differentiable wrt the embeddings (useful as an auxiliary objective in
Month 3) and reduces to the classical estimator when the embedding dimension
equals the number of phenotypes.

We also bootstrap the MZ/DZ similarity samples to get a 95% confidence
interval on the estimate - essential for the dissertation's quantitative
claim and for the grant's "recovered h^2 tracks ground truth within CI"
figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

SimilarityMetric = Literal["cosine", "neg_euclidean"]


@dataclass
class HeritabilityEstimate:
    """Point estimate and uncertainty of h^2 from a cohort of twin embeddings."""

    h2: float
    r_mz: float
    r_dz: float
    n_mz: int
    n_dz: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    bootstrap_mean: Optional[float] = None
    method: str = "cosine"

    def as_dict(self) -> dict:
        return {
            "h2": self.h2,
            "r_mz": self.r_mz,
            "r_dz": self.r_dz,
            "n_mz": self.n_mz,
            "n_dz": self.n_dz,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "bootstrap_mean": self.bootstrap_mean,
            "method": self.method,
        }


def _pair_similarity(
    z_a: Tensor, z_b: Tensor, metric: SimilarityMetric
) -> Tensor:
    """Return per-pair similarity scores in [-1, 1] (cosine) or R (neg_euclidean)."""
    if metric == "cosine":
        # Unit-normalise defensively; if upstream already normalised this is a no-op.
        za = F.normalize(z_a, dim=-1, eps=1e-8)
        zb = F.normalize(z_b, dim=-1, eps=1e-8)
        return (za * zb).sum(dim=-1)
    if metric == "neg_euclidean":
        return -(((z_a - z_b) ** 2).sum(dim=-1).clamp(min=1e-12).sqrt())
    raise ValueError(f"Unknown similarity metric: {metric!r}")


def pair_similarities_from_embeddings(
    z_a: Tensor,
    z_b: Tensor,
    zygosities: Sequence[str],
    metric: SimilarityMetric = "cosine",
) -> dict[str, Tensor]:
    """Bucket per-pair similarities by zygosity.

    Parameters
    ----------
    z_a, z_b : Tensor [B, D]
    zygosities : sequence of {'MZ', 'DZ', 'UNREL'} of length B
    metric : similarity metric

    Returns
    -------
    dict mapping zygosity label -> 1D tensor of similarities.
    """
    sims = _pair_similarity(z_a, z_b, metric)
    buckets: dict[str, list[Tensor]] = {"MZ": [], "DZ": [], "UNREL": []}
    for s, z in zip(sims, zygosities):
        buckets.setdefault(z, []).append(s)
    return {
        k: (torch.stack(v) if v else sims.new_empty(0))
        for k, v in buckets.items()
    }


def falconer_h2(
    mz_similarities: Tensor,
    dz_similarities: Tensor,
    bootstrap: int = 0,
    rng_seed: int = 0,
    clamp: bool = True,
    method: str = "cosine",
) -> HeritabilityEstimate:
    """Estimate h^2 = 2*(mean(MZ) - mean(DZ)) with optional bootstrap CI.

    Parameters
    ----------
    mz_similarities, dz_similarities : 1D tensors (any device)
    bootstrap : int
        Number of bootstrap resamples for a 95% CI. 0 disables.
    rng_seed : int
    clamp : bool
        If True, clip the point estimate to [0, 1].
    method : str
        Passed through to the returned estimate for bookkeeping.
    """
    if mz_similarities.numel() == 0 or dz_similarities.numel() == 0:
        return HeritabilityEstimate(
            h2=float("nan"),
            r_mz=float("nan"),
            r_dz=float("nan"),
            n_mz=int(mz_similarities.numel()),
            n_dz=int(dz_similarities.numel()),
            method=method,
        )

    mz_np = mz_similarities.detach().cpu().numpy().astype(np.float64)
    dz_np = dz_similarities.detach().cpu().numpy().astype(np.float64)

    r_mz = float(mz_np.mean())
    r_dz = float(dz_np.mean())
    h2 = 2.0 * (r_mz - r_dz)
    if clamp:
        h2 = float(np.clip(h2, 0.0, 1.0))

    ci_low = ci_high = boot_mean = None
    if bootstrap > 0:
        rng = np.random.default_rng(rng_seed)
        n_mz, n_dz = mz_np.size, dz_np.size
        draws = np.empty(bootstrap, dtype=np.float64)
        for i in range(bootstrap):
            mz_sample = mz_np[rng.integers(0, n_mz, size=n_mz)]
            dz_sample = dz_np[rng.integers(0, n_dz, size=n_dz)]
            h = 2.0 * (mz_sample.mean() - dz_sample.mean())
            draws[i] = np.clip(h, 0.0, 1.0) if clamp else h
        ci_low = float(np.percentile(draws, 2.5))
        ci_high = float(np.percentile(draws, 97.5))
        boot_mean = float(draws.mean())

    return HeritabilityEstimate(
        h2=h2,
        r_mz=r_mz,
        r_dz=r_dz,
        n_mz=int(mz_np.size),
        n_dz=int(dz_np.size),
        ci_low=ci_low,
        ci_high=ci_high,
        bootstrap_mean=boot_mean,
        method=method,
    )


class HeritabilityHead(nn.Module):
    """Differentiable module: embeddings + zygosities -> h^2 estimate.

    Can be wired as an auxiliary training objective (Month 3). Currently we
    expose it as a stateless forward pass for validation-time use.

    Note: this module has zero learnable parameters; it's an ``nn.Module`` only
    so it plays nicely with ``model.train()`` / ``model.eval()`` hooks and
    auto-moves with ``.to(device)``.
    """

    def __init__(
        self,
        metric: SimilarityMetric = "cosine",
        clamp: bool = True,
    ):
        super().__init__()
        self.metric = metric
        self.clamp = clamp

    def forward(
        self,
        z_a: Tensor,
        z_b: Tensor,
        zygosities: Sequence[str],
    ) -> Tensor:
        """Return a scalar tensor ĥ² with autograd preserved."""
        sims = _pair_similarity(z_a, z_b, self.metric)
        is_mz = torch.tensor(
            [z == "MZ" for z in zygosities], device=sims.device, dtype=torch.bool
        )
        is_dz = torch.tensor(
            [z == "DZ" for z in zygosities], device=sims.device, dtype=torch.bool
        )
        if is_mz.sum() == 0 or is_dz.sum() == 0:
            return sims.new_tensor(float("nan"))
        r_mz = sims[is_mz].mean()
        r_dz = sims[is_dz].mean()
        h2 = 2.0 * (r_mz - r_dz)
        if self.clamp:
            h2 = h2.clamp(0.0, 1.0)
        return h2


def twin_separation_metrics(
    z_a: Tensor,
    z_b: Tensor,
    zygosities: Sequence[str],
    distance_metric: Literal["cosine", "euclidean"] = "cosine",
) -> dict[str, float]:
    """Summary metrics for how well MZ pairs separate from DZ pairs.

    When UNREL pairs are present, their mean distance is also reported; these
    should typically be largest (no shared family genetics).

    Returns
    -------
    dict with keys:
        'mean_mz_distance', 'mean_dz_distance', 'mean_unrel_distance',
        'distance_gap', 'auc', 'pair_accuracy', 'n_mz', 'n_dz', 'n_unrel'
    """
    if distance_metric == "cosine":
        d = 1.0 - (
            F.normalize(z_a, dim=-1, eps=1e-8) * F.normalize(z_b, dim=-1, eps=1e-8)
        ).sum(dim=-1)
    elif distance_metric == "euclidean":
        d = ((z_a - z_b) ** 2).sum(dim=-1).clamp(min=1e-12).sqrt()
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric!r}")

    d_np = d.detach().cpu().numpy()
    zy = np.asarray(zygosities)
    mz_d = d_np[zy == "MZ"]
    dz_d = d_np[zy == "DZ"]
    unrel_d = d_np[zy == "UNREL"]

    # AUC: treat MZ as "positive" (should have small distance). Flip sign so
    # a larger "score" predicts MZ, then compute ROC-AUC with sklearn.
    auc: float
    if mz_d.size > 0 and dz_d.size > 0:
        try:
            from sklearn.metrics import roc_auc_score

            y_true = np.concatenate(
                [np.ones(mz_d.size), np.zeros(dz_d.size)]
            )
            y_score = np.concatenate([-mz_d, -dz_d])
            auc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc = float("nan")
    else:
        auc = float("nan")

    # Threshold split at midpoint of cohort mean MZ & DZ distances; predict MZ
    # when distance is below the threshold (embeddings "closer" for true MZ).
    pair_accuracy: float
    if mz_d.size > 0 and dz_d.size > 0:
        t = 0.5 * (float(np.mean(mz_d)) + float(np.mean(dz_d)))
        n_correct = (mz_d < t).sum() + (dz_d >= t).sum()
        pair_accuracy = float(n_correct / (mz_d.size + dz_d.size))
    else:
        pair_accuracy = float("nan")

    return {
        "mean_mz_distance": float(mz_d.mean()) if mz_d.size else float("nan"),
        "mean_dz_distance": float(dz_d.mean()) if dz_d.size else float("nan"),
        "mean_unrel_distance": float(unrel_d.mean()) if unrel_d.size else float("nan"),
        "distance_gap": float((dz_d.mean() if dz_d.size else 0.0)
                              - (mz_d.mean() if mz_d.size else 0.0)),
        "auc": auc,
        "pair_accuracy": pair_accuracy,
        "n_mz": int(mz_d.size),
        "n_dz": int(dz_d.size),
        "n_unrel": int(unrel_d.size),
    }


# --- Classical (Falconer) per-feature h^2 from raw twin-pair phenotypes ---------------


def pearson_twin_phenotype_r(x_a: np.ndarray, x_b: np.ndarray) -> float:
    """Pearson r between co-twins' same scalar phenotype across *pairs* (one value per subject per pair)."""
    x_a = np.asarray(x_a, dtype=np.float64).ravel()
    x_b = np.asarray(x_b, dtype=np.float64).ravel()
    if x_a.size < 2 or x_b.size < 2 or x_a.size != x_b.size:
        return 0.0
    if float(np.std(x_a)) < 1e-12 or float(np.std(x_b)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x_a, x_b)[0, 1])


def per_feature_falconer_h2(
    twin_a_mz: np.ndarray,
    twin_b_mz: np.ndarray,
    twin_a_dz: np.ndarray,
    twin_b_dz: np.ndarray,
) -> np.ndarray:
    """Falconer h^2 = 2(r_MZ − r_DZ) per feature dimension, clamped to [0, 1].

    twin_* shapes must share (P_*, D). For each j in 0..D-1, r is the
    across-pairs Pearson correlation of the twin-A vector vs twin-B vector for
    that feature (KCL P65 classical baseline vs the GNN).
    """
    twin_a_mz = np.asarray(twin_a_mz, dtype=np.float64)
    twin_b_mz = np.asarray(twin_b_mz, dtype=np.float64)
    twin_a_dz = np.asarray(twin_a_dz, dtype=np.float64)
    twin_b_dz = np.asarray(twin_b_dz, dtype=np.float64)
    if (
        twin_a_mz.shape != twin_b_mz.shape
        or twin_a_dz.shape != twin_b_dz.shape
    ):
        raise ValueError("MZ and DZ twin A/B array pairs must have matching shape.")
    d = int(twin_a_mz.shape[1])
    out = np.empty(d, dtype=np.float32)
    for j in range(d):
        r_mz = pearson_twin_phenotype_r(twin_a_mz[:, j], twin_b_mz[:, j])
        r_dz = pearson_twin_phenotype_r(twin_a_dz[:, j], twin_b_dz[:, j])
        out[j] = float(np.clip(2.0 * (r_mz - r_dz), 0.0, 1.0))
    return out
