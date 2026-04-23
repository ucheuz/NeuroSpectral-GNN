"""Convert an (N x N) connectivity matrix to a sparse PyTorch Geometric graph.

Sparsification primer:
    A raw Fisher-z connectivity matrix is effectively a fully-connected graph
    with ~N^2 edges. Feeding that directly to GCNConv collapses message
    passing to a weighted MLP because every node aggregates from every other,
    destroying the inductive bias of graph locality. The standard fix in
    network neuroscience is to retain only the strongest edges:

    * Proportional thresholding (default): keep the top X% of edges by |z|
      globally, preserving symmetry. Recommended X in [10%, 30%].
    * Top-k per node: keep each node's k strongest neighbours. More uniform
      node degree but can introduce directionality artefacts.

    We drop the sign of the correlation from the graph topology (edges are
    unsigned) but *retain it* in edge_attr, so the GCN message passing can
    still condition on positive vs. negative coupling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

SparsifyStrategy = Literal["proportional", "topk", "absolute"]
NodeFeatureMode = Literal["profile", "identity", "degree_profile"]


@dataclass(frozen=True)
class GraphBuildConfig:
    """Configuration for converting a connectivity matrix to a PyG Data object."""

    sparsify_strategy: SparsifyStrategy = "proportional"
    keep_top_fraction: float = 0.20  # used if strategy == 'proportional'
    topk_per_node: int = 10  # used if strategy == 'topk'
    absolute_threshold: float = 0.3  # used if strategy == 'absolute'
    use_absolute_weights_for_ranking: bool = True
    node_feature_mode: NodeFeatureMode = "profile"
    include_edge_attr: bool = True
    self_loops: bool = False  # GCNConv adds its own via add_self_loops=True


def _proportional_mask(
    matrix: np.ndarray,
    keep_fraction: float,
    use_abs: bool,
) -> np.ndarray:
    """Return a symmetric boolean mask keeping the top ``keep_fraction`` edges.

    Operates on the upper triangle to guarantee symmetry, then mirrors.
    """
    n = matrix.shape[0]
    iu = np.triu_indices(n, k=1)
    weights = np.abs(matrix[iu]) if use_abs else matrix[iu]
    n_edges = len(weights)
    n_keep = max(1, int(round(keep_fraction * n_edges)))
    if n_keep >= n_edges:
        threshold_mask = np.ones(n_edges, dtype=bool)
    else:
        # argpartition is O(n) vs full sort's O(n log n); critical for N=400.
        cutoff_idx = np.argpartition(-weights, n_keep - 1)[:n_keep]
        threshold_mask = np.zeros(n_edges, dtype=bool)
        threshold_mask[cutoff_idx] = True

    mask = np.zeros_like(matrix, dtype=bool)
    mask[iu] = threshold_mask
    mask = mask | mask.T
    return mask


def _topk_mask(matrix: np.ndarray, k: int, use_abs: bool) -> np.ndarray:
    """Keep the top-k neighbours per row, then symmetrize via union."""
    n = matrix.shape[0]
    weights = np.abs(matrix) if use_abs else matrix
    np.fill_diagonal(weights, -np.inf)
    idx = np.argpartition(-weights, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    mask = np.zeros_like(matrix, dtype=bool)
    rows = np.repeat(np.arange(n), k)
    mask[rows, idx.ravel()] = True
    mask = mask | mask.T
    return mask


def _absolute_mask(matrix: np.ndarray, threshold: float) -> np.ndarray:
    mask = np.abs(matrix) >= threshold
    np.fill_diagonal(mask, False)
    return mask


def _build_node_features(
    connectivity: np.ndarray,
    mode: NodeFeatureMode,
) -> torch.Tensor:
    """Compute node features X in R^{N x F} from the connectivity matrix."""
    n = connectivity.shape[0]
    if mode == "profile":
        # Each node's feature = its row of the Fisher-z matrix. This gives the
        # GNN access to the full connectivity profile before any message
        # passing, which empirically outperforms identity features for brain
        # graphs (Kawahara et al., 2017; Kim & Ye, 2020).
        feats = connectivity.astype(np.float32, copy=True)
    elif mode == "identity":
        feats = np.eye(n, dtype=np.float32)
    elif mode == "degree_profile":
        # Summary statistics per row: mean, std, max-abs, degree (# edges).
        # Low-dim alternative for memory-constrained runs on Schaefer-400+.
        stats = np.stack(
            [
                connectivity.mean(axis=1),
                connectivity.std(axis=1),
                np.abs(connectivity).max(axis=1),
                (connectivity != 0).sum(axis=1).astype(np.float32),
            ],
            axis=1,
        )
        feats = stats.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown node_feature_mode: {mode!r}")
    return torch.from_numpy(feats)


def connectivity_to_pyg_data(
    connectivity: np.ndarray,
    config: GraphBuildConfig,
    metadata: dict | None = None,
) -> Data:
    """Convert an NxN connectivity matrix into a ``torch_geometric.data.Data``.

    Parameters
    ----------
    connectivity : np.ndarray, shape (N, N)
        Fisher-z-transformed symmetric connectivity matrix with zero diagonal.
    config : GraphBuildConfig
    metadata : dict, optional
        Arbitrary key/value pairs (e.g. subject_id, zygosity, family_id) that
        will be attached as attributes on the Data object for downstream use.

    Returns
    -------
    torch_geometric.data.Data
        Fields:
            x             : FloatTensor [N, F]   node features
            edge_index    : LongTensor  [2, E]   COO edge indices
            edge_attr     : FloatTensor [E]      Fisher-z edge weights (if enabled)
            num_nodes     : int
            connectivity  : FloatTensor [N, N]   full matrix (for heritability calc)
            + any metadata fields
    """
    if connectivity.ndim != 2 or connectivity.shape[0] != connectivity.shape[1]:
        raise ValueError(
            f"connectivity must be square 2D, got {connectivity.shape}"
        )
    if not np.allclose(connectivity, connectivity.T, atol=1e-5):
        logger.warning("Connectivity matrix is not symmetric; symmetrising.")
        connectivity = 0.5 * (connectivity + connectivity.T)

    # 1. Pick edges
    if config.sparsify_strategy == "proportional":
        mask = _proportional_mask(
            connectivity,
            config.keep_top_fraction,
            config.use_absolute_weights_for_ranking,
        )
    elif config.sparsify_strategy == "topk":
        mask = _topk_mask(
            connectivity,
            config.topk_per_node,
            config.use_absolute_weights_for_ranking,
        )
    elif config.sparsify_strategy == "absolute":
        mask = _absolute_mask(connectivity, config.absolute_threshold)
    else:
        raise ValueError(f"Unknown sparsify_strategy: {config.sparsify_strategy!r}")

    if not config.self_loops:
        np.fill_diagonal(mask, False)

    # 2. Build COO edge_index + edge_attr
    src, dst = np.nonzero(mask)
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()
    edge_attr = (
        torch.from_numpy(connectivity[src, dst].astype(np.float32))
        if config.include_edge_attr
        else None
    )

    # 3. Node features
    x = _build_node_features(connectivity, config.node_feature_mode)

    # 4. Assemble Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=connectivity.shape[0],
    )
    # Keep the full dense matrix around for heritability / ablation analyses.
    # It's tiny (100x100 float32 = 40 KB) and saves re-computation later.
    data.connectivity = torch.from_numpy(connectivity.astype(np.float32))

    if metadata:
        for k, v in metadata.items():
            setattr(data, k, v)

    return data
