"""Siamese Graph Neural Network for twin brain connectome metric learning.

Architecture
------------
Twin A's graph G_a = (X_a, A_a)  ->  encoder f_theta  ->  z_a in R^d
Twin B's graph G_b = (X_b, A_b)  ->  encoder f_theta  ->  z_b in R^d
                                                          |
                                                          v
                                           contrastive_loss(z_a, z_b, y)

Design notes (for the grant proposal: spectral + metric-learning storyline)
--------------------------------------------------------------------------
1. Shared-weights encoder = same function applied to both twins, so the
   learned embedding is twin-agnostic.
2. ``GCNConv`` implements a first-order Chebyshev approximation of the
   spectral graph convolution with the symmetric normalised Laplacian
   L_sym = I - D^{-1/2} A D^{-1/2}. Every forward pass is effectively a
   low-pass filter on the graph spectrum - the inductive bias we want for
   functional connectomes.
3. A projection head (2-layer MLP) sits between the graph embedding and the
   contrastive loss. Following SimCLR/BYOL practice, at inference time we
   evaluate on the *encoder* output, not the projection - the projection is
   trained-and-thrown-away; it gives the encoder room to learn richer
   representations.
4. Dropout + BatchNorm regularise the tiny twin cohort (~80 pairs) which
   would otherwise overfit trivially.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool


PoolingStrategy = Literal["mean", "add", "mean+add"]


@dataclass
class SiameseConfig:
    """Hyperparameters for the Siamese encoder + projection head."""

    in_channels: int = 100  # node feature dim; sum(modality_feature_dims) if set
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    use_edge_weight: bool = True  # feed Fisher-z weights into GCNConv
    pooling: PoolingStrategy = "mean+add"
    projection_dim: int = 32
    projection_hidden: int = 64
    normalize_embeddings: bool = True

    # --- MRI supervoxel / cross-modal fusion (KCL P65: DWI+T1+FLAIR) ---
    # If ``use_cross_modal_attention`` is True, ``x`` is split into contiguous
    # blocks of sizes ``modality_feature_dims`` (must sum to ``in_channels``).
    # A MultiheadAttention block runs *per supervoxel* over modality tokens, then
    # mean-pools to a single vector of size ``cross_modal_d_model`` before GCNs.
    modality_feature_dims: Optional[tuple[int, ...]] = None
    use_cross_modal_attention: bool = False
    cross_modal_d_model: int = 64
    cross_modal_num_heads: int = 4
    cross_modal_dropout: float = 0.1
    # Optional modality names (for attention matrix axes in plots); length must
    # match ``modality_feature_dims`` when the latter is set.
    modality_names: tuple[str, ...] = ()

    # Multimodal PRS (polygenic-risk-score) fusion. 0 disables.
    prs_dim: int = 0
    prs_hidden: int = 64
    prs_embed_dim: int = 64
    prs_dropout: float = 0.1
    prs_fusion: Literal["concat", "gated"] = "concat"
    # Which encoder to build (``auto`` = graph if prs_dim==0 else multimodal).
    model_type: Literal["auto", "graph", "multimodal", "genetics_only"] = "auto"


def _modality_dim_tuple(cfg: SiameseConfig) -> tuple[int, ...]:
    if cfg.modality_feature_dims is None:
        return (cfg.in_channels,)
    m = cfg.modality_feature_dims
    if sum(m) != cfg.in_channels:
        raise ValueError(
            f"modality_feature_dims {m} must sum to in_channels={cfg.in_channels}"
        )
    return m


class ModalityCrossAttentionBlock(nn.Module):
    r"""Per-node fusion of modality-specific vectors with self-attention over modalities.

    For each supervoxel (graph node) we form a short sequence
    :math:`\{\mathbf{h}^{(m)}\}_{m=1}^{M}` of length :math:`M` (modalities) after
    linear projection to a shared :math:`d_{\text{model}}` space, apply one
    multi-head self-attention block (residual + layer norm) and a small FFN, then
    **mean-pool** over :math:`m` to obtain a single node descriptor before GCN layers.

    This is the natural PyTorch analogue of *cross-modality* mixing used in
    multi-parametric imaging: attention weights
    :math:`\mathbb{R}^{M \times M}` (averaged over heads) can be visualised *per
    supervoxel* to build a **modality / heritability-relevant atlas** in concert
    with node-wise gradients of the task loss.

    Args:
        dims: per-modality raw feature sizes (``sum(dims) == in_channels`` on ``Data``).
        d_model: hidden width (must be divisible by ``n_heads``).
    """

    def __init__(
        self,
        dims: tuple[int, ...],
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("ModalityCrossAttentionBlock needs at least two modalities")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.dims = dims
        self.n_mod = len(dims)
        self.d_model = d_model
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in dims])
        self.mha = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, return_attn: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """``x`` is ``[N, sum(dims)]``; returns ``[N, d_model]`` (or + attn weights)."""
        parts = torch.split(x, self.dims, dim=-1)
        seq = torch.stack([p(f) for p, f in zip(self.proj, parts)], dim=1)  # [N, M, d]
        out, w = self.mha(
            seq,
            seq,
            seq,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        h = self.norm1(seq + self.dropout(out))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        fused = h.mean(dim=1)  # [N, d_model]
        if return_attn:
            return fused, w
        return fused


class BrainGNNEncoder(nn.Module):
    """Graph encoder: shared by both twins (weight sharing is the 'Siamese' part).

    Inputs
    ------
    x : Tensor [num_nodes_total, in_channels]
    edge_index : LongTensor [2, num_edges_total]
    edge_weight : Tensor [num_edges_total], optional
    batch : LongTensor [num_nodes_total], graph-membership indices

    **Cross-modal (optional).**  When ``cfg.use_cross_modal_attention`` and more
    than one block is listed in ``modality_feature_dims``, a
    :class:`ModalityCrossAttentionBlock` runs *before* the GCN stack.  Use
    `forward(..., return_modality_attn=True)` to obtain per-node :math:`M \times M`
    attention for interpretability (Heritability / modality importance maps on the
    supervoxel graph).

    Output
    ------
    graph_embedding : Tensor [batch_size, graph_embed_dim]
        Where ``graph_embed_dim = hidden_channels * (2 if pooling=='mean+add' else 1)``.
    """

    def __init__(self, cfg: SiameseConfig):
        super().__init__()
        self.cfg = cfg
        self._modality_dims: tuple[int, ...] = _modality_dim_tuple(cfg)
        self.modality_mha: Optional[ModalityCrossAttentionBlock] = None
        gcn_in: int
        if cfg.use_cross_modal_attention and len(self._modality_dims) > 1:
            self.modality_mha = ModalityCrossAttentionBlock(
                self._modality_dims,
                cfg.cross_modal_d_model,
                cfg.cross_modal_num_heads,
                cfg.cross_modal_dropout,
            )
            gcn_in = cfg.cross_modal_d_model
        else:
            gcn_in = cfg.in_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._gcn_in_channels = gcn_in

        dims = [gcn_in] + [cfg.hidden_channels] * cfg.num_layers
        for i in range(cfg.num_layers):
            # add_self_loops=True is the standard GCN formulation: we inject
            # \tilde{A} = A + I so each node also considers its own features.
            # improved=False keeps us in the classical Kipf-Welling regime.
            self.convs.append(
                GCNConv(dims[i], dims[i + 1], add_self_loops=True, improved=False)
            )
            self.norms.append(nn.BatchNorm1d(dims[i + 1]))

        pooled_dim = cfg.hidden_channels * (2 if cfg.pooling == "mean+add" else 1)
        self.graph_embed_dim = pooled_dim

    def _pool(self, x: Tensor, batch: Tensor) -> Tensor:
        if self.cfg.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.cfg.pooling == "add":
            return global_add_pool(x, batch)
        if self.cfg.pooling == "mean+add":
            return torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=-1)
        raise ValueError(f"Unknown pooling: {self.cfg.pooling!r}")

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        return_modality_attn: bool = False,
        return_node_features: bool = False,
    ) -> Union[
        Tensor,
        tuple[Tensor, Tensor],
        tuple[Tensor, Tensor, Tensor],
    ]:
        if batch is None:
            # Single-graph path: all nodes belong to graph 0.
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        mod_attn: Optional[Tensor] = None
        if self.modality_mha is not None:
            if return_modality_attn:
                x, mod_attn = self.modality_mha(x, return_attn=True)  # type: ignore[assignment]
            else:
                x = self.modality_mha(x, return_attn=False)  # type: ignore[assignment]
        elif x.size(1) != self._gcn_in_channels:
            raise ValueError(
                f"Node features have {x.size(1)} channels; encoder expects {self._gcn_in_channels}"
            )

        ew: Optional[Tensor]
        if self.cfg.use_edge_weight and edge_weight is not None:
            # Fisher-z edge weights can be negative. GCNConv's symmetric
            # normalisation D^{-1/2} A D^{-1/2} requires non-negative degrees,
            # so we feed |z| as the weight magnitude. This is the canonical
            # choice in brain-GNN literature (Li et al., BrainGNN 2021); the
            # sign information is lost by GCNConv but could be re-introduced
            # via GATConv or a separate signed-graph branch in future work.
            ew = edge_weight.abs()
        else:
            ew = None
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_weight=ew)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)
        out = self._pool(x, batch)
        if return_node_features and return_modality_attn:
            return out, x, mod_attn  # type: ignore[misc]  # mod_attn set above
        if return_node_features:
            return out, x
        if return_modality_attn:
            return out, mod_attn  # type: ignore[return-value]
        return out


class GeneticEncoder(nn.Module):
    """MLP that maps a polygenic risk score (PRS) vector into a latent space
    comparable with the graph embedding.

    For Project 65 we treat the PRS as a dense low-dimensional summary of
    common-variant genetic liability. The encoder is intentionally small
    (2 hidden layers) because PRS dimensions rarely exceed a few hundred in
    practice and we want to avoid over-parameterising the non-GNN branch.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, prs: Tensor) -> Tensor:
        return self.net(prs)


class GatedFusion(nn.Module):
    """Learned sigmoid gate g in [0, 1]^d that mixes graph and PRS embeddings:

        h = g * z_graph + (1 - g) * z_prs

    Requires ``dim(z_graph) == dim(z_prs)``. More expressive than concat when
    the two modalities carry redundant information (which is our prior here).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, z_graph: Tensor, z_prs: Tensor) -> Tensor:
        g = self.gate(torch.cat([z_graph, z_prs], dim=-1))
        return g * z_graph + (1.0 - g) * z_prs


class ProjectionHead(nn.Module):
    """2-layer MLP on top of the encoder. Used for training only (SimCLR-style)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SiameseBrainNet(nn.Module):
    """Twin Siamese wrapper: encodes Twin A and Twin B with shared weights.

    Inference-time usage:
        z_a = model.encode(data_a)  # [B, graph_embed_dim]  <- use this for analysis
    Training-time usage:
        z_a_proj, z_b_proj = model(data_a, data_b)
        loss = contrastive_loss(z_a_proj, z_b_proj, label)
    """

    def __init__(self, cfg: SiameseConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = BrainGNNEncoder(cfg)
        self.projector = ProjectionHead(
            self.encoder.graph_embed_dim,
            cfg.projection_hidden,
            cfg.projection_dim,
        )

    @staticmethod
    def _extract(data) -> tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
        # edge_attr may be absent; handle gracefully.
        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.view(-1)
        return data.x, data.edge_index, edge_weight, getattr(data, "batch", None)

    def encode(self, data) -> Tensor:
        """Compute the encoder embedding (no projection). For downstream analysis."""
        x, edge_index, edge_weight, batch = self._extract(data)
        z = self.encoder(x, edge_index, edge_weight, batch)
        if self.cfg.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z

    def project(self, z: Tensor) -> Tensor:
        out = self.projector(z)
        if self.cfg.normalize_embeddings:
            out = F.normalize(out, dim=-1, eps=1e-8)
        return out

    def forward(self, data_a, data_b) -> tuple[Tensor, Tensor]:
        """Training forward: returns projected embeddings for both twins."""
        z_a = self.encode(data_a)
        z_b = self.encode(data_b)
        return self.project(z_a), self.project(z_b)


class GeneticsOnlySiameseNet(nn.Module):
    """PRS-only Siamese path for ablation: no GNN, only ``GeneticEncoder`` + head.

    Expects each ``Data`` to have ``prs`` of shape ``[B, prs_dim]`` (batched) or
    ``[1, prs_dim]`` (single graph). The graph structure in ``x`` / ``edge_index``
    is ignored—use the same preprocessed files as the multimodal run so the
    ablation is fair except for the encoder.
    """

    def __init__(self, cfg: SiameseConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.prs_dim <= 0:
            raise ValueError("GeneticsOnlySiameseNet requires cfg.prs_dim > 0")
        self.prs_encoder = GeneticEncoder(
            in_dim=cfg.prs_dim,
            hidden_dim=cfg.prs_hidden,
            out_dim=cfg.prs_embed_dim,
            dropout=cfg.prs_dropout,
        )
        self.projector = ProjectionHead(
            cfg.prs_embed_dim, cfg.projection_hidden, cfg.projection_dim
        )

    def encode(self, data) -> Tensor:
        prs = getattr(data, "prs", None)
        if prs is None:
            raise ValueError(
                "GeneticsOnlySiameseNet requires data.prs. Generate cohorts with --prs-dim > 0."
            )
        z = self.prs_encoder(prs)
        if self.cfg.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z

    def project(self, z: Tensor) -> Tensor:
        out = self.projector(z)
        if self.cfg.normalize_embeddings:
            out = F.normalize(out, dim=-1, eps=1e-8)
        return out

    def forward(self, data_a, data_b) -> tuple[Tensor, Tensor]:
        return self.project(self.encode(data_a)), self.project(self.encode(data_b))


def _resolve_siamese_model_type(cfg: SiameseConfig) -> str:
    if cfg.model_type == "auto":
        return "graph" if cfg.prs_dim <= 0 else "multimodal"
    return cfg.model_type


class MultimodalSiameseBrainNet(SiameseBrainNet):
    """Fuses the graph embedding with a polygenic risk score (PRS) embedding.

    Forward-path topology:

        data.x, data.edge_index --[GNN encoder]--> z_graph  (R^G)
              data.prs          --[GeneticEncoder]--> z_prs (R^P)
                   |__________ fuse (concat or gated) __________|
                                         v
                                      z_fused  (R^F)
                                         |
                                         v
                                   ProjectionHead  (R^D)

    The encoder-level embedding returned by :meth:`encode` is the fused
    representation (pre-projection), so the HeritabilityAuxLoss operates on
    the multimodal manifold - exactly what we want for grant validation.
    """

    def __init__(self, cfg: SiameseConfig):
        if cfg.prs_dim <= 0:
            raise ValueError(
                "MultimodalSiameseBrainNet requires cfg.prs_dim > 0; "
                "use SiameseBrainNet for graph-only."
            )
        super().__init__(cfg)
        # Rebuild projector so it operates on the fused dimension.
        graph_dim = self.encoder.graph_embed_dim
        prs_dim = cfg.prs_embed_dim

        self.prs_encoder = GeneticEncoder(
            in_dim=cfg.prs_dim,
            hidden_dim=cfg.prs_hidden,
            out_dim=prs_dim,
            dropout=cfg.prs_dropout,
        )

        if cfg.prs_fusion == "concat":
            fused_dim = graph_dim + prs_dim
            self.fusion: Optional[nn.Module] = None
        elif cfg.prs_fusion == "gated":
            if graph_dim != prs_dim:
                raise ValueError(
                    f"Gated fusion requires matching dims (got graph={graph_dim}, "
                    f"prs={prs_dim}); set hidden_channels and prs_embed_dim equal "
                    f"or use concat fusion."
                )
            fused_dim = graph_dim
            self.fusion = GatedFusion(fused_dim)
        else:
            raise ValueError(f"Unknown prs_fusion: {cfg.prs_fusion}")

        self.fused_dim = fused_dim
        self.projector = ProjectionHead(
            fused_dim, cfg.projection_hidden, cfg.projection_dim
        )

    def encode(self, data) -> Tensor:  # type: ignore[override]
        x, edge_index, edge_weight, batch = self._extract(data)
        z_graph = self.encoder(x, edge_index, edge_weight, batch)
        prs = getattr(data, "prs", None)
        if prs is None:
            raise ValueError(
                "MultimodalSiameseBrainNet.encode: data.prs is missing. "
                "Ensure the synthetic generator was run with --prs-dim > 0."
            )
        # PyG's Batch stacks prs tensors of shape [1, d] into [B, d].
        # When called with a single Data (no batch), prs is [1, d] already.
        z_prs = self.prs_encoder(prs)
        if self.fusion is None:  # concat
            z = torch.cat([z_graph, z_prs], dim=-1)
        else:
            z = self.fusion(z_graph, z_prs)
        if self.cfg.normalize_embeddings:
            z = F.normalize(z, dim=-1, eps=1e-8)
        return z


def build_siamese_model(cfg: SiameseConfig) -> nn.Module:
    """Factory: graph, multimodal (GNN+PRS), or genetics-only (PRS ablation)."""
    kind = _resolve_siamese_model_type(cfg)
    if kind == "genetics_only":
        return GeneticsOnlySiameseNet(cfg)
    if kind == "multimodal":
        if cfg.prs_dim <= 0:
            raise ValueError("model_type=multimodal requires prs_dim > 0")
        return MultimodalSiameseBrainNet(cfg)
    # kind == "graph" — graph encoder only; PRS in files is ignored
    return SiameseBrainNet(cfg)


class ContrastiveLoss(nn.Module):
    """Hadsell-et-al 2006 contrastive loss for Siamese twin embeddings.

        L(z_a, z_b, y) =  (1 - y) * 0.5 * D^2
                        +   y     * 0.5 * max(0, margin - D)^2

    Convention matching our manifest labels:
        y = 0  -> positive pair (MZ)           -> minimise distance.
        y = 1  -> negative pair (DZ / UNREL)   -> push apart beyond margin.

    Distance metric is configurable:
        'euclidean' : sqrt(sum((z_a - z_b)^2 + eps))
        'cosine'    : 1 - <z_a, z_b>  (inputs assumed L2-normalised)

    The cosine flavour is numerically well-behaved on MPS and scale-invariant.
    """

    def __init__(
        self,
        margin: float = 1.0,
        metric: Literal["euclidean", "cosine"] = "cosine",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.margin = margin
        self.metric = metric
        self.eps = eps

    def distance(self, z_a: Tensor, z_b: Tensor) -> Tensor:
        if self.metric == "euclidean":
            # sqrt of sum-of-squares with epsilon for grad stability at 0.
            return torch.sqrt(torch.clamp(((z_a - z_b) ** 2).sum(dim=-1), min=self.eps))
        if self.metric == "cosine":
            return 1.0 - (z_a * z_b).sum(dim=-1)
        raise ValueError(f"Unknown metric: {self.metric!r}")

    def forward(self, z_a: Tensor, z_b: Tensor, label: Tensor) -> Tensor:
        d = self.distance(z_a, z_b)
        label = label.float().view(-1)
        positive = (1.0 - label) * 0.5 * d.pow(2)
        negative = label * 0.5 * torch.clamp(self.margin - d, min=0.0).pow(2)
        return (positive + negative).mean()


class HeritabilityAuxLoss(nn.Module):
    r"""Mean-squared error between the batch-level \hat{h}^2 and a known
    ground-truth target. Only defined on synthetic cohorts where h^2 is known.

    The differentiable h^2 estimate is

        \hat{h}^2 = 2 * (mean(cos(z_MZ_a, z_MZ_b)) - mean(cos(z_DZ_a, z_DZ_b)))

    If a mini-batch lacks either MZ or DZ pairs we emit 0 (and a flag) so the
    caller can either skip or weight the aux term accordingly.
    """

    def __init__(self, target_h2: float, clamp: bool = True):
        super().__init__()
        # Use a buffer so it auto-moves with .to(device) and lives in state_dict.
        self.register_buffer("target_h2", torch.tensor(float(target_h2)))
        self.clamp = clamp

    def forward(
        self,
        z_a: Tensor,
        z_b: Tensor,
        zygosities: list[str],
    ) -> tuple[Tensor, bool]:
        """Returns (loss, is_valid). ``is_valid`` is False when the batch
        cannot produce a well-defined heritability estimate."""
        is_mz = torch.tensor(
            [z == "MZ" for z in zygosities], device=z_a.device, dtype=torch.bool
        )
        is_dz = torch.tensor(
            [z == "DZ" for z in zygosities], device=z_a.device, dtype=torch.bool
        )
        if is_mz.sum() == 0 or is_dz.sum() == 0:
            return z_a.new_zeros(()), False

        za = F.normalize(z_a, dim=-1, eps=1e-8)
        zb = F.normalize(z_b, dim=-1, eps=1e-8)
        sims = (za * zb).sum(dim=-1)
        r_mz = sims[is_mz].mean()
        r_dz = sims[is_dz].mean()
        h2 = 2.0 * (r_mz - r_dz)
        if self.clamp:
            h2 = h2.clamp(0.0, 1.0)
        return (h2 - self.target_h2).pow(2), True


@dataclass
class TwinBatch:
    """Type-safe container for a twin-pair minibatch coming out of the DataLoader."""

    data_a: object  # torch_geometric.data.Batch
    data_b: object  # torch_geometric.data.Batch
    label: Tensor
    family_ids: list[str] = field(default_factory=list)
    zygosities: list[str] = field(default_factory=list)

    def to(self, device: torch.device) -> "TwinBatch":
        return TwinBatch(
            data_a=self.data_a.to(device),
            data_b=self.data_b.to(device),
            label=self.label.to(device),
            family_ids=self.family_ids,
            zygosities=self.zygosities,
        )


# Back-compat: keep the original class name around, now aliased to the
# hardened encoder so existing imports don't break.
BrainGNN = BrainGNNEncoder

__all__ = [
    "BrainGNN",
    "BrainGNNEncoder",
    "ModalityCrossAttentionBlock",
    "ContrastiveLoss",
    "GatedFusion",
    "GeneticEncoder",
    "GeneticsOnlySiameseNet",
    "HeritabilityAuxLoss",
    "MultimodalSiameseBrainNet",
    "ProjectionHead",
    "SiameseBrainNet",
    "SiameseConfig",
    "TwinBatch",
    "build_siamese_model",
]
