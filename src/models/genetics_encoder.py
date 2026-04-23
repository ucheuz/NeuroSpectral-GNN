"""MLP encoder for polygenic risk scores (PRS) and related genetic feature vectors.

Handles **variable input dimension** by fixing ``in_dim`` at construction
(different studies / PRS panels map to different *K*; rebuild the module when
``K`` changes). Uses BatchNorm1d and Dropout on sparse, heterogeneous PRS inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class GeneticsEncoderConfig:
    """Configuration for :class:`GeneticsEncoder`."""

    in_dim: int
    out_dim: int
    hidden_dim: int = 64
    num_hidden_blocks: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.in_dim < 1:
            raise ValueError("in_dim must be >= 1")
        if self.out_dim < 1:
            raise ValueError("out_dim must be >= 1")
        if self.num_hidden_blocks < 1:
            raise ValueError("num_hidden_blocks must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


def _mlp_from_config(cfg: GeneticsEncoderConfig) -> nn.Sequential:
    layers: List[nn.Module] = []
    h = cfg.hidden_dim
    layers += [
        nn.Linear(cfg.in_dim, h),
        nn.BatchNorm1d(h),
        nn.ReLU(inplace=True),
        nn.Dropout(p=cfg.dropout),
    ]
    for _ in range(1, cfg.num_hidden_blocks):
        layers += [
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg.dropout),
        ]
    layers.append(nn.Linear(h, cfg.out_dim))
    return nn.Sequential(*layers)


class GeneticsEncoder(nn.Module):
    """MLP: ``in_dim → (Linear+BN+ReLU+Dropout) × num_hidden_blocks → out_dim`` for PRS.

    The first block maps ``in_dim → hidden_dim``; any further blocks map
    ``hidden_dim → hidden_dim``. The final **Linear** maps to ``out_dim`` (no
    norm after, standard embedding). **MPS-compatible.**
    """

    def __init__(self, cfg: GeneticsEncoderConfig | None = None, **kwargs):
        super().__init__()
        if cfg is not None and kwargs:
            raise ValueError("Pass either GeneticsEncoderConfig or keyword args, not both.")
        if cfg is None:
            cfg = GeneticsEncoderConfig(**kwargs)  # type: ignore[assignment]
        self.cfg = cfg
        self._net = _mlp_from_config(cfg)

    @classmethod
    def from_dims(
        cls,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_hidden_blocks: int = 2,
        dropout: float = 0.1,
    ) -> "GeneticsEncoder":
        return cls(
            GeneticsEncoderConfig(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=hidden_dim,
                num_hidden_blocks=num_hidden_blocks,
                dropout=dropout,
            )
        )

    def forward(self, prs: Tensor) -> Tensor:
        if prs.dim() != 2:
            raise ValueError(f"PRS must be (batch, in_dim), got {prs.shape!r}")
        if prs.size(1) != self.cfg.in_dim:
            raise ValueError(
                f"Expected in_dim={self.cfg.in_dim}, got feature dim {prs.size(1)}"
            )
        return self._net(prs)
