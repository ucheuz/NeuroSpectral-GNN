"""Gradient-based saliency on supervoxel node features (KCL P65 / BrainGNN-style).

Operates on **node** inputs ``data.x`` with autograd. Use outputs with
``map_nodes_to_volume`` to build a **Heritability / saliency atlas** in MNI
voxel space (per supervoxel scalar → repainted volume).

On **Apple Silicon**, pass ``device=torch.device('mps')``; unsupported ops may
fall back to CPU via ``PYTORCH_ENABLE_MPS_FALLBACK``.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from src.models.siamese_gnn import ContrastiveLoss, SiameseBrainNet


def _replace_batch_x(data, x_new: Tensor):
    """Clone PyG Data/Batch with new x (keeps edge_index, batch, etc.)."""
    out = data.clone()
    out.x = x_new
    return out


def gradient_saliency_contrastive_pair(
    model: SiameseBrainNet,
    data_a,
    data_b,
    label: Tensor,
    device: torch.device,
    *,
    loss_fn: Optional[ContrastiveLoss] = None,
    twin: Literal["a", "b"] = "a",
) -> np.ndarray:
    """|∂L/∂x| aggregated over feature columns → one saliency scalar per supervoxel.

    L is the **contrastive** loss on *projected* embeddings (same objective as
    training). This highlights supervoxels whose node features most influence
    twin-pair discrimination (MZ proximity vs DZ separation) under the current
    weights.

    Parameters
    ----------
    twin
        Which graph's node features receive the gradient (``'a'`` or ``'b'``).

    Returns
    -------
    numpy.ndarray
        Shape ``[num_nodes]`` (same node order as ``data_*.x`` in the object).
    """
    if loss_fn is None:
        loss_fn = ContrastiveLoss(margin=1.0, metric="cosine")

    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    data_a = data_a.to(device)
    data_b = data_b.to(device)
    label = label.to(device)

    da = data_a.clone()
    db = data_b.clone()
    if twin == "a":
        x = da.x.detach().clone().float().requires_grad_(True)
        da = _replace_batch_x(da, x)
        z_a = model.encode(da)
        z_b = model.encode(db)
    else:
        x = db.x.detach().clone().float().requires_grad_(True)
        db = _replace_batch_x(db, x)
        z_a = model.encode(da)
        z_b = model.encode(db)

    z_ap = model.project(z_a)
    z_bp = model.project(z_b)
    lab = label.to(z_ap.device)
    if lab.dim() == 0:
        lab = lab.view(1)
    loss = loss_fn(z_ap, z_bp, lab)
    loss.backward()

    g = x.grad
    if g is None:
        return np.zeros(x.size(0), dtype=np.float32)
    sal = g.abs().sum(dim=-1).detach().cpu().numpy().astype(np.float32)
    return sal


def integrated_gradients_contrastive_pair(
    model: SiameseBrainNet,
    data_a,
    data_b,
    label: Tensor,
    device: torch.device,
    *,
    twin: Literal["a", "b"] = "a",
    n_steps: int = 16,
    loss_fn: Optional[ContrastiveLoss] = None,
) -> np.ndarray:
    """Integrated gradients of the contrastive loss w.r.t. ``data_a.x`` or ``data_b.x``.

    Baseline is **zero** (same shape as ``x``). More stable than raw gradients for
    attribution across scales; costs ``n_steps`` forward+backward passes.
    """
    if loss_fn is None:
        loss_fn = ContrastiveLoss(margin=1.0, metric="cosine")

    model.to(device)
    model.eval()
    data_a = data_a.to(device)
    data_b = data_b.to(device)
    if twin == "a":
        x_full = data_a.x.detach().float()
    else:
        x_full = data_b.x.detach().float()

    baseline = torch.zeros_like(x_full)
    diff = x_full - baseline
    integral = torch.zeros_like(x_full)

    for k in range(1, n_steps + 1):
        t = k / n_steps
        x = (baseline + t * diff).clone().requires_grad_(True)
        if twin == "a":
            da = _replace_batch_x(data_a, x)
            z_a = model.encode(da)
            z_b = model.encode(data_b)
        else:
            db = _replace_batch_x(data_b, x)
            z_a = model.encode(data_a)
            z_b = model.encode(db)
        z_ap = model.project(z_a)
        z_bp = model.project(z_b)
        lab = label.to(z_ap.device)
        if lab.dim() == 0:
            lab = lab.view(1)
        loss = loss_fn(z_ap, z_bp, lab)
        g = torch.autograd.grad(loss, x, retain_graph=False)[0]
        if g is not None:
            integral = integral + g.detach()

    ig = (diff * integral) / float(n_steps)
    sal = ig.abs().sum(dim=-1).cpu().numpy().astype(np.float32)
    return sal
