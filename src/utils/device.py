"""Device selection helper optimized for Apple Silicon (MPS) with CPU fallback."""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def get_device(prefer: str = "auto", verbose: bool = True) -> torch.device:
    """Return the best available torch device.

    Priority order for ``prefer='auto'``:
        1. CUDA (if available) - for HPC / cloud runs.
        2. MPS (Apple Silicon unified memory) - our primary laptop target.
        3. CPU - always available fallback.

    Parameters
    ----------
    prefer : {"auto", "cuda", "mps", "cpu"}
        Override automatic selection. ``'auto'`` picks the best available.
    verbose : bool
        Log which device was selected.

    Notes
    -----
    On MPS, we also enable the PYTORCH_ENABLE_MPS_FALLBACK flag so that any op
    not yet implemented on MPS transparently falls back to CPU rather than
    crashing. This matters for ops like ``torch.linalg.eigh`` used in
    spectral analysis. Setting this env var has no effect if it's already set
    or if we're not on MPS.
    """
    prefer = prefer.lower()

    if prefer == "cuda" or (prefer == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    elif prefer == "mps" or (
        prefer == "auto"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        logger.info("Selected torch device: %s", device)
    return device
