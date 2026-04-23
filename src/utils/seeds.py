"""Reproducibility helpers: global seed setting across numpy, python, torch."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA + MPS) for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value.
    deterministic : bool
        If True, enable PyTorch deterministic algorithms. Slower but guarantees
        bit-exact reproducibility where supported. Many GNN ops (scatter_add)
        are non-deterministic on GPU; keep this False for training and True
        only for scientific reproduction runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            logger.warning("Could not enable full determinism: %s", exc)

    logger.info("Global seed set to %d (deterministic=%s)", seed, deterministic)
