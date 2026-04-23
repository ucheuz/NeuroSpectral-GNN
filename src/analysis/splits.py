"""Family-stratified K-fold cross-validation for twin studies.

Key property: both twins of every family land in the same fold. This prevents
the trivial leak where Twin A sits in train and Twin B in validation - which
would let the GNN memorise each family's genetic component and fake high
heritability recovery.

    Secondary property: zygosity is stratified across folds so every fold has
    a mixture of zygosities. When UNREL pairs exist, they are distributed
    across folds the same way (Falconer still uses only MZ and DZ in validation).
"""

from __future__ import annotations

import logging
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def family_stratified_kfold(
    pairs_df: pd.DataFrame,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) index arrays into ``pairs_df``.

    Each index refers to a ROW of ``pairs_df`` (i.e. one twin pair). Because
    rows are pairs (not individual subjects), family-level leakage is already
    precluded if ``family_id`` is one-to-one with pairs - which is true for
    our manifest convention. We stratify by zygosity (MZ, DZ, UNREL, …) so
    each stratum is spread across folds.

    Parameters
    ----------
    pairs_df : DataFrame
        Must contain columns ``family_id`` and ``zygosity``.
    n_splits : int
        Number of folds.
    shuffle : bool
    seed : int

    Notes
    -----
    We use a simple "round-robin within each zygosity bucket" to guarantee
    zygosity balance. This is simpler than ``StratifiedGroupKFold`` and
    avoids an sklearn version dependency.
    """
    required = {"family_id", "zygosity"}
    if not required.issubset(pairs_df.columns):
        raise ValueError(
            f"pairs_df missing required columns; needs {required}, got {set(pairs_df.columns)}"
        )

    rng = np.random.default_rng(seed)

    fold_assignments = np.full(len(pairs_df), -1, dtype=int)
    for zyg, group in pairs_df.groupby("zygosity"):
        idx = group.index.to_numpy()
        if shuffle:
            rng.shuffle(idx)
        # Round-robin assignment ensures each fold has ~len(group)/n_splits pairs
        for i, row_idx in enumerate(idx):
            fold_assignments[row_idx] = i % n_splits

    for fold in range(n_splits):
        val_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]
        if len(val_idx) == 0 or len(train_idx) == 0:
            logger.warning("Fold %d is empty; skipping.", fold)
            continue
        yield train_idx, val_idx


def summarise_splits(
    pairs_df: pd.DataFrame,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Return a per-fold summary DataFrame for logging / sanity checking."""
    rows = []
    for i, (tr, va) in enumerate(splits):
        tr_df = pairs_df.iloc[tr]
        va_df = pairs_df.iloc[va]
        rows.append({
            "fold": i,
            "n_train": len(tr_df),
            "n_val": len(va_df),
            "train_mz": int((tr_df["zygosity"] == "MZ").sum()),
            "train_dz": int((tr_df["zygosity"] == "DZ").sum()),
            "train_unrel": int((tr_df["zygosity"] == "UNREL").sum()),
            "val_mz": int((va_df["zygosity"] == "MZ").sum()),
            "val_dz": int((va_df["zygosity"] == "DZ").sum()),
            "val_unrel": int((va_df["zygosity"] == "UNREL").sum()),
        })
    return pd.DataFrame(rows)
