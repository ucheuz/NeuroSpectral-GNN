"""PyTorch Dataset + collator for preprocessed twin brain graphs.

Expected on-disk layout (produced by ``scripts/preprocess_twins.py`` or
``scripts/generate_synthetic_twins.py``):

    {root}/subjects/{subject_id}.pt   -> torch_geometric.data.Data
    {root}/pairs.csv                  -> columns: family_id, subject_a,
                                                   subject_b, zygosity, label

Why a custom ``Dataset`` + collator (not PyG's ``Dataset`` class)?
    We need each __getitem__ to return *two* graphs + a scalar label. PyG's
    built-in Dataset assumes a single graph per index. Implementing a
    ``collate_fn`` that calls ``Batch.from_data_list`` on each half of the
    pair is explicit, efficient, and type-safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.models.siamese_gnn import TwinBatch

logger = logging.getLogger(__name__)


class TwinBrainDataset(Dataset):
    """Serves (Twin A graph, Twin B graph, label, metadata) tuples.

    Parameters
    ----------
    root : str or Path
        Directory containing ``subjects/`` and ``pairs.csv``.
    pairs_csv : str, optional
        Override for the pairs file name (relative to root).
    subject_subdir : str, optional
        Subdirectory inside ``root`` containing ``{subject_id}.pt`` files.
    include_zygosities : iterable of str, optional
        If provided, restrict to pairs whose zygosity is in this set
        (e.g. ``{'MZ', 'DZ'}`` to exclude UNREL controls during training).
    preload : bool
        If True, load all .pt files into RAM at construction time. Fine for
        Schaefer-100 on a few hundred subjects (100x100 float32 = ~40 KB each).
    """

    def __init__(
        self,
        root: str | Path,
        pairs_csv: str = "pairs.csv",
        subject_subdir: str = "subjects",
        include_zygosities: Optional[set[str]] = None,
        preload: bool = True,
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.subjects_dir = self.root / subject_subdir
        pairs_path = self.root / pairs_csv
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")
        if not self.subjects_dir.is_dir():
            raise FileNotFoundError(f"Subjects dir not found: {self.subjects_dir}")

        df = pd.read_csv(pairs_path)
        if include_zygosities is not None:
            df = df[df["zygosity"].isin(include_zygosities)].reset_index(drop=True)
        self.pairs = df.to_dict(orient="records")

        self._cache: dict[str, Data] = {}
        if preload:
            logger.info("Preloading %d pairs into RAM", len(self.pairs))
            for record in self.pairs:
                for sid in (record["subject_a"], record["subject_b"]):
                    if sid not in self._cache:
                        self._cache[sid] = self._load_subject(sid)

        logger.info(
            "TwinBrainDataset initialised: root=%s, n_pairs=%d, preload=%s",
            self.root, len(self.pairs), preload,
        )

    def _load_subject(self, subject_id: str) -> Data:
        path = self.subjects_dir / f"{subject_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Subject file missing: {path}")
        # weights_only=False because our Data objects carry extra attrs
        # (connectivity tensor, string metadata) that torch.load refuses
        # to deserialise under the stricter default.
        return torch.load(path, weights_only=False, map_location="cpu")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Data, Data, dict]:
        rec = self.pairs[idx]
        a_id, b_id = rec["subject_a"], rec["subject_b"]
        if self._cache:
            data_a = self._cache[a_id]
            data_b = self._cache[b_id]
        else:
            data_a = self._load_subject(a_id)
            data_b = self._load_subject(b_id)

        meta = {
            "family_id": rec["family_id"],
            "zygosity": rec["zygosity"],
            "label": int(rec["label"]),
            "subject_a": a_id,
            "subject_b": b_id,
        }
        return data_a, data_b, meta


def twin_collate(samples: list[tuple[Data, Data, dict]]) -> TwinBatch:
    """Custom collator: batch Twin A graphs and Twin B graphs separately.

    This is what makes the Siamese DataLoader work. PyG's ``Batch.from_data_list``
    packs a list of Data objects into a single Batch with a contiguous
    ``batch`` vector - essential for ``global_mean_pool`` to know which nodes
    belong to which graph.
    """
    data_a_list = [s[0] for s in samples]
    data_b_list = [s[1] for s in samples]
    metas = [s[2] for s in samples]

    batch_a = Batch.from_data_list(data_a_list)
    batch_b = Batch.from_data_list(data_b_list)
    labels = torch.tensor([m["label"] for m in metas], dtype=torch.float32)

    return TwinBatch(
        data_a=batch_a,
        data_b=batch_b,
        label=labels,
        family_ids=[m["family_id"] for m in metas],
        zygosities=[m["zygosity"] for m in metas],
    )
