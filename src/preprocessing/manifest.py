"""Manifest parsing: turn a CSV of subjects into strongly-typed records and
twin-pair tuples.

Expected manifest schema (header row required)
----------------------------------------------
subject_id   : str      unique subject identifier
nii_path     : str      absolute or relative path to 4D fMRI NIfTI
confounds_path : str    optional path to nuisance regressors TSV/CSV (empty = none)
family_id    : str      shared within twin pairs; unique across pairs
twin_id      : str      'A' or 'B' (or '1'/'2') - role within the family
zygosity     : str      one of {MZ, DZ, UNREL}
t_r          : float    optional, repetition time in seconds (overrides global)

Zygosity -> label convention for the contrastive loss:
    MZ   -> 0   ('positive'  / similar pair - should embed close together)
    DZ   -> 1   ('negative'  / dissimilar    - should embed farther apart)
    UNREL-> 1   (unrelated controls act as additional negatives)

We keep this mapping explicit in a lookup so the dissertation can cite it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ZYGOSITY_TO_LABEL: dict[str, int] = {
    "MZ": 0,
    "DZ": 1,
    "UNREL": 1,
}

REQUIRED_COLUMNS = {"subject_id", "nii_path", "family_id", "twin_id", "zygosity"}
OPTIONAL_COLUMNS = {"confounds_path", "t_r"}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    nii_path: Path
    family_id: str
    twin_id: str
    zygosity: str
    confounds_path: Optional[Path] = None
    t_r: Optional[float] = None


@dataclass(frozen=True)
class TwinPair:
    family_id: str
    subject_a: str
    subject_b: str
    zygosity: str  # 'MZ' | 'DZ' | 'UNREL'
    label: int  # 0 for MZ, 1 otherwise
    meta: dict = field(default_factory=dict)


def load_manifest(manifest_csv: str | Path) -> list[SubjectRecord]:
    """Load and validate a subject manifest CSV.

    Raises
    ------
    ValueError
        If required columns are missing or zygosity values are invalid.
    """
    manifest_csv = Path(manifest_csv).expanduser()
    df = pd.read_csv(manifest_csv)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest {manifest_csv} missing required columns: {sorted(missing)}"
        )

    df["zygosity"] = df["zygosity"].astype(str).str.upper().str.strip()
    bad_zyg = set(df["zygosity"]) - set(ZYGOSITY_TO_LABEL)
    if bad_zyg:
        raise ValueError(
            f"Unknown zygosity values: {sorted(bad_zyg)}. "
            f"Valid values: {sorted(ZYGOSITY_TO_LABEL)}"
        )

    records: list[SubjectRecord] = []
    for _, row in df.iterrows():
        confounds = row.get("confounds_path")
        confounds_path: Optional[Path] = None
        if isinstance(confounds, str) and confounds.strip():
            confounds_path = Path(confounds).expanduser()

        t_r = row.get("t_r")
        if isinstance(t_r, str) and not t_r.strip():
            t_r = None
        t_r_val = float(t_r) if t_r is not None and not pd.isna(t_r) else None

        records.append(
            SubjectRecord(
                subject_id=str(row["subject_id"]),
                nii_path=Path(str(row["nii_path"])).expanduser(),
                family_id=str(row["family_id"]),
                twin_id=str(row["twin_id"]),
                zygosity=str(row["zygosity"]),
                confounds_path=confounds_path,
                t_r=t_r_val,
            )
        )

    logger.info("Loaded %d subject records from %s", len(records), manifest_csv)
    return records


def build_twin_pairs(
    records: Iterable[SubjectRecord],
    include_unrelated: bool = False,
    unrelated_per_subject: int = 1,
    rng_seed: int = 42,
) -> list[TwinPair]:
    """Group records by family_id into twin pairs.

    Parameters
    ----------
    records : iterable of SubjectRecord
    include_unrelated : bool
        If True, additionally sample ``unrelated_per_subject`` unrelated pairs
        for each subject as hard negatives for the contrastive loss.
    unrelated_per_subject : int
        Number of UNREL pairs to sample per subject.
    rng_seed : int

    Returns
    -------
    list[TwinPair]
    """
    import random

    by_family: dict[str, list[SubjectRecord]] = {}
    for r in records:
        by_family.setdefault(r.family_id, []).append(r)

    pairs: list[TwinPair] = []
    for fid, members in by_family.items():
        if len(members) < 2:
            logger.warning("Family %s has <2 members; skipping.", fid)
            continue
        if len(members) > 2:
            logger.warning(
                "Family %s has %d members (>2). Using first two sorted by twin_id.",
                fid,
                len(members),
            )
        a, b = sorted(members, key=lambda r: r.twin_id)[:2]
        if a.zygosity != b.zygosity:
            logger.warning(
                "Family %s zygosity mismatch between twins (%s vs %s); using A's.",
                fid,
                a.zygosity,
                b.zygosity,
            )
        zyg = a.zygosity
        pairs.append(
            TwinPair(
                family_id=fid,
                subject_a=a.subject_id,
                subject_b=b.subject_id,
                zygosity=zyg,
                label=ZYGOSITY_TO_LABEL[zyg],
            )
        )

    if include_unrelated:
        rng = random.Random(rng_seed)
        all_records = list(records)
        subject_ids = [r.subject_id for r in all_records]
        family_of = {r.subject_id: r.family_id for r in all_records}
        for subj in subject_ids:
            for _ in range(unrelated_per_subject):
                # Sample until we get a subject from a different family.
                for _attempt in range(20):
                    candidate = rng.choice(subject_ids)
                    if family_of[candidate] != family_of[subj]:
                        pairs.append(
                            TwinPair(
                                family_id=f"UNREL_{subj}_{candidate}",
                                subject_a=subj,
                                subject_b=candidate,
                                zygosity="UNREL",
                                label=ZYGOSITY_TO_LABEL["UNREL"],
                            )
                        )
                        break

    logger.info(
        "Built %d twin pairs (MZ=%d, DZ=%d, UNREL=%d)",
        len(pairs),
        sum(1 for p in pairs if p.zygosity == "MZ"),
        sum(1 for p in pairs if p.zygosity == "DZ"),
        sum(1 for p in pairs if p.zygosity == "UNREL"),
    )
    return pairs


def pairs_to_dataframe(pairs: list[TwinPair]) -> pd.DataFrame:
    """Export TwinPair list to a DataFrame for CSV writing."""
    return pd.DataFrame(
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
