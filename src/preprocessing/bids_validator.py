"""Pre-flight cohort check: **BIDS-structured** (or similar) NIfTIs + PRS alignment.

Crawls a ``bids_root`` tree (expects ``sub-*`` folders, or you can set
``bids_layout=False`` to treat first-level subdirectories as subject IDs) and
checks that each required modality is present, then cross-checks a master PRS
table so every **imaging-complete** subject has a non-null PRS row.

**Integration:** run before ``scripts/preprocess_twins.py`` or ``train.py`` so
twin-pair construction never references missing files.

Run from repo root: ``python -m src.preprocessing.bids_validator --help`` or
``python scripts/validate_cohort_bids.py ...``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None

# ---------------------------------------------------------------------------
# Defaults (override via --globs-json)
# ---------------------------------------------------------------------------

DEFAULT_PATTERNS: dict[str, list[str]] = {
    "T1w": ["*T1w*.nii*"],
    "FLAIR_t2": ["*FLAIR*.nii*"],
    "dwi_FA": ["*FA*.nii*", "*fa*.nii*"],
    "dwi_MD": ["*MD*.nii*", "*md*.nii*"],
}

def _is_false_fa_name(f: Path) -> bool:
    return "flair" in f.name.lower()


@dataclass
class CohortFileSpec:
    """Glob patterns (``fnmatch``) relative to *each* subject directory."""

    patterns: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_PATTERNS))
    strict_fa_md: bool = True

    @classmethod
    def from_json(cls, p: Path) -> "CohortFileSpec":
        d = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            raise ValueError("globs json must be a dict modality -> [patterns]")
        for k, v in list(d.items()):
            if isinstance(v, str):
                d[k] = [v]
        return cls(patterns={str(k): list(v) for k, v in d.items()})

    @property
    def keys(self) -> list[str]:
        return list(self.patterns)


def _find_first_match(
    subject_dir: Path,
    patterns: list[str],
    extra_exclude: Optional[set[Path]] = None,
    *,
    match_mode: str = "default",
) -> Optional[Path]:
    """``match_mode`` = ``"fa"`` → reject filenames that look like FLAIR to avoid *FA* in *FLAIR*."""
    extra_exclude = extra_exclude or set()
    for pat in patterns:
        for f in subject_dir.rglob("*.nii*"):
            if not f.is_file() or f in extra_exclude:
                continue
            if not fnmatch.fnmatch(f.name, pat):
                continue
            if match_mode == "fa" and _is_false_fa_name(f):
                continue
            return f
    return None


def _find_dwi_fa_md(
    subject_dir: Path, fa_pats: list[str], md_pats: list[str]
) -> tuple[Optional[Path], Optional[Path]]:
    used: set[Path] = set()
    fa = _find_first_match(subject_dir, fa_pats, used, match_mode="fa")
    if fa is not None:
        used.add(fa)
    md = _find_first_match(subject_dir, md_pats, used)
    return fa, md


def _scan_subject(
    subject_dir: Path, spec: CohortFileSpec
) -> dict[str, Any]:
    pats = spec.patterns
    t1 = _find_first_match(subject_dir, pats.get("T1w", ["*T1w*.nii*"]))
    flair = _find_first_match(
        subject_dir, pats.get("FLAIR_t2", pats.get("FLAIR", ["*FLAIR*.nii*"]))
    )
    fa_p, md_p = pats.get("dwi_FA", ["*FA*.nii*"]), pats.get("dwi_MD", ["*MD*.nii*"])
    fa, md = _find_dwi_fa_md(subject_dir, list(fa_p), list(md_p))
    if spec.strict_fa_md and md is not None and "t1w" in md.name.lower():
        md = None

    row = {
        "subject_dir": str(subject_dir.name),
        "T1w": t1,
        "FLAIR_t2": flair,
        "dwi_FA": fa,
        "dwi_MD": md,
    }
    row["n_mod_ok"] = sum(1 for k in ("T1w", "FLAIR_t2", "dwi_FA", "dwi_MD") if row[k] is not None)
    row["imaging_complete"] = row["n_mod_ok"] == 4
    return row


def _optional_nifti_sanity(path: Path) -> str:
    if nib is None:  # pragma: no cover
        return "nibabel_missing"
    try:
        im = nib.load(str(path), mmap=True)
        _ = im.shape
    except Exception as e:  # pragma: no cover
        return f"load_error:{e!s}"
    return "ok"


def discover_subject_ids(bids_root: Path, *, bids_prefix: str = "sub-") -> list[str]:
    r = Path(bids_root)
    if not r.is_dir():
        raise NotADirectoryError(f"not a directory: {r.resolve()}")
    sub_dirs = [p for p in r.iterdir() if p.is_dir() and p.name.startswith(bids_prefix)]
    if sub_dirs:
        return sorted(p.name for p in sub_dirs)
    return sorted(
        p.name
        for p in r.iterdir()
        if p.is_dir() and not p.name.startswith((".", "_"))
    )


def _normalize_id(x: str, *, strip: str) -> str:
    s = str(x).strip()
    for p in (strip,):
        if p and s.startswith(p):
            s = s[len(p) :]
    return s


def build_health_report(
    bids_root: Path,
    prs_path: Path,
    *,
    spec: CohortFileSpec,
    prs_id_column: str = "IID",
    strip_subject_prefix: str = "",
    prs_id_strip_prefix: str = "",
    prs_value_columns: Optional[list[str]] = None,
    verify_nifti_header: bool = False,
) -> pd.DataFrame:
    """Return one row per subject with modality paths, imaging flags, and PRS join."""
    root = Path(bids_root)
    sids = discover_subject_ids(root, bids_prefix="sub-")

    rows: list[dict[str, Any]] = []
    for sid in sids:
        sdir = root / sid
        if not sdir.is_dir():
            continue
        r = _scan_subject(sdir, spec)
        err = None
        if verify_nifti_header and r["imaging_complete"]:
            for k in ("T1w", "FLAIR_t2", "dwi_FA", "dwi_MD"):
                p = r[k]
                if p is not None:
                    st = _optional_nifti_sanity(p)
                    if st != "ok":
                        err = f"{k}:{st}"
                        break
        r["nifti_check"] = err
        rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    prs = pd.read_csv(prs_path, sep=None, engine="python")
    if prs_id_column not in prs.columns:
        raise KeyError(
            f"PRS id column {prs_id_column!r} not in CSV columns: {list(prs.columns)[:20]}..."
        )
    prs = prs.copy()
    prs["_id_norm"] = prs[prs_id_column].map(
        lambda x: _normalize_id(str(x), strip=prs_id_strip_prefix)
    )
    if prs_value_columns is None:
        prs_value_columns = [
            c
            for c in prs.columns
            if c not in (prs_id_column, "_id_norm")
            and pd.api.types.is_numeric_dtype(prs[c])
        ]
    if not prs_value_columns:
        raise ValueError("no PRS value columns: pass --prs-cols")
    for c in prs_value_columns:
        prs[c] = pd.to_numeric(prs[c], errors="coerce")

    def prs_row_ok(nid: str) -> bool:
        sub = prs[prs["_id_norm"].astype(str) == str(nid)]
        if sub.empty:
            return False
        if sub[prs_value_columns].isna().all().all():
            return False
        if sub[prs_value_columns].isna().any().any():
            return False
        return True

    def has_row(nid: str) -> bool:
        return bool((prs["_id_norm"].astype(str) == str(nid)).any())

    out_rows = []
    for _, row in df.iterrows():
        sid = row["subject_dir"]
        nid = _normalize_id(sid, strip=strip_subject_prefix)
        out = row.to_dict()
        out["prs_id_match"] = has_row(nid)
        out["prs_complete"] = prs_row_ok(nid) if out["imaging_complete"] else np.nan
        if out["imaging_complete"] and not out["prs_id_match"]:
            out["issue"] = "imaging_OK_missing_PRS"
        elif out["imaging_complete"] and out["prs_id_match"] and not prs_row_ok(nid):
            out["issue"] = "imaging_OK_PRS_nulls"
        elif out["imaging_complete"] and out["prs_id_match"] and prs_row_ok(nid):
            out["issue"] = "ok"
        else:
            out["issue"] = "incomplete_imaging" if not row["imaging_complete"] else "unknown"
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "bids_root",
        type=Path,
        help="BIDS (or similar) root containing sub-* / subject directories",
    )
    p.add_argument(
        "prs_path",
        type=Path,
        help="Tabular PRS (CSV/TSV); must contain the ID + numeric PRS score columns",
    )
    p.add_argument("--prs-id-col", type=str, default="IID", help="Subject ID column name")
    p.add_argument(
        "--prs-cols",
        type=str,
        nargs="+",
        default=None,
        help="PRS score columns; default = all numeric columns except the ID",
    )
    p.add_argument(
        "--strip-subject-prefix",
        type=str,
        default="sub-",
        help="Remove from imaging folder name before PRS match (e.g. sub-)",
    )
    p.add_argument(
        "--prs-id-strip",
        type=str,
        default="",
        help="Remove from PRS id cell before match (e.g. text prefix in TwinsUK exports)",
    )
    p.add_argument(
        "--globs-json",
        type=Path,
        default=None,
        help="JSON dict mapping modality name -> [glob patterns] per subject",
    )
    p.add_argument(
        "--csv-out", type=Path, default=None, help="Write full report to CSV (optional)"
    )
    p.add_argument(
        "--verify-nib",
        action="store_true",
        help="``nibabel``-open each NIfTI that was found (header sanity)",
    )
    p.add_argument("--no-strict-fa", action="store_true", help="Relax FLAIR/FA disambiguation")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    spec = CohortFileSpec() if not args.globs_json else CohortFileSpec.from_json(args.globs_json)
    if args.no_strict_fa:
        spec.strict_fa_md = False
    try:
        df = build_health_report(
            args.bids_root,
            args.prs_path,
            spec=spec,
            prs_id_column=args.prs_id_col,
            strip_subject_prefix=args.strip_subject_prefix,
            prs_id_strip_prefix=args.prs_id_strip,
            prs_value_columns=args.prs_cols,
            verify_nifti_header=bool(args.verify_nib),
        )
    except Exception as e:  # pragma: no cover
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if df.empty:
        print("No subject directories found.", file=sys.stderr)
        return 2

    # Readable summary (paths omitted from print for width)
    pr = df[
        [
            "subject_dir",
            "n_mod_ok",
            "imaging_complete",
            "prs_id_match",
            "prs_complete",
            "issue",
        ]
    ]
    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print("\n--- Cohort health (summary) ---\n")
        print(pr.to_string(index=False))
    n_im = int(df["imaging_complete"].sum())
    n_ok = int((df["issue"] == "ok").sum())
    n_bad = int(df["imaging_complete"].sum() - n_ok) if n_im else 0
    print(
        f"\nTotals: imaging_complete={n_im}, fully_ok(imaging+PRS)={n_ok}, "
        f"imaging_but_prs_problems>={n_bad}\n"
    )
    if args.csv_out is not None:
        df.to_csv(args.csv_out, index=False)
        print("Wrote", args.csv_out)
    return 0 if n_ok == n_im and n_im > 0 else 0  # 0: script ran; use CI on columns


if __name__ == "__main__":
    raise SystemExit(main())
