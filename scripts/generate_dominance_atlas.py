#!/usr/bin/env python
"""Per-supervoxel **modality dominance** atlas from cross-modal MHA (KCL P65 Aim 2).

Loads a trained Siamese checkpoint, runs ``encode_modality_attention`` on one graph,
pools each node's (M,M) attention like :func:`pooled_modality_query_importance`
(default: row-sum), takes **argmax** → categorical **1..M**, and reverse-maps into
the SLIC label NIfTI via :func:`map_nodes_to_volume`.

**Pair selection**

* ``--pair-index`` with ``--search-split val|all`` — use a row of the validation
  fold or the full ``pairs.csv`` list.
* ``--subject-id`` — find the first pair whose ``subject_a`` or ``subject_b``
  matches (``sub-12`` vs ``12`` accepted). Use ``--search-split val`` (default)
  or ``all`` to search the whole cohort. With ``--subject-id``, ``--twin auto``
  picks the graph for the matching subject.

**Inputs**
  - ``--run-dir`` (``config.json`` + ``fold_XX/best.pt``) — one trained checkpoint per
    fold, shared for all subjects (graph identity is in the per-subject ``.pt``).
  - ``--slic-labels`` — 3D integer supervoxel ``.nii.gz`` for **that** individual;
    or omit it and set ``--data-root`` + ``--deriv-slic-subdir`` so a file matching
    that subject (see ``*subject*slic*`` under that directory) is chosen.

Example::

    python scripts/generate_dominance_atlas.py \\
        --run-dir runs/myexp \\
        --subject-id sub-01234 \\
        --data-root /path/to/cohort \\
        --output-dir runs/myexp/dominance \\
        --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.splits import family_stratified_kfold
from src.models.siamese_gnn import build_siamese_model
from src.training import TrainConfig
from src.utils import TwinBrainDataset, get_device, set_seed
from src.utils.visualization import (
    map_nodes_to_volume,
    per_node_dominant_modality,
    plot_dominance_atlas_orthogonal,
)


def _load_train_config(path: Path) -> TrainConfig:
    raw: dict[str, Any] = json.loads(path.read_text())
    if raw.get("modality_feature_dims") is not None:
        raw["modality_feature_dims"] = tuple(raw["modality_feature_dims"])
    if isinstance(raw.get("include_zygosities"), list):
        raw["include_zygosities"] = tuple(raw["include_zygosities"])
    if isinstance(raw.get("modality_names"), list):
        raw["modality_names"] = tuple(raw["modality_names"])
    for key in ("data_root", "output_dir"):
        if isinstance(raw.get(key), str):
            raw[key] = Path(raw[key])
    kwargs: dict[str, Any] = {}
    for f in TrainConfig.__dataclass_fields__:
        if f in raw:
            kwargs[f] = raw[f]
    return TrainConfig(**kwargs)


def _subject_id_variants(s: str) -> set[str]:
    """BIDS / TwinsUK: match ``x`` to ``sub-x`` and vice versa."""
    x = str(s).strip()
    o: set[str] = {x}
    if x.lower().startswith("sub-"):
        o.add(x[4:])
    else:
        o.add("sub-" + x)
    return o


def _ids_match(s_id: str, target: str) -> bool:
    a, b = _subject_id_variants(s_id), _subject_id_variants(target)
    return not a.isdisjoint(b)


def _find_val_pair_for_subject(
    ds: TwinBrainDataset,
    val_idx: np.ndarray,
    target: str,
) -> tuple[int, str]:
    """Return (index into ``val_ds`` subset) and which twin ``"a"`` or ``"b"`` has ``target``."""
    for j, global_i in enumerate(val_idx.tolist()):
        rec = ds.pairs[global_i]
        a_id, b_id = str(rec["subject_a"]), str(rec["subject_b"])
        if _ids_match(a_id, target):
            return j, "a"
        if _ids_match(b_id, target):
            return j, "b"
    raise KeyError(target)


def _find_any_pair_for_subject(ds: TwinBrainDataset, target: str) -> tuple[int, str]:
    for j in range(len(ds)):
        rec = ds.pairs[j]
        a_id, b_id = str(rec["subject_a"]), str(rec["subject_b"])
        if _ids_match(a_id, target):
            return j, "a"
        if _ids_match(b_id, target):
            return j, "b"
    raise KeyError(target)


def _slic_basename_globs_for_subject(subject_id: str) -> list[str]:
    """Filenames to try when resolving ``derivatives/slic/`` layout."""
    s = str(subject_id).strip()
    short = s[4:] if s.lower().startswith("sub-") else s
    return [f"*{s}*slic*.nii*", f"*{s}*slic*.nii", f"*{short}*slic*.nii*"]


def _resolve_slic_nifti(
    data_root: Path,
    subject_id: str,
    explicit: Optional[Path],
    deriv_slic_subdir: str = "derivatives/slic",
) -> Path:
    """Resolve SLIC label NIfTI: explicit path, else search under *data_root*."""
    if explicit is not None and str(explicit).strip():
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(f"SLIC labels file not found: {p}")
    # Convention: e.g. {data_root}/derivatives/slic/*<subject>*slic*.nii.gz
    subdir = (data_root / deriv_slic_subdir).resolve()
    if not subdir.is_dir():
        raise FileNotFoundError(
            f"No --slic-labels given and {subdir} is not a directory. "
            "Pass --slic-labels or create that folder with per-subject SLIC NIfTIs."
        )
    for pattern in _slic_basename_globs_for_subject(subject_id):
        matches = sorted(subdir.glob(pattern))
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            # Deterministic: prefer shortest name (avoids long derivative paths in stem)
            matches = sorted(matches, key=lambda p: (len(p.name), p.name))
            return matches[0].resolve()
    # Explicit fallbacks: exact stem names
    s = str(subject_id).strip()
    short = s[4:] if s.lower().startswith("sub-") else s
    for stem in (s, f"sub-{short}", short):
        for suffix in (".nii.gz", ".nii"):
            cand = subdir / f"{stem}_slic{suffix}"
            if cand.is_file():
                return cand.resolve()
    tried = _slic_basename_globs_for_subject(subject_id)
    raise FileNotFoundError(
        f"Could not find a SLIC NIfTI for subject {subject_id!r} under {subdir}. "
        f"Tried glob patterns {tried!r} and <subject>_slic.nii.gz. "
        "Pass --slic-labels to an existing file."
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--slic-labels",
        type=Path,
        default=None,
        help="SLIC / supervoxel label NIfTI. If omitted, must resolve under "
        "--data-root / --deriv-slic-subdir (see script doc).",
    )
    p.add_argument(
        "--deriv-slic-subdir",
        type=str,
        default="derivatives/slic",
        help="Under --data-root: searched for *<subject>*slic*.nii* when --slic-labels is omitted.",
    )
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument(
        "--pair-index",
        type=int,
        default=0,
        help="Index into the chosen split (ignored when --subject-id is set).",
    )
    p.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Select the pair that contains this subject (subject_a or subject_b). "
        "Accepts e.g. sub-12 or 12; first match in --search-split wins.",
    )
    p.add_argument(
        "--search-split",
        choices=["val", "all"],
        default="val",
        help="val=validation fold of --fold only; all=entire cohort in pairs.csv",
    )
    p.add_argument(
        "--twin",
        choices=["a", "b", "auto"],
        default="auto",
        help="Which twin graph to use: auto=the half that contains --subject-id, "
        " else force twin A or B (must match the subject for that side).",
    )
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--pool", default="rowsum", choices=["rowsum", "colsum", "trace"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out = args.output_dir or (args.run_dir / "dominance_atlas")
    out.mkdir(parents=True, exist_ok=True)

    cfg = _load_train_config(args.run_dir / "config.json")
    data_root = args.data_root or cfg.data_root
    s_cfg = cfg.to_siamese()
    names: list[str]
    if s_cfg.modality_names:
        names = list(s_cfg.modality_names)
    elif s_cfg.modality_feature_dims:
        names = [f"M{i+1}" for i in range(len(s_cfg.modality_feature_dims))]
    else:
        print("Run config has no modality_feature_dims / modality_names.", file=sys.stderr)
        return 2

    model = build_siamese_model(s_cfg).to(device)
    ckpt = args.run_dir / f"fold_{args.fold:02d}" / "best.pt"
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    ds = TwinBrainDataset(
        data_root,
        include_zygosities=set(cfg.include_zygosities),
        preload=True,
    )
    df = pd.DataFrame(ds.pairs)
    splits = list(
        family_stratified_kfold(
            df, n_splits=cfg.n_splits, shuffle=True, seed=cfg.seed
        )
    )
    _, val_idx = splits[args.fold]
    val_ds = Subset(ds, val_idx.tolist())
    if args.subject_id is not None:
        try:
            if args.search_split == "val":
                pi, side = _find_val_pair_for_subject(ds, val_idx, args.subject_id)
                data_a, data_b, meta = val_ds[pi]
            else:
                pi, side = _find_any_pair_for_subject(ds, args.subject_id)
                data_a, data_b, meta = ds[pi]
        except KeyError:
            sp = f"validation fold {args.fold}" if args.search_split == "val" else "the cohort"
            print(
                f"Subject {args.subject_id!r} not found in {sp}.",
                file=sys.stderr,
            )
            return 3
        if args.twin == "auto":
            data = data_a if side == "a" else data_b
        elif args.twin == "a" and side != "a":
            print(
                f"Subject is twin B in this pair; use --twin b or --twin auto (got {meta}).",
                file=sys.stderr,
            )
            return 6
        elif args.twin == "b" and side != "b":
            print(
                f"Subject is twin A in this pair; use --twin a or --twin auto (got {meta}).",
                file=sys.stderr,
            )
            return 6
        else:
            data = data_a if args.twin == "a" else data_b
        print(
            f"Selected pair index {pi} in {'val' if args.search_split == 'val' else 'all'}: "
            f"subject_a={meta['subject_a']!r} subject_b={meta['subject_b']!r} zygosity={meta['zygosity']}"
        )
    else:
        if args.search_split == "val":
            if len(val_ds) == 0:
                print("empty validation split", file=sys.stderr)
                return 3
            n_pairs = len(val_ds)
            if args.pair_index >= n_pairs:
                print("pair-index out of range", file=sys.stderr)
                return 3
            data_a, data_b, _ = val_ds[args.pair_index]
        else:
            if args.pair_index >= len(ds):
                print("pair-index out of range", file=sys.stderr)
                return 3
            data_a, data_b, _ = ds[args.pair_index]
        if args.twin not in ("a", "b"):
            data = data_a  # auto: same as previous default (twin A)
        else:
            data = data_a if args.twin == "a" else data_b

    subj_for_slic = str(getattr(data, "subject_id", "") or "").strip()
    if not subj_for_slic and args.subject_id:
        subj_for_slic = str(args.subject_id).strip()
    if not subj_for_slic and args.slic_labels is None:
        print(
            "Graph has no subject_id and --slic-labels is missing: "
            "set --slic-labels to the subject's SLIC NIfTI.",
            file=sys.stderr,
        )
        return 7
    try:
        slic_nii = _resolve_slic_nifti(
            Path(data_root), subj_for_slic, args.slic_labels, args.deriv_slic_subdir
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 7

    data = data.to(device)

    attn: Optional[torch.Tensor] = model.encode_modality_attention(data)  # type: ignore[union-attr]
    if attn is None:
        print("No modality attention in this run (enable cross-modal MHA).", file=sys.stderr)
        return 4
    a = attn.detach().float().cpu().numpy()
    if a.ndim != 3:
        print(f"Unexpected attention shape {a.shape}", file=sys.stderr)
        return 5
    labels = per_node_dominant_modality(a, pool=args.pool)  # float 0..M

    nii_out = out / "dominance_atlas.nii.gz"
    map_nodes_to_volume(
        labels,
        slic_nii,
        nii_out,
        fill_background=0.0,
        dtype=np.float32,
    )
    png = out / "dominance_atlas_preview.png"
    plot_dominance_atlas_orthogonal(nii_out, png, modality_names=names)
    print("Wrote", nii_out)
    print("Wrote", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
