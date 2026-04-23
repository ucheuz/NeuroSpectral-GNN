#!/usr/bin/env python
"""Average cross-modal M×M attention over **MZ** vs **DZ** validation pairs (KCL P65).

Requires a training run directory with ``config.json`` and ``fold_XX/best.pt``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.splits import family_stratified_kfold
from src.models.siamese_gnn import build_siamese_model
from src.training import TrainConfig
from src.utils import TwinBrainDataset, get_device, set_seed
from src.utils.brain_dataset import twin_collate
from src.utils.visualization import (
    plot_modality_importance_barchart,
    pooled_modality_query_importance,
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


def _load_ckpt(ckpt: Path, device: torch.device) -> dict:
    return torch.load(ckpt, map_location=device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override ``TrainConfig.data_root`` (default: from run config.json).",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device, verbose=True)
    out = args.output_dir or (args.run_dir / "p65_modality_attention")
    out.mkdir(parents=True, exist_ok=True)

    cfg = _load_train_config(args.run_dir / "config.json")
    data_root: Path = args.data_root if args.data_root is not None else cfg.data_root
    model = build_siamese_model(cfg.to_siamese()).to(device)
    ckpt = args.run_dir / f"fold_{args.fold:02d}" / "best.pt"
    if not ckpt.is_file():
        print(f"Missing checkpoint: {ckpt}", file=sys.stderr)
        return 1
    state = _load_ckpt(ckpt, device)
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
    loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, collate_fn=twin_collate, drop_last=False
    )

    s_cfg = cfg.to_siamese()
    m_dims = s_cfg.modality_feature_dims
    names: list[str]
    if s_cfg.modality_names:
        names = list(s_cfg.modality_names)
    elif m_dims is not None:
        names = [f"Mod{i+1}" for i in range(len(m_dims))]
    else:
        names = ["m1", "m2"]

    acc_mz: list[np.ndarray] = []
    acc_dz: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        m_a = model.encode_modality_attention(batch.data_a)  # type: ignore[union-attr]
        m_b = model.encode_modality_attention(batch.data_b)  # type: ignore[union-attr]
        if m_a is None or m_b is None:
            print(
                "This run has no cross-modal MHA; set use_cross_modal_attention=True.",
                file=sys.stderr,
            )
            return 2
        ba = batch.data_a.batch
        bb = batch.data_b.batch
        bsz = int(ba.max().item()) + 1
        for b in range(bsz):
            a_sub = m_a[ba == b].mean(dim=0)
            b_sub = m_b[bb == b].mean(dim=0)
            ap = 0.5 * (a_sub + b_sub)
            ap_np = ap.detach().cpu().numpy()
            zy = batch.zygosities[b]
            if zy == "MZ":
                acc_mz.append(ap_np)
            elif zy == "DZ":
                acc_dz.append(ap_np)

    if not acc_mz or not acc_dz:
        print("Not enough MZ or DZ pairs in the validation fold.", file=sys.stderr)
        return 3
    arr_mz = np.stack(acc_mz, axis=0).mean(axis=0)
    arr_dz = np.stack(acc_dz, axis=0).mean(axis=0)
    np.savez(
        out / "modality_attention_mz_dz.npz",
        attention_mz=arr_mz,
        attention_dz=arr_dz,
    )

    imp_mz = pooled_modality_query_importance(arr_mz, pool="rowsum")
    imp_dz = pooled_modality_query_importance(arr_dz, pool="rowsum")
    m = int(len(imp_mz))
    names = names[:m] + [f"M{i+1}" for i in range(len(names), m)]
    plot_modality_importance_barchart(
        imp_mz, imp_dz, names, out / "modality_importance_mz_dz.png"
    )
    print(f"Wrote {out / 'modality_importance_mz_dz.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
