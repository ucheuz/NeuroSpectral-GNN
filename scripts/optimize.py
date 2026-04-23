#!/usr/bin/env python
"""Hyperparameter search for the Siamese GNN using Optuna (KCL P65).

Optimises (on validation loss from a single CV fold):
  - learning_rate (log-uniform)
  - contrastive_margin in [0.5, 2.0]
  - hidden_channels (GCN)
  - cross_modal_num_heads (divides cross_modal_d_model)

Writes ``best_config.json``, ``optuna_trials.csv``, and optional HTML/PNG
summaries (if visualization backends load).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import optuna
except ModuleNotFoundError as e:  # pragma: no cover
    if e.name != "optuna":
        raise
    exe = sys.executable
    sys.stderr.write(
        "Missing dependency: optuna in the **same** Python you use to run this script.\n"
        f"  This interpreter: {exe}\n"
        "Install with (always use this so pip targets the right env):\n"
        f"  {exe} -m pip install 'optuna>=3.6.0'\n"
        "Plain `pip install` can target a different Python (e.g. system 3.13 vs conda env).\n"
    )
    raise SystemExit(1) from e
import pandas as pd
from optuna.trial import Trial
from torch.utils.data import Subset

from src.analysis.splits import family_stratified_kfold
from src.training import TrainConfig, train_single_fold
from src.utils import TwinBrainDataset, get_device, set_seed

_STORAGE_SKIP = (ImportError, RuntimeError, AttributeError)


def _objective(
    trial: Trial,
    base: TrainConfig,
    fold: int,
) -> float:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    margin = trial.suggest_float("contrastive_margin", 0.5, 2.0)
    hidden = trial.suggest_categorical("hidden_channels", [32, 64, 96, 128])
    d_model = base.cross_modal_d_model
    valid_heads = [h for h in (2, 4, 8) if d_model % h == 0]
    if not valid_heads:
        valid_heads = [4]
    num_heads = trial.suggest_categorical("cross_modal_num_heads", valid_heads)

    cfg = replace(
        base,
        output_dir=base.output_dir,
        hidden_channels=hidden,
        contrastive_margin=margin,
        cross_modal_num_heads=num_heads,
        lr=lr,
        patience=min(base.patience, 8),
        tensorboard=False,
        save_checkpoints=False,
    )

    set_seed(base.seed)
    device = get_device(base.device_preference)
    base_dataset = TwinBrainDataset(
        cfg.data_root, include_zygosities=set(cfg.include_zygosities), preload=True
    )
    pairs_df = pd.DataFrame(base_dataset.pairs)
    splits = list(
        family_stratified_kfold(
            pairs_df, n_splits=cfg.n_splits, shuffle=True, seed=cfg.seed
        )
    )
    tr_idx, va_idx = splits[fold]
    train_ds = Subset(base_dataset, tr_idx.tolist())
    val_ds = Subset(base_dataset, va_idx.tolist())

    result = train_single_fold(0, train_ds, val_ds, cfg, device)
    return float(result.best_val_loss)


def _require_cohort_layout(data_root: Path) -> None:
    """Exit with a clear message if ``data_root`` is not a preprocessed twin cohort."""
    root = data_root.resolve()
    if not root.is_dir():
        raise SystemExit(
            f"data-root is not a directory: {root}\n"
            "Expected a folder with ``pairs.csv`` and ``subjects/`` (see ``TwinBrainDataset``)."
        )
    pairs = root / "pairs.csv"
    subj = root / "subjects"
    if not pairs.is_file():
        raise SystemExit(
            f"Missing: {pairs}\n"
            "``--data-root`` is not a valid cohort (common mistake: the placeholder data/your_cohort).\n"
            "Create one first, e.g.\n"
            "  python scripts/generate_synthetic_twins.py --output-dir data/synthetic_h060\n"
            "then:\n"
            "  python scripts/optimize.py --data-root data/synthetic_h060 --output-dir runs/hpo1 ...\n"
        )
    if not subj.is_dir():
        raise SystemExit(
            f"Missing directory: {subj}/\n"
            f"(Expected one ``*.pt`` graph per subject next to {pairs}.)"
        )


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Cohort root with pairs.csv and subjects/ (e.g. data/synthetic_h060). "
        "Not a placeholder path.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--fold", type=int, default=0, help="CV fold index for validation")
    p.add_argument("--max-epochs", type=int, default=25, help="Short runs for HPO")
    p.add_argument("--in-channels", type=int, default=100)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--patience", type=int, default=12, help="Capped in-objective for speed"
    )
    p.add_argument("--use-cross-modal-attention", action="store_true")
    p.add_argument(
        "--modality-feature-dims",
        type=int,
        nargs="+",
        default=None,
    )
    p.add_argument("--cross-modal-d-model", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    p.add_argument("--prs-dim", type=int, default=0)
    p.add_argument("--model-type", default="auto")
    return p.parse_args()


def main() -> int:
    args = _parse()
    _require_cohort_layout(args.data_root)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    modality: tuple[int, ...] | None
    if args.modality_feature_dims:
        modality = tuple(args.modality_feature_dims)
    else:
        modality = None
    if modality is not None and sum(modality) != args.in_channels:
        raise SystemExit("modality_feature_dims must sum to --in-channels")

    mha = args.use_cross_modal_attention or (
        modality is not None and len(modality) > 1
    )

    base = TrainConfig(
        data_root=args.data_root,
        output_dir=out / "_trial_base",
        in_channels=args.in_channels,
        max_epochs=args.max_epochs,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        patience=args.patience,
        use_cross_modal_attention=mha,
        modality_feature_dims=modality,
        cross_modal_d_model=args.cross_modal_d_model,
        prs_dim=args.prs_dim,
        model_type=args.model_type,
        seed=args.seed,
        device_preference=args.device,
    )

    def objective(trial: Trial) -> float:
        trial_dir = out / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        tcfg = replace(base, output_dir=trial_dir)
        return _objective(trial, tcfg, fold=args.fold)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best: dict[str, Any] = dict(study.best_trial.params)
    best["best_val_loss"] = study.best_value
    best["n_trials"] = len(study.trials)
    (out / "best_config.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )

    rows = []
    for t in study.trials:
        row: dict[str, Any] = {"number": t.number, "value": t.value, "state": t.state.name}
        row.update(t.params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "optuna_trials.csv", index=False)

    try:
        import optuna.visualization as vis  # type: ignore

        vis.plot_parallel_coordinate(study).write_html(str(out / "optuna_parallel.html"))
        vis.plot_param_importances(study).write_html(
            str(out / "optuna_param_importances.html")
        )
    except _STORAGE_SKIP:
        pass

    try:
        import optuna.visualization.matplotlib as ovm  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        ovm.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(out / "optuna_optimization_history.png", dpi=150)
        plt.close()
    except _STORAGE_SKIP:
        pass

    print("Best params:", study.best_params)
    print("Saved:", out / "best_config.json", out / "optuna_trials.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
