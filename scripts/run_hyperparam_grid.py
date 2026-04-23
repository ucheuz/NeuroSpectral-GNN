#!/usr/bin/env python
"""Small hyperparameter grid on synthetic data (for dissertation / methods).

Default: sweeps **contrastive** ``metric`` × ``margin`` on one generated cohort
(unless ``--data-root`` points to an existing one).

Optionally add **data** ablations (new cohort per combo):

    ``--sweep-keep-fraction 0.2 0.3`` and/or ``--sweep-node-features profile degree_profile``

Example
-------
    python scripts/run_hyperparam_grid.py --output-dir runs/hparam_grid
    python scripts/run_hyperparam_grid.py --data-root data/pipeline_test/cohort_graph_only \\
        --output-dir runs/hparam_grid
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.graph import GraphBuildConfig  # noqa: E402
from src.preprocessing.synthetic import (  # noqa: E402
    SyntheticCohortConfig,
    save_synthetic_cohort,
)
from src.training import TrainConfig, run_cross_validation  # noqa: E402
from src.utils import set_seed  # noqa: E402

logger = logging.getLogger("hyperparam_grid")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Pre-built cohort (skips auto cohort generation unless data sweeps are on).",
    )
    p.add_argument("--metrics", nargs="+", default=["cosine", "euclidean"])
    p.add_argument("--margins", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--n-mz", type=int, default=32)
    p.add_argument("--n-dz", type=int, default=32)
    p.add_argument("--n-unrelated", type=int, default=0)
    p.add_argument("--n-rois", type=int, default=64)
    p.add_argument("--heritability", type=float, default=0.6)
    p.add_argument(
        "--sweep-keep-fraction",
        type=float,
        nargs="*",
        default=[],
        help="If set (one or more values), build one cohort per value.",
    )
    p.add_argument(
        "--sweep-node-features",
        type=str,
        nargs="*",
        default=[],
        help="If set, one of: profile, identity, degree_profile (cohort per value).",
    )
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--n-splits", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return p.parse_args()


def _write_cohort(
    out: Path, args: argparse.Namespace, keep: float, node_mode: str
) -> None:
    cfg = SyntheticCohortConfig(
        n_mz_pairs=args.n_mz,
        n_dz_pairs=args.n_dz,
        n_unrelated_pairs=args.n_unrelated,
        n_rois=args.n_rois,
        heritability=args.heritability,
        seed=args.seed,
        graph_config=GraphBuildConfig(
            sparsify_strategy="proportional",
            keep_top_fraction=keep,
            node_feature_mode=node_mode,  # type: ignore[arg-type]
        ),
    )
    out.mkdir(parents=True, exist_ok=True)
    save_synthetic_cohort(cfg, out)
    logger.info("Cohort: %s (keep=%.2f, node_feature=%s)", out, keep, node_mode)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "grid_config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
    )

    cohort_specs: list[tuple[Path, str, float | None, str | None]] = []
    data_sweep = bool(args.sweep_keep_fraction) or bool(args.sweep_node_features)

    if not data_sweep and args.data_root is not None:
        p = args.data_root.resolve()
        cohort_specs.append((p, str(p), None, None))
    else:
        kfs: list[float] = (
            list(args.sweep_keep_fraction) if args.sweep_keep_fraction else [0.2]
        )
        nfs: list[str] = (
            list(args.sweep_node_features)
            if args.sweep_node_features
            else ["profile"]
        )
        for i, (kf, nfm) in enumerate(product(kfs, nfs)):
            p = (args.output_dir / f"_cohort_{i:02d}").resolve()
            _write_cohort(p, args, kf, nfm)
            cohort_specs.append((p, f"keep={kf},nodes={nfm}", kf, nfm))

    zyg: tuple[str, ...] = (
        ("MZ", "DZ", "UNREL") if args.n_unrelated > 0 else ("MZ", "DZ")
    )
    rows: list[dict[str, Any]] = []

    for data_root, data_label, kf, nfm in cohort_specs:
        for mi, margin in product(args.metrics, args.margins):
            tag = f"{data_label}_metric_{mi}_mg{margin}".replace(" ", "_").replace(
                ".", "p"
            )[:200]
            run_dir = (args.output_dir / "runs" / tag).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info("=== %s | metric=%s margin=%.2f ===", data_label, mi, margin)
            cfg = TrainConfig(
                data_root=data_root,
                output_dir=run_dir,
                include_zygosities=zyg,
                in_channels=args.n_rois,
                max_epochs=args.max_epochs,
                n_splits=args.n_splits,
                batch_size=args.batch_size,
                contrastive_metric=mi,
                contrastive_margin=margin,
                tensorboard=not args.no_tensorboard,
                save_checkpoints=False,
                seed=args.seed,
            )
            try:
                summary = run_cross_validation(cfg)
            except Exception as e:
                logger.exception("Run failed: %s", e)
                rows.append(
                    {
                        "data_label": data_label,
                        "data_root": str(data_root),
                        "keep_top_fraction": kf,
                        "node_feature_mode": nfm,
                        "contrastive_metric": mi,
                        "contrastive_margin": margin,
                        "mean_h2": None,
                        "mean_auc": None,
                        "error": repr(e),
                    }
                )
                continue
            rows.append(
                {
                    "data_label": data_label,
                    "data_root": str(data_root),
                    "keep_top_fraction": kf,
                    "node_feature_mode": nfm,
                    "contrastive_metric": mi,
                    "contrastive_margin": margin,
                    "mean_h2": summary["mean_h2"],
                    "std_h2": summary["std_h2"],
                    "mean_auc": summary["mean_auc"],
                    "mean_distance_gap": summary.get("mean_distance_gap"),
                    "error": None,
                }
            )

    out_csv = args.output_dir / "grid_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Saved %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
