#!/usr/bin/env python
"""Generate a synthetic twin cohort with a tunable ground-truth heritability.

Outputs mimic the real preprocessing pipeline layout so they're a drop-in
replacement for a ``TwinBrainDataset``.

Example
-------
    # 40 MZ + 40 DZ pairs, h^2 = 0.6, Schaefer-100 sized graphs
    python scripts/generate_synthetic_twins.py \
        --output-dir data/synthetic_h060 \
        --n-mz 40 --n-dz 40 --heritability 0.6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.graph import GraphBuildConfig  # noqa: E402
from src.preprocessing.manifest import load_manifest  # noqa: E402, F401
from src.preprocessing.synthetic import (  # noqa: E402
    SyntheticCohortConfig,
    empirical_heritability_from_connectivities,
    generate_cohort,
    save_synthetic_cohort,
)
from src.utils.seeds import set_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-mz", type=int, default=40)
    p.add_argument("--n-dz", type=int, default=40)
    p.add_argument("--n-unrelated", type=int, default=0)
    p.add_argument("--n-rois", type=int, default=100)
    p.add_argument("--heritability", type=float, default=0.6)
    p.add_argument("--n-modules", type=int, default=7)
    p.add_argument("--mean-template", default="block", choices=["zero", "block"])
    p.add_argument("--keep-top-fraction", type=float, default=0.20)
    p.add_argument("--sparsify-strategy", default="proportional",
                   choices=["proportional", "topk", "absolute"])
    p.add_argument("--topk-per-node", type=int, default=10)
    p.add_argument("--node-feature-mode", default="profile",
                   choices=["profile", "identity", "degree_profile"])
    p.add_argument("--prs-dim", type=int, default=0,
                   help="Dimensionality of emulated polygenic risk score vector. "
                        "0 disables PRS generation.")
    p.add_argument("--prs-informativeness", type=float, default=1.0,
                   help="Signal scale of the PRS vector (how much genetic info it carries).")
    p.add_argument("--prs-noise-scale", type=float, default=0.25,
                   help="Per-subject PRS measurement noise (independent per twin).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("generate_synthetic_twins")
    set_seed(args.seed)

    cfg = SyntheticCohortConfig(
        n_mz_pairs=args.n_mz,
        n_dz_pairs=args.n_dz,
        n_unrelated_pairs=args.n_unrelated,
        n_rois=args.n_rois,
        heritability=args.heritability,
        n_modules=args.n_modules,
        mean_template=args.mean_template,
        prs_dim=args.prs_dim,
        prs_informativeness=args.prs_informativeness,
        prs_noise_scale=args.prs_noise_scale,
        seed=args.seed,
        graph_config=GraphBuildConfig(
            sparsify_strategy=args.sparsify_strategy,
            keep_top_fraction=args.keep_top_fraction,
            topk_per_node=args.topk_per_node,
            node_feature_mode=args.node_feature_mode,
        ),
    )

    paths = save_synthetic_cohort(cfg, args.output_dir)
    log.info("Outputs:\n  %s", "\n  ".join(f"{k}: {v}" for k, v in paths.items()))

    # Sanity check: Falconer's h^2 estimate on the raw matrices should
    # track the ground truth closely with ~40+ pairs.
    connectivities, pairs, _ = generate_cohort(cfg)
    h2_hat = empirical_heritability_from_connectivities(connectivities, pairs)
    log.info(
        "Sanity check: ground truth h^2=%.3f | Falconer estimate h^2=%.3f",
        cfg.heritability,
        h2_hat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
