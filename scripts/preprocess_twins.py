#!/usr/bin/env python
"""Batch-preprocess a cohort of twin fMRI scans into PyG graphs.

Usage
-----
    python scripts/preprocess_twins.py \
        --manifest data/manifest.csv \
        --output-dir data/processed \
        --n-rois 100 \
        --t-r 0.72 \
        --keep-top-fraction 0.2 \
        --n-jobs 4

Outputs
-------
    {output_dir}/subjects/{subject_id}.pt   One PyG Data per subject.
    {output_dir}/pairs.csv                  TwinPair table for the DataLoader.
    {output_dir}/preprocess_report.csv      Per-subject timing + error log.
    {output_dir}/config.json                Snapshot of all preprocessing flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

# Ensure the repo root is on the path whether run as a script or a module.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from joblib import Parallel, delayed  # noqa: E402

from src.preprocessing.atlas import load_schaefer_atlas  # noqa: E402
from src.preprocessing.connectivity import TimeseriesConfig  # noqa: E402
from src.preprocessing.graph import GraphBuildConfig  # noqa: E402
from src.preprocessing.manifest import (  # noqa: E402
    build_twin_pairs,
    load_manifest,
    pairs_to_dataframe,
)
from src.preprocessing.pipeline import (  # noqa: E402
    PreprocessConfig,
    preprocess_subject,
)
from src.utils.seeds import set_seed  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, type=Path,
                   help="CSV with subject_id, nii_path, family_id, twin_id, zygosity")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory where processed subjects and pairs.csv go")

    # Atlas
    p.add_argument("--n-rois", type=int, default=100,
                   choices=[100, 200, 300, 400, 500, 600, 800, 1000])
    p.add_argument("--yeo-networks", type=int, default=7, choices=[7, 17])
    p.add_argument("--atlas-resolution-mm", type=int, default=2, choices=[1, 2])

    # Timeseries / filtering
    p.add_argument("--t-r", type=float, default=None,
                   help="Repetition time (s) for bandpass. Omit to disable filtering.")
    p.add_argument("--low-pass", type=float, default=0.10)
    p.add_argument("--high-pass", type=float, default=0.01)
    p.add_argument("--no-detrend", action="store_true")
    p.add_argument("--smoothing-fwhm", type=float, default=None)
    p.add_argument("--standardize", default="zscore_sample")

    # Graph construction
    p.add_argument("--sparsify-strategy",
                   choices=["proportional", "topk", "absolute"],
                   default="proportional")
    p.add_argument("--keep-top-fraction", type=float, default=0.20)
    p.add_argument("--topk-per-node", type=int, default=10)
    p.add_argument("--absolute-threshold", type=float, default=0.3)
    p.add_argument("--node-feature-mode",
                   choices=["profile", "identity", "degree_profile"],
                   default="profile")

    # Twin pairs
    p.add_argument("--include-unrelated", action="store_true",
                   help="Sample extra UNREL pairs for hard negatives.")
    p.add_argument("--unrelated-per-subject", type=int, default=1)

    # Execution
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--nilearn-cache-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p.parse_args()


def _snapshot_config(args: argparse.Namespace, atlas_name: str) -> dict:
    """Freeze CLI args + derived atlas name into a JSON-serializable record."""
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    d["atlas_name"] = atlas_name
    return d


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("preprocess_twins")

    set_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load atlas (once, reused across subjects)
    atlas = load_schaefer_atlas(
        n_rois=args.n_rois,
        yeo_networks=args.yeo_networks,
        resolution_mm=args.atlas_resolution_mm,
    )

    # 2. Build configs
    ts_config = TimeseriesConfig(
        standardize=args.standardize,
        detrend=not args.no_detrend,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        t_r=args.t_r,
        smoothing_fwhm=args.smoothing_fwhm,
    )
    graph_config = GraphBuildConfig(
        sparsify_strategy=args.sparsify_strategy,
        keep_top_fraction=args.keep_top_fraction,
        topk_per_node=args.topk_per_node,
        absolute_threshold=args.absolute_threshold,
        node_feature_mode=args.node_feature_mode,
    )
    pp_config = PreprocessConfig(
        output_dir=args.output_dir,
        atlas=atlas,
        ts_config=ts_config,
        graph_config=graph_config,
        nilearn_cache_dir=args.nilearn_cache_dir,
        overwrite=args.overwrite,
    )

    # 3. Snapshot config for reproducibility
    config_snapshot = {
        "atlas": atlas.name,
        "ts_config": asdict(ts_config),
        "graph_config": asdict(graph_config),
        "cli": _snapshot_config(args, atlas.name),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config_snapshot, indent=2, default=str)
    )

    # 4. Load manifest + build twin pairs
    records = load_manifest(args.manifest)
    pairs = build_twin_pairs(
        records,
        include_unrelated=args.include_unrelated,
        unrelated_per_subject=args.unrelated_per_subject,
        rng_seed=args.seed,
    )
    pairs_df = pairs_to_dataframe(pairs)
    pairs_df.to_csv(args.output_dir / "pairs.csv", index=False)
    log.info("Wrote %d pairs to %s", len(pairs_df), args.output_dir / "pairs.csv")

    # 5. Run preprocessing in parallel
    log.info("Preprocessing %d subjects with n_jobs=%d", len(records), args.n_jobs)
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=10)(
        delayed(preprocess_subject)(record, pp_config) for record in records
    )

    # 6. Save per-subject report
    report_df = pd.DataFrame(results)
    report_path = args.output_dir / "preprocess_report.csv"
    report_df.to_csv(report_path, index=False)
    log.info("Wrote preprocessing report to %s", report_path)

    n_ok = int((report_df["status"] == "ok").sum())
    n_skip = int((report_df["status"] == "skipped").sum())
    n_err = int((report_df["status"] == "error").sum())
    log.info("Summary: ok=%d  skipped=%d  error=%d", n_ok, n_skip, n_err)

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
