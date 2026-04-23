#!/usr/bin/env python
"""End-to-end smoke test for the preprocessing pipeline.

Downloads a small nilearn development_fmri sample (2 subjects), builds a fake
'twin' manifest (treating them as two members of one family), runs the full
pipeline, and asserts the outputs are well-formed PyG Data objects.

This is safe to run without the supervisor's real twin dataset.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nilearn import datasets  # noqa: E402

from src.preprocessing.atlas import load_schaefer_atlas  # noqa: E402
from src.preprocessing.connectivity import TimeseriesConfig  # noqa: E402
from src.preprocessing.graph import GraphBuildConfig  # noqa: E402
from src.preprocessing.manifest import (  # noqa: E402
    build_twin_pairs,
    load_manifest,
    pairs_to_dataframe,
)
from src.preprocessing.pipeline import PreprocessConfig, preprocess_subject  # noqa: E402


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("smoke_test")

    log.info("Fetching nilearn development_fmri sample...")
    try:
        sample = datasets.fetch_development_fmri(n_subjects=2)
        funcs = list(sample.func)
        confs = list(sample.confounds)
    except Exception as exc:
        log.warning("Full fetch failed (%s); falling back to 1 cached subject.", exc)
        sample = datasets.fetch_development_fmri(n_subjects=1)
        funcs = list(sample.func) * 2  # reuse for mechanical smoke test
        confs = list(sample.confounds) * 2

    with tempfile.TemporaryDirectory(prefix="nsgnn_smoke_") as tmp:
        tmp = Path(tmp)
        manifest_rows = []
        for i, (func, conf) in enumerate(zip(funcs, confs)):
            manifest_rows.append(
                {
                    "subject_id": f"SUBJ{i:02d}",
                    "nii_path": func,
                    "confounds_path": conf,
                    "family_id": "FAM01",
                    "twin_id": "A" if i == 0 else "B",
                    "zygosity": "MZ",
                    "t_r": 2.0,  # development_fmri TR
                }
            )
        manifest_csv = tmp / "manifest.csv"
        pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

        log.info("Loading Schaefer-100 atlas...")
        # Use whichever resolution is already cached to avoid network fetches
        # during the smoke test. 1mm is slower but equivalent in ROI layout.
        atlas = load_schaefer_atlas(n_rois=100, yeo_networks=7, resolution_mm=1)

        config = PreprocessConfig(
            output_dir=tmp / "processed",
            atlas=atlas,
            ts_config=TimeseriesConfig(t_r=2.0),
            graph_config=GraphBuildConfig(
                sparsify_strategy="proportional",
                keep_top_fraction=0.2,
                node_feature_mode="profile",
            ),
            nilearn_cache_dir=tmp / "nilearn_cache",
            overwrite=True,
        )

        records = load_manifest(manifest_csv)
        pairs = build_twin_pairs(records)
        assert len(pairs) == 1, f"Expected 1 twin pair, got {len(pairs)}"
        pairs_to_dataframe(pairs).to_csv(config.output_dir.parent / "pairs.csv", index=False)

        for rec in records:
            result = preprocess_subject(rec, config)
            log.info("Result for %s: %s", rec.subject_id, result)
            assert result["status"] == "ok", f"Preprocessing failed: {result}"

            data = torch.load(result["output_path"], weights_only=False)
            log.info(
                "Loaded %s: x=%s edge_index=%s edge_attr=%s zygosity=%s",
                rec.subject_id,
                tuple(data.x.shape),
                tuple(data.edge_index.shape),
                tuple(data.edge_attr.shape) if data.edge_attr is not None else None,
                data.zygosity,
            )
            assert data.x.shape == (100, 100), f"Unexpected x shape: {data.x.shape}"
            assert data.edge_index.size(0) == 2
            assert data.num_nodes == 100
            assert data.connectivity.shape == (100, 100)

        log.info("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
