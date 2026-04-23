#!/usr/bin/env python
"""Smoke: synthetic label NIfTI + ``map_nodes_to_volume`` (no training, no real MRI)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.synthetic_atlas import generate_synthetic_atlas_nifti
from src.utils.visualization import map_nodes_to_volume


def main() -> int:
    n = 32
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        lab = tdir / "lab.nii.gz"
        out = tdir / "painted.nii.gz"
        generate_synthetic_atlas_nifti(
            n, (64, 64, 64), lab, affine=None
        )
        scores = (np.arange(n, dtype=np.float32) + 1.0) / n
        map_nodes_to_volume(scores, lab, out, fill_background=0.0)
        import nibabel as nib

        o = np.asarray(nib.load(str(out)).dataobj, dtype=np.float32)
        assert o.max() > 0
    print("synthetic_atlas + reverse-map smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
