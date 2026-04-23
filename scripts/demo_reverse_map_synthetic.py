#!/usr/bin/env python
"""End-to-end test of ``map_nodes_to_volume`` with **synthetic** 3D labels (no real MRI).

This is a valid software pipeline: fake SLIC-style integer labels in MNI-style space
(identity affine) + random per-label scores → ``.nii.gz`` for FSLeyes / ITK-SNAP.
It does **not** claim biological validity; it proves the reverse-map + nibabel path.

    python scripts/demo_reverse_map_synthetic.py --output-dir data/synthetic_atlas_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.visualization import map_nodes_to_volume


def _make_wedge_labels(shape: tuple[int, int, int], n_labels: int) -> np.ndarray:
    """Voxel (i,j,k) -> label 1..min(n_labels, k+1) for a simple 3D toy partition."""
    z, y, x = shape
    d = np.zeros((z, y, x), dtype=np.int32)
    # slab along k (axis 2) so we get distinct connected-ish regions
    for lab in range(1, n_labels + 1):
        sl = (lab - 1) * (x // n_labels) + min((lab - 1) * 2, x - 1)
        w = max(1, x // n_labels)
        d[:, :, sl : sl + w] = lab
    d[d == 0] = 1
    return d


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "synthetic_atlas_demo")
    p.add_argument("--n-labels", type=int, default=8, help="Number of non-zero supervoxel IDs 1..K")
    p.add_argument("--shape", type=int, nargs=3, default=[32, 32, 32], metavar=("Z", "Y", "X"))
    args = p.parse_args()

    try:
        import nibabel as nib
    except ImportError as e:  # pragma: no cover
        print("nibabel required: pip install nibabel", file=sys.stderr)
        raise SystemExit(1) from e

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = out_dir / "synthetic_slic_labels.nii.gz"
    heat_path = out_dir / "synthetic_heritability_map.nii.gz"

    shape = (args.shape[0], args.shape[1], args.shape[2])
    n = int(args.n_labels)
    d = _make_wedge_labels(shape, n)
    affine = np.eye(4, dtype=np.float64)
    nib.save(nib.Nifti1Image(d, affine, nib.Nifti1Header()), str(label_path))

    # One scalar per supervoxel (1..K) — e.g. fake "importance" ramp
    node_values = np.linspace(0.0, 1.0, n, dtype=np.float32) ** 2
    map_nodes_to_volume(
        node_values,
        label_path,
        heat_path,
        background_label=0,
        fill_background=0.0,
    )
    print(f"Label NIfTI:  {label_path}")
    print(f"Heat NIfTI:  {heat_path}")
    print("Open the heat in FSLeyes/ITK-SNAP on top of labels (synthetic identity affine).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
