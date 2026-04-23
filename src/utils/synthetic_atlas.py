"""Synthetic 3D label volumes for node ids 1..K (KCL P65 / reverse-map testing).

``int32`` data and ``float32`` affines keep memory low on M1. Pair with
``map_nodes_to_volume`` in ``src.utils.visualization``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    nib = None
    _NIB = e
else:
    _NIB = None


def generate_synthetic_label_volume(
    num_nodes: int,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """Return ``(Z, Y, X) int32`` with labels **1…num_nodes** in block partitions.

    Every voxel is assigned exactly one node id (no background) so
    ``node_values[i-1]`` in the reverse-mapper matches label ``i``.

    Partitions a **3D grid** of sub-boxes (``gz×gy×gx``) so that
    ``gz·gy·gx ≥ num_nodes`` with **as few** cells as possible; ties prefer a
    **nearly cubic** layout (e.g. ``4×4×4`` for 64) so viewers see **patches in
    every slice**, not 1×1×64 “string” grids that look like a smooth gradient in
    Slicer.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    z, y, x = (int(volume_shape[0]), int(volume_shape[1]), int(volume_shape[2]))
    if num_nodes > z * y * x:
        raise ValueError("num_nodes cannot exceed total voxels in volume_shape")

    out = np.zeros((z, y, x), dtype=np.int32)
    # Minimal cell count p≥n, then a **balanced** 3D factorisation (avoids
    # degenerate 1×1×K grids that look like a one-axis ramp in Slicer/FSL).
    best: Optional[Tuple[int, int, int, int, int, int]] = None
    best_key: Optional[Tuple[int, int, int, int, int, int]] = None
    for gz in range(1, z + 1):
        for gy in range(1, y + 1):
            for gx in range(1, x + 1):
                p = gz * gy * gx
                if p < num_nodes:
                    continue
                min_side = min(gz, gy, gx)
                max_vox = max(
                    (z + gz - 1) // gz,
                    (y + gy - 1) // gy,
                    (x + gx - 1) // gx,
                )
                key = (p, -min_side, max_vox, gz, gy, gx)
                if best_key is None or key < best_key:
                    best_key, best = key, (p, -min_side, max_vox, gz, gy, gx)
    if best is None:
        raise ValueError(
            f"Cannot partition {num_nodes} nodes into a sub-grid of {(z, y, x)}; "
            "increase the volume or reduce num_nodes."
        )
    _, _, _, gz, gy, gx = best
    n_cell = 0
    for iz in range(gz):
        for iy in range(gy):
            for ix in range(gx):
                n_cell += 1
                if n_cell > num_nodes:
                    return out
                z0, z1 = (iz * z) // gz, ((iz + 1) * z) // gz
                y0, y1 = (iy * y) // gy, ((iy + 1) * y) // gy
                x0, x1 = (ix * x) // gx, ((ix + 1) * x) // gx
                z1 = max(z1, z0 + 1)
                y1 = max(y1, y0 + 1)
                x1 = max(x1, x0 + 1)
                out[z0:z1, y0:y1, x0:x1] = n_cell
    return out


def _default_affine() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def write_label_nifti(
    labels: np.ndarray,
    output_path: str | Path,
    *,
    affine: Optional[np.ndarray] = None,
) -> Path:
    if nib is None:  # pragma: no cover
        raise ImportError("nibabel is required for NIfTI I/O") from _NIB
    if affine is None:
        affine = _default_affine()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(
        np.asarray(labels, dtype=np.int32), np.asarray(affine, dtype=np.float32)
    )
    img.set_data_dtype(np.int32)
    nib.save(img, str(output_path))
    return output_path.resolve()


def generate_synthetic_atlas_nifti(
    num_nodes: int,
    volume_shape: Tuple[int, int, int] = (64, 64, 64),
    output_path: str | Path = "synthetic_node_labels.nii.gz",
    *,
    affine: Optional[np.ndarray] = None,
) -> Path:
    """Convenience: build the label volume and write ``.nii.gz``; returns path."""
    d = generate_synthetic_label_volume(num_nodes, volume_shape=volume_shape)
    return write_label_nifti(d, output_path, affine=affine)
