import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("skimage")

_ROOT = Path(__file__).resolve().parents[1]
_slic_path = _ROOT / "src" / "preprocessing" / "slic_supervoxels.py"
_spec = importlib.util.spec_from_file_location("slic_supervoxels_test", _slic_path)
assert _spec and _spec.loader
_slic = importlib.util.module_from_spec(_spec)
sys.modules["slic_supervoxels_test"] = _slic
_spec.loader.exec_module(_slic)

approximate_lr_masks_from_x_mid = _slic.approximate_lr_masks_from_x_mid
run_slic_with_midline_masks = _slic.run_slic_with_midline_masks
SlicNiftiConfig = _slic.SlicNiftiConfig


def test_midline_slic_3d_separates_hemispheres():
    z, y, x = 12, 12, 12
    rng = np.random.default_rng(0)
    vol = rng.random((z, y, x), dtype=np.float32)
    left, right = approximate_lr_masks_from_x_mid((z, y, x), x_mid=5)
    assert not np.any(left & right)
    cfg = SlicNiftiConfig(n_segments=40, compactness=0.2, multichannel=False)
    lab = run_slic_with_midline_masks(vol, left, right, cfg)
    assert lab.shape == (z, y, x)
    # Labels on left should not appear on right (unique runs per side)
    u_left = set(np.unique(lab[left]))
    u_right = set(np.unique(lab[right]))
    u_left.discard(0)
    u_right.discard(0)
    assert u_left.isdisjoint(u_right)


def test_midline_masks_overlap_raises():
    z, y, x = 8, 8, 8
    left = np.ones((z, y, x), dtype=bool)
    right = np.ones((z, y, x), dtype=bool)
    with pytest.raises(ValueError, match="overlap"):
        run_slic_with_midline_masks(
            np.zeros((z, y, x), dtype=np.float32),
            left,
            right,
            SlicNiftiConfig(n_segments=8, multichannel=False),
        )
