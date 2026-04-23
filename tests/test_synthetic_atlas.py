import numpy as np
import pytest

from src.utils.synthetic_atlas import (
    generate_synthetic_label_volume,
    generate_synthetic_atlas_nifti,
)


def test_slab_partition_matches_num_nodes():
    d = generate_synthetic_label_volume(8, (8, 8, 8))
    assert d.shape == (8, 8, 8)
    assert int(d.max()) == 8
    assert set(np.unique(d)) == set(range(1, 9))


def test_3d_grid_high_node_count():
    d = generate_synthetic_label_volume(50, (20, 20, 20))
    assert int(d.max()) == 50


def test_nifti_roundtrip(tmp_path):
    nib = pytest.importorskip("nibabel")
    p = generate_synthetic_atlas_nifti(4, (8, 8, 8), tmp_path / "l.nii.gz")
    a = np.asarray(nib.load(str(p)).dataobj, dtype=np.int32)
    assert a.max() == 4
