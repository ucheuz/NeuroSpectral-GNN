import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.preprocessing.registration import align_modalities, verify_mni152_space


def test_nilearn_resample_align_moving_to_fixed(tmp_path: Path) -> None:
    pytest.importorskip("nilearn")
    p_fix = tmp_path / "f.nii.gz"
    p_mov = tmp_path / "m.nii.gz"
    aff = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), np.float32), aff), str(p_fix))
    nib.save(
        nib.Nifti1Image(
            np.random.default_rng(0).standard_normal((20, 20, 20)).astype(np.float32),
            aff,
        ),
        str(p_mov),
    )
    p_out = align_modalities(
        p_mov, p_fix, tmp_path / "o.nii.gz", method="nilearn_resample"
    )
    assert p_out.is_file()


def test_mni152_verification_self() -> None:
    pytest.importorskip("nilearn")
    from nilearn.datasets import load_mni152_brain_mask

    ref = load_mni152_brain_mask(resolution=2)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mni.nii.gz"
        nib.save(ref, str(p))
        ok, msg = verify_mni152_space(p, resolution_mm=2, affine_atol=0.01)
    assert ok, msg
