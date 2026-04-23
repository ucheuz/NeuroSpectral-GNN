"""Linear (rigid/affine) alignment of 3D medical images (KCL P65, TwinUK prep).

* **Resample to fixed** uses the moving image in **world** coordinates, sampled on
  the **fixed** grid (nilearn) — *no* optimisation; for already aligned pairs.

* **Full registration** uses **SimpleITK** (Mattes MI + multi-resolution) to map
  **moving** → **fixed**; output matches the **fixed** header / spacing (so SLIC
  and reverse-mapping stay in T1 space).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

try:
    import nibabel as nib
except ImportError as e:  # pragma: no cover
    nib = None
    _NIB = e
else:
    _NIB = None

try:
    from nilearn import image as nli  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    nli = None  # type: ignore[assignment]


def align_modalities(
    moving_img_path: Union[str, Path],
    fixed_img_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    method: Literal["sitk_rigid", "sitk_affine", "nilearn_resample"] = "sitk_rigid",
    max_iterations: int = 200,
) -> Path:
    """Align ``moving`` to the **space** of ``fixed** and save ``.nii`` / ``.nii.gz``.

    Parameters
    ----------
    method
        * ``sitk_rigid`` — rigid + isotropic **scale** (SITK *Similarity3D*; fast).
        * ``sitk_affine`` — 12-dof **affine** (SITK); heavier.
        * ``nilearn_resample`` — only **resample** to fixed (world-space sampling
          via fixed affine, **no** optimiser). Use if inputs are *already* in the
          same world frame or for smoke tests when SimpleITK is not installed.
    max_iterations
        SITK optimiser cap (tune down on low-RAM systems).

    Returns
    -------
    Path
        Path to the written NIfTI (on-disk pixel type follows ``moving`` where
        possible; SITK uses ``float32`` for registration, then you may cast
        discretely for labels **outside** this routine).
    """
    if nib is None:  # pragma: no cover
        raise ImportError("nibabel is required for NIfTI I/O") from _NIB
    moving_img_path, fixed_img_path = Path(moving_img_path), Path(fixed_img_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "nilearn_resample":
        if nli is None:  # pragma: no cover
            raise ImportError("nilearn is required for nilearn_resample") from None
        mov = nib.load(str(moving_img_path))
        fix = nib.load(str(fixed_img_path))
        res = nli.resample_to_img(mov, fix, interpolation="continuous")
        nib.save(res, str(output_path))
        return output_path.resolve()

    if method in ("sitk_rigid", "sitk_affine"):
        try:
            import SimpleITK as sitk
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SimpleITK is required for sitk_rigid / sitk_affine. "
                "pip install SimpleITK  — or use method='nilearn_resample'."
            ) from e
        return _align_sitk(
            moving_img_path, fixed_img_path, output_path, method, max_iterations, sitk
        )
    raise ValueError(f"Unknown method: {method!r}")


def _align_sitk(
    moving_path: Path,
    fixed_path: Path,
    output_path: Path,
    method: str,
    max_iter: int,
    sitk,
) -> Path:
    fix = sitk.ReadImage(str(fixed_path))
    mov = sitk.ReadImage(str(moving_path))
    fixf = sitk.Cast(fix, sitk.sitkFloat32)
    movf = sitk.Cast(mov, sitk.sitkFloat32)
    if method == "sitk_rigid":
        init = sitk.Similarity3DTransform()
    else:
        init = sitk.AffineTransform(3)
    init = sitk.CenteredTransformInitializer(
        fixf,
        movf,
        init,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.04)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradient(learningRate=1.0, minStep=1e-4, numberOfIterations=max_iter)
    R.SetInitialTransform(init, inPlace=False)
    try:
        R.SetOptimizerScalesFromPhysicalShift()
    except Exception:  # pragma: no cover
        pass
    t_out = R.Execute(fixf, movf)
    resampled = sitk.Resample(
        movf,
        fixf,
        t_out,
        sitk.sitkLinear,
        0.0,
        movf.GetPixelID(),
    )
    sitk.WriteImage(resampled, str(output_path))
    return output_path.resolve()


def verify_mni152_space(
    nifti_path: Union[str, Path],
    *,
    resolution_mm: int = 2,
    affine_atol: float = 0.5,
) -> tuple[bool, str]:
    """QC: check whether a NIfTI is in **(approx.) MNI152** space.

    Compares **affine** to nilearn’s MNI152 **brain mask** at ``resolution_mm``
    (1 or 2). We do **not** require identical array shape (zoom levels differ
    for partial FOV) — the affine encodes the voxel→world map.

    Returns
    -------
    (ok, message)
    """
    if nib is None:  # pragma: no cover
        raise ImportError("nibabel is required for MNI check") from _NIB
    nifti_path = Path(nifti_path)
    subj = nib.load(str(nifti_path))
    if nli is None:  # pragma: no cover
        raise ImportError("nilearn is required for verify_mni152_space") from None
    from nilearn.datasets import load_mni152_brain_mask

    ref = load_mni152_brain_mask(resolution=resolution_mm)
    ref_aff = np.asarray(ref.affine, dtype=np.float64)
    sub_aff = np.asarray(subj.affine, dtype=np.float64)
    if ref_aff.shape != (4, 4) or sub_aff.shape != (4, 4):
        return False, "Invalid 4x4 affines in reference or subject"

    if np.allclose(sub_aff, ref_aff, atol=affine_atol, rtol=0.0):
        return (
            True,
            f"Affine matches MNI152 {resolution_mm}mm reference within atol={affine_atol}mm.",
        )
    dist = float(np.linalg.norm(sub_aff - ref_aff, ord="fro"))
    return (
        False,
        f"Affine differs from MNI152 {resolution_mm}mm template (Frobenius err≈{dist:.3f} mm; "
        f"increase --affine-atol or re-register to MNI).",
    )
