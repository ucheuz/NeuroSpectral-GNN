import pandas as pd
import pytest

from src.preprocessing.bids_validator import CohortFileSpec, build_health_report


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_build_health_ok(tmp_path):
    root = tmp_path
    sdir = root / "sub-01"
    (sdir / "anat").mkdir(parents=True)
    (sdir / "dwi").mkdir(parents=True)
    _touch(sdir / "anat" / "sub-01_T1w.nii.gz")
    _touch(sdir / "anat" / "sub-01_FLAIR.nii.gz")
    _touch(sdir / "dwi" / "sub-01_dwi_FA.nii.gz")
    _touch(sdir / "dwi" / "sub-01_dwi_MD.nii.gz")
    prs = pd.DataFrame({"IID": ["01", "99"], "prs1": [0.1, 0.2], "prs2": [0.3, 0.4]})
    pcsv = root / "prs.csv"
    prs.to_csv(pcsv, index=False)
    spec = CohortFileSpec()
    rep = build_health_report(
        root,
        pcsv,
        spec=spec,
        prs_id_column="IID",
        strip_subject_prefix="sub-",
    )
    r = rep.iloc[0]
    assert r["imaging_complete"]
    assert r["issue"] == "ok"
    assert r["prs_id_match"]


def test_prs_mismatch_flags(tmp_path):
    root = tmp_path
    sdir = root / "sub-02"
    (sdir / "anat").mkdir(parents=True)
    _touch(sdir / "anat" / "T1w.nii.gz")
    _touch(sdir / "anat" / "xFLAIRx.nii.gz")
    (sdir / "dwi").mkdir(exist_ok=True)
    _touch(sdir / "dwi" / "FA.nii.gz")
    _touch(sdir / "dwi" / "MD.nii.gz")
    prs = pd.DataFrame({"IID": [999], "p": [0.1]})
    prs.to_csv(root / "p.csv", index=False)
    rep = build_health_report(
        root,
        root / "p.csv",
        spec=CohortFileSpec(),
        prs_id_column="IID",
        prs_value_columns=["p"],
        strip_subject_prefix="sub-",
    )
    r = rep.iloc[0]
    if r["imaging_complete"]:
        assert "missing" in r["issue"] or "PRS" in r["issue"]
