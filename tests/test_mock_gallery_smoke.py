"""Smoke: mock gallery script runs and writes outputs."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="paths only checked on unix layout",
)
def test_generate_mock_gallery(tmp_path):
    nib = pytest.importorskip("nibabel")
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_mock_gallery.py"),
            "--output-dir",
            str(tmp_path),
            "--n-nodes",
            "16",
            "--device",
            "cpu",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "mock_gallery_saliency.png").exists()
    assert (tmp_path / "mock_node_labels.nii.gz").exists()
    assert (tmp_path / "mock_saliency_mz_high.nii.gz").exists()
    # nib can load
    nib.load(str(tmp_path / "mock_saliency_mz_high.nii.gz"))
