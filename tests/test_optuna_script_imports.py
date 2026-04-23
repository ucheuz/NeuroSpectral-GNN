"""Light check that ``scripts/optimize.py`` loads when Optuna is installed."""

import importlib.util
from pathlib import Path

import pytest


def test_optimize_module_loads():
    pytest.importorskip("optuna")
    path = Path(__file__).resolve().parents[1] / "scripts" / "optimize.py"
    spec = importlib.util.spec_from_file_location("optimize_hpo", path)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    assert callable(m.main)


def test_optuna_available():
    optuna = pytest.importorskip("optuna")
    assert optuna.__version__
