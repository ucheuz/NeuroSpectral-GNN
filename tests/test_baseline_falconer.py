import numpy as np
import pytest

from src.analysis.heritability import per_feature_falconer_h2, pearson_twin_phenotype_r


def test_pearson_twin_phenotype_r_perfect():
    x = np.array([1.0, 2.0, 3.0])
    r = pearson_twin_phenotype_r(x, x * 2 + 0.5)
    assert r == pytest.approx(1.0)


def test_per_feature_falconer_h2_in_unit_interval():
    rng = np.random.default_rng(42)
    p_mz, p_dz, d = 25, 30, 8
    a_mz = rng.standard_normal((p_mz, d))
    b_mz = a_mz + rng.normal(0, 0.3, size=(p_mz, d))
    a_dz = rng.standard_normal((p_dz, d))
    b_dz = a_dz + rng.normal(0, 0.9, size=(p_dz, d))
    h2 = per_feature_falconer_h2(a_mz, b_mz, a_dz, b_dz)
    assert h2.shape == (d,)
    assert np.isfinite(h2).all()
    assert (h2 >= 0.0).all() and (h2 <= 1.0).all()
