import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.utils.visualization import per_node_dominant_modality, pooled_modality_query_importance

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "gen_dom", _ROOT / "scripts" / "generate_dominance_atlas.py"
)
assert _spec and _spec.loader
_gd = importlib.util.module_from_spec(_spec)
sys.modules["gen_dom"] = _gd
_spec.loader.exec_module(_gd)


def test_per_node_dominant_rowsum():
    m = 3
    a = np.zeros((4, m, m), dtype=np.float32)
    a[0, 0, :] = 1.0  # node 0: row 0 wins
    a[1, 1, :] = 1.0
    a[2, 2, :] = 1.0
    a[3, :, :] = 0.0
    out = per_node_dominant_modality(a, pool="rowsum")
    assert out[0] == 1 and out[1] == 2 and out[2] == 3
    assert out[3] == 0.0


def test_subject_id_variants_match():
    a = _gd._subject_id_variants("sub-7")
    b = _gd._subject_id_variants("7")
    assert not a.isdisjoint(b)
    assert _gd._ids_match("sub-01", "01")


def test_pooled_unchanged_3d_mean():
    n = 2
    m = 2
    t = np.zeros((m, m))
    t[0, 0] = 1.0
    stack = np.stack([t, t], axis=0)
    v1 = pooled_modality_query_importance(stack, "rowsum")
    v0 = pooled_modality_query_importance(t, "rowsum")
    np.testing.assert_allclose(v1, v0, rtol=1e-5)
