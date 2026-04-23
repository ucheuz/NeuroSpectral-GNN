import torch
from torch_geometric.data import Data, Batch

from src.utils.brain_dataset import twin_collate


def test_twin_collate_shapes():
    d1a = Data(x=torch.zeros(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    d1b = Data(x=torch.zeros(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    d2a = Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    d2b = Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 2]]))
    m1 = {"family_id": "F1", "zygosity": "MZ", "label": 0, "subject_a": "A", "subject_b": "B"}
    m2 = {"family_id": "F2", "zygosity": "DZ", "label": 1, "subject_a": "C", "subject_b": "D"}
    batch = twin_collate([(d1a, d1b, m1), (d2a, d2b, m2)])
    assert batch.label.shape == (2,)
    assert len(batch.zygosities) == 2
    assert batch.data_a.x.shape[0] == 6
    assert batch.data_b.x.shape[0] == 6
