import torch
from torch_geometric.data import Dataset, Data
import numpy as np

class TwinBrainDataset(Dataset):
    def __init__(self, root, twin_pairs_list, transform=None, pre_transform=None):
        """
        twin_pairs_list: A list of tuples/dicts containing paths to Twin A and Twin B scans
                         and their label (0 for MZ, 1 for Unrelated/DZ).
        """
        super(TwinBrainDataset, self).__init__(root, transform, pre_transform)
        self.twin_pairs = twin_pairs_list

    def len(self):
        return len(self.twin_pairs)

    def get(self, idx):
        # 1. Load the pre-processed Adjacency Matrices for Twin A and Twin B
        # In the real project, you'll load the .npy files you saved earlier
        pair = self.twin_pairs[idx]
        
        # This is where your 'nilearn' logic will eventually sit
        # For now, we return the structure the Siamese model expects
        data_a = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 300)))
        data_b = Data(x=torch.randn(100, 16), edge_index=torch.randint(0, 100, (2, 300)))
        
        label = torch.tensor([pair['label']], dtype=torch.float)
        
        return data_a, data_b, label