import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BrainGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(BrainGNN, self).__init__()
        # These are the 'Spectral' layers that use the Graph Laplacian
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer: This turns 100 nodes into 1 single 'Brain Vector'
        # This is vital for comparing two different brains
        x = global_mean_pool(x, batch) 
        return x

class SiameseBrainNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(SiameseBrainNet, self).__init__()
        self.encoder = BrainGNN(num_node_features, hidden_channels)

    def forward(self, data_a, data_b):
        # Pass both twins through the SAME encoder (Weight Sharing)
        out_a = self.encoder(data_a.x, data_a.edge_index, data_a.batch)
        out_b = self.encoder(data_b.x, data_b.edge_index, data_b.batch)

        # Calculate the 'Phenotypic Distance'
        # Identical twins should have a distance close to 0
        distance = F.pairwise_distance(out_a, out_b)
        return distance