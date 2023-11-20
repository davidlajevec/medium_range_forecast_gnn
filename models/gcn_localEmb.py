import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from .custom_layers.LocalEmb2stepAgg import CustomGNN

class CustomGraphNetwork(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
        super(CustomGraphNetwork, self).__init__()
        self.custom_gnn1 = CustomGNN(node_in_features, edge_in_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        # If you have more layers, you can add them similarly.
        self.custom_gnn_last = CustomGNN(hidden_channels, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        # First layer
        x = self.custom_gnn1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.05, training=self.training)
        # ... (if you have additional layers, process them similarly) ...
        # Last layer
        x = self.custom_gnn_last(x, edge_index, edge_attr)
        return x

if __name__ == "__main__":
    # Example inputs
    x = torch.randn(10, 3)  # 10 nodes, 3 features per node
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3, 4, 5, 7, 9]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 2)  # Example edge attributes

    # Initialize your CustomGraphNetwork
    model = CustomGraphNetwork(node_in_features=3, edge_in_features=2, hidden_channels=16, out_features=3)

    # Forward pass
    output = model(x, edge_index, edge_attr)
    print(output.shape)  # Expected output shape
