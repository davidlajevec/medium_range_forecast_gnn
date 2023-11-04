import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # first layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.05, training=self.training)

        # second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.05, training=self.training)

        # third layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.05, training=self.training)

        # output layer
        x = self.conv4(x, edge_index)
        return x

if __name__ == "__main__":
    x = torch.randn(10, 3)  # 10 nodes, 16 features per node
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 4, 6, 8],
                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 3, 4, 5, 7, 9]], dtype=torch.long)

    model = GCN(in_channels=3, hidden_channels=16, out_channels=3)

    output = model(x, edge_index)
    print(output.shape)  # torch.Size([10, 3])