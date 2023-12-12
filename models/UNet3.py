import torch
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from .custom_layers.LEL4MBN import CustomGraphLayer

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features, depth=1):
        super(GNN, self).__init__()

        self.down_convs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        in_channels = node_in_features
        for i in range(depth):
            self.down_convs.append(CustomGraphLayer(in_channels, edge_in_features, hidden_channels))
            self.down_pools.append(TopKPooling(hidden_channels, ratio=0.5))
            in_channels = hidden_channels
            hidden_channels *= 2

        self.bottleneck = CustomGraphLayer(in_channels, edge_in_features, hidden_channels)
        hidden_channels //= 2

        for i in range(depth):
            self.up_convs.append(CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels))
            hidden_channels //= 2

        self.final_layer = CustomGraphLayer(hidden_channels * 2, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr, batch=None):
        skip_connections = []

        for conv, pool in zip(self.down_convs, self.down_pools):
            x = conv(x, edge_index, edge_attr)
            skip_connections.append(x)
            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr, batch=batch)

        x = self.bottleneck(x, edge_index, edge_attr)

        for conv, skip in zip(self.up_convs, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = conv(x, edge_index, edge_attr)

        x = self.final_layer(x, edge_index, edge_attr)

        return x