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

        # Creating encoder layers with corresponding pooling layers
        in_channels = node_in_features
        
        self.down_convs.append(CustomGraphLayer(in_channels, edge_in_features, hidden_channels))
        self.down_pools.append(TopKPooling(hidden_channels, ratio=0.5))
        in_channels = hidden_channels
        hidden_channels *= 2  # Double the number of channels after each layer

        # Bottleneck without pooling
        self.bottleneck = CustomGraphLayer(in_channels, edge_in_features, hidden_channels)

        # Resetting hidden_channels for the decoder
        hidden_channels //= 2

        # Creating decoder layers
        for i in range(depth):
            self.up_convs.append(CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels))
            hidden_channels //= 2  # Halve the number of channels before each layer

        # Final layer to produce output features
        self.final_layer = CustomGraphLayer(hidden_channels * 2, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr, batch=None):
        skip_connections = []

        # Encoder path with pooling
        for conv, pool in zip(self.down_convs, self.down_pools):
            x = conv(x, edge_index, edge_attr)
            skip_connections.append(x)
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch=batch)  # Corrected unpacking here

        # Bottleneck (no pooling)
        x = self.bottleneck(x, edge_index, edge_attr)

        # Decoder path with skip connections
        for conv, skip in zip(self.up_convs, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=1)
            x = conv(x, edge_index, edge_attr)

        # Final layer to produce output features
        x = self.final_layer(x, edge_index, edge_attr)

        return x
