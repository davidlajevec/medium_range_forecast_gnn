

from .custom_layers.CustomUNetLayer import CustomGraphLayer
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# class GNN(nn.Module):
#     def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
#         super(GNN, self).__init__()

#         # Encoder
#         self.down_conv1 = GCNConv(node_in_features, hidden_channels)
#         self.down_conv2 = GCNConv(hidden_channels, hidden_channels)
#         # ... more layers as needed ...

#         # Decoder
#         self.up_conv1 = GCNConv(hidden_channels, hidden_channels)
#         # Note: For simplicity, not doubling the channels here, but you can design more complex structures
#         self.up_conv2 = GCNConv(hidden_channels, hidden_channels)
#         # ... more layers as needed ...

#         self.final_layer = GCNConv(hidden_channels, out_features)

#     def forward(self, x, edge_index):
#         # Encoder path
#         x1 = self.down_conv1(x, edge_index)
#         x2 = self.down_conv2(x1, edge_index)
#         # ... more layers as needed ...

#         # Decoder path with skip connections
#         x = self.up_conv1(x2, edge_index)
#         x = torch.cat([x, x2], dim=1) # Skip connection
#         x = self.up_conv2(x, edge_index)
#         # ... more layers as needed ...

#         x = self.final_layer(x, edge_index)

#         return x


class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
        super(GNN, self).__init__()

        # Encoder
        self.down_conv1 = CustomGraphLayer(node_in_features, edge_in_features, hidden_channels)
        self.down_conv2 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        # ... more layers as needed ...

        # Decoder
        self.up_conv1 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.up_conv2 = CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels) # Note the doubling due to skip connections
        # ... more layers as needed ...

        self.final_layer = CustomGraphLayer(hidden_channels, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        # Encoder path
        x1 = self.down_conv1(x, edge_index, edge_attr)
        x2 = self.down_conv2(x1, edge_index, edge_attr)
        # ... more layers as needed ...

        # Decoder path with skip connections
        x = self.up_conv1(x2, edge_index, edge_attr)
        x = torch.cat([x, x2], dim=1) # Skip connection
        x = self.up_conv2(x, edge_index, edge_attr)
        # ... more layers as needed ...

        x = self.final_layer(x, edge_index, edge_attr)

        return x
