import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
#from .custom_layers.CustomUNetLayer import CustomGraphLayer
from .custom_layers.LEL4MBN import CustomGraphLayer

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features, depth=0):
        super(GNN, self).__init__()

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Creating encoder layers
        self.down_convs = nn.ModuleList()
        in_channels = node_in_features
        for i in range(depth):
            # Encoder
            self.down_convs.append(CustomGraphLayer(in_channels, edge_in_features, hidden_channels))
            in_channels = hidden_channels
            hidden_channels *= 2

        # Bottleneck
        self.bottleneck = CustomGraphLayer(in_channels, edge_in_features, hidden_channels)

        # Resetting hidden_channels for the decoder
        hidden_channels //= 2

        for i in range(depth):
            # Decoder
            self.up_convs.append(CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels))
            hidden_channels //= 2  # Reducing for the next layer

        self.final_layer = CustomGraphLayer(hidden_channels * 2, edge_in_features, out_features) # Notice the adjustment here


    def forward(self, x, edge_index, edge_attr):
        skip_connections = []
        #print("sth")
        # Encoder path
        for conv in self.down_convs:
            print(f"shape before:{x.shape}")
            x = conv(x, edge_index, edge_attr)
            print(f"shape after:{x.shape}")
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, edge_index, edge_attr)

        # Decoder path with skip connections
        for conv, skip in zip(self.up_convs, reversed(skip_connections)):
            x = torch.cat([x, skip], dim=1)  # Skip connection
            x = conv(x, edge_index, edge_attr)

        x = self.final_layer(x, edge_index, edge_attr)

        return x


# class GNN(nn.Module):
#     def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
#         super(GNN, self).__init__()

#         # Encoder
#         self.down_conv1 = CustomGraphLayer(node_in_features, edge_in_features, hidden_channels)
#         self.down_conv2 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
#         # ... more layers as needed ...

#         # Decoder
#         self.up_conv1 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
#         self.up_conv2 = CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels) # Note the doubling due to skip connections
#         # ... more layers as needed ...

#         self.final_layer = CustomGraphLayer(hidden_channels, edge_in_features, out_features)

#     def forward(self, x, edge_index, edge_attr):
#         # Encoder path
#         x1 = self.down_conv1(x, edge_index, edge_attr)
#         x2 = self.down_conv2(x1, edge_index, edge_attr)
#         # ... more layers as needed ...

#         # Decoder path with skip connections
#         x = self.up_conv1(x2, edge_index, edge_attr)
#         x = torch.cat([x, x2], dim=1) # Skip connection
#         x = self.up_conv2(x, edge_index, edge_attr)
#         # ... more layers as needed ...

#         x = self.final_layer(x, edge_index, edge_attr)

#         return x
