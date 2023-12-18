import torch
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from .custom_layers.LEL4MBN import CustomGraphLayer

class UnpoolingLayer(nn.Module):
    def __init__(self):
        super(UnpoolingLayer, self).__init__()

    def forward(self, x, perm, orig_size):
        # `orig_size` should be the original number of nodes. The feature dimension remains the same.
        num_features = x.size(1)
        unpool_x = x.new_zeros(orig_size, num_features)  # Initialize unpool_x with the correct shape
        unpool_x[perm] = x  # Assign x to the correct positions in unpool_x
        return unpool_x

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features, depth=1):
        super(GNN, self).__init__()

        self.down_convs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_unpools = nn.ModuleList() 

        in_channels = node_in_features
        for i in range(depth):
            self.down_convs.append(CustomGraphLayer(in_channels, edge_in_features, hidden_channels))
            self.down_pools.append(TopKPooling(hidden_channels, ratio=0.5))
            in_channels = hidden_channels
            hidden_channels *= 2

        self.bottleneck = CustomGraphLayer(in_channels, edge_in_features, hidden_channels)
        hidden_channels //= 2

        for i in range(depth):
            self.up_unpools.append(UnpoolingLayer())  # Add UnpoolingLayer instances
            self.up_convs.append(CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels))
            hidden_channels //= 2

        self.final_layer = CustomGraphLayer(hidden_channels*4, edge_in_features, out_features)


    def forward(self, x, edge_index, edge_attr, batch=None):
        perm_list = []
        orig_size_list = []
        skip_connections = []  # List to store the skip connections

        # Downward pass (Pooling)
        for conv, pool in zip(self.down_convs, self.down_pools):
            x = conv(x, edge_index, edge_attr)
            skip_connections.append(x)  # Store the result for later skip connection
            
            # Store the current size of x before pooling
            orig_size_list.append(x.size(0))
            
            # Pool and store the permutation indices
            x, edge_index, edge_attr, batch, perm, _ = pool(x, edge_index, edge_attr, batch)
            perm_list.append(perm)

        x = self.bottleneck(x, edge_index, edge_attr)

        # Upward pass (Unpooling)
        for unpool, up_conv, skip in zip(reversed(self.up_unpools), reversed(self.up_convs), reversed(skip_connections)):
            perm = perm_list.pop()
            orig_size = orig_size_list.pop()
            print(f"x before:{x.shape}")
            x = unpool(x, perm, orig_size)  # Unpooling operation
            print(f"x after:{x.shape}")
            x = torch.cat([x, skip], dim=1)  # Concatenating with skip connection
            print(f"x after cat:{x.shape}")
            x = up_conv(x, edge_index, edge_attr)  # Applying the convolution

        x = self.final_layer(x, edge_index, edge_attr)

        return x