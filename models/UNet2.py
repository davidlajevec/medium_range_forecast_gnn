import torch
import torch.nn as nn
from torch_geometric.nn import TopKPooling, GCNConv
#from .custom_layers.CustomUNetLayer import CustomGraphLayer
from .custom_layers.LEL4MBN import CustomGraphLayer

def filter_edge_attributes(new_edge_index, edge_attr, old_edge_index):
    # Create a set of tuples for the new edges
    new_edges_set = set(map(tuple, new_edge_index.t().tolist()))

    # Filter the edge attributes
    filtered_edge_attr = []
    for i, edge in enumerate(old_edge_index.t().tolist()):
        if tuple(edge) in new_edges_set:
            filtered_edge_attr.append(edge_attr[i])

    return torch.stack(filtered_edge_attr, dim=0)



# Modify the TopKPooling to also return edge attributes
class CustomTopKPooling(TopKPooling):
    def forward(self, x, edge_index, edge_attr, batch=None, attn=None):
        # Original TopKPooling operation
        x, new_edge_index, _, batch, perm, score = super(CustomTopKPooling, self).forward(x, edge_index, batch=batch, attn=attn)

        # Filter edge attributes
        new_edge_attr = filter_edge_attributes(new_edge_index, edge_attr, edge_index)

        return x, new_edge_index, new_edge_attr, batch, perm, score

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features, depth=1):
        super(GNN, self).__init__()

        self.down_convs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_pools = nn.ModuleList()  # This will store indices for unpooling

        # Creating encoder layers with corresponding pooling layers
        in_channels = node_in_features
        for i in range(depth):
            self.down_convs.append(CustomGraphLayer(in_channels, edge_in_features, hidden_channels))
            self.down_pools.append(CustomTopKPooling(hidden_channels, ratio=0.5))
            in_channels = hidden_channels
            hidden_channels *= 2  # Double the number of channels after each layer

        # Bottleneck without pooling
        self.bottleneck = CustomGraphLayer(in_channels, edge_in_features, hidden_channels)

        # Resetting hidden_channels for the decoder
        hidden_channels //= 2

        # Creating decoder layers with corresponding unpooling indices
        for i in range(depth):
            self.up_convs.append(CustomGraphLayer(hidden_channels * 2, edge_in_features, hidden_channels))
            hidden_channels //= 2  # Halve the number of channels before each layer

        # Final layer to produce output features, no unpooling needed
        self.final_layer = CustomGraphLayer(hidden_channels * 2, edge_in_features, out_features)

    def custom_unpool(self, x, edge_index, edge_attr, skip_idx, batch):
        # Create a tensor to hold the unpooled features, initialized to zeros
        # This tensor has a size of [max_original_node_index + 1, num_features]
        # The +1 is important to accommodate the zero-indexing of nodes
        original_size = skip_idx.max().item() + 1
        num_features = x.size(1)
        unpooled_features = torch.zeros((original_size, num_features), device=x.device)
        
        # Map the pooled features back to their original indices
        unpooled_features[skip_idx] = x

        # Return the unpooled features and the original edge_index
        # Note that this approach assumes that the edge_index does not change during pooling
        #new_edge_attr = reconstruct_edge_attributes(edge_attr, skip_idx, batch)
        return x, edge_index, edge_attr

    def forward(self, x, edge_index, edge_attr, batch=None):
        skip_connections = []
        skip_indices = []  # Store indices for unpooling

        # Encoder path with pooling
        for conv, pool in zip(self.down_convs, self.down_pools):
            x = conv(x, edge_index, edge_attr)
            skip_connections.append(x)
            x, edge_index, edge_attr, batch, _, _ = pool(x, edge_index, edge_attr, batch=batch)
            skip_indices.append(edge_index)  # Save the edge indices after pooling

        # Bottleneck (no pooling)
        x = self.bottleneck(x, edge_index, edge_attr)

        # Decoder path with skip connections
        for conv, skip, skip_idx in zip(self.up_convs, reversed(skip_connections), reversed(skip_indices)):
            x, edge_index, edge_attr = self.custom_unpool(x, edge_index, edge_attr, skip_idx, batch)
            x = torch.cat([x, skip], dim=1)
            x = conv(x, edge_index, edge_attr)

        # Final layer to produce output features
        x = self.final_layer(x, edge_index, edge_attr)

        return x
