import torch
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool as gap
from torch_geometric.utils import to_dense_batch, to_dense_adj

## neki ne dela tuki kul vrjetn

class GraphUNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, pool_ratios):
        super(GraphUNet, self).__init__()
        
        # Assuming pool_ratios is a list of ratios for each pooling step, e.g., [0.5, 0.5]
        self.pool_ratios = pool_ratios

        # Encoder (Downsampling)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=self.pool_ratios[0])

        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.pool2 = TopKPooling(hidden_channels * 2, ratio=self.pool_ratios[1])

        # Bottleneck
        self.bottleneck_conv = GCNConv(hidden_channels * 2, hidden_channels * 4)

        # Decoder (Upsampling)
        self.unpool1 = GraphUnpooling()  # The unpooling layer
        self.deconv1 = GCNConv(hidden_channels * 4, hidden_channels * 2)

        self.unpool2 = GraphUnpooling()  # The unpooling layer
        self.deconv2 = GCNConv(hidden_channels * 2, hidden_channels)

        self.final_conv = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # Store the pooling indices and intermediate x for unpooling later
        pooling_indices = []
        intermediate_xs = []

        # Encoder 1
        x1 = torch.relu(self.conv1(x, edge_index))
        intermediate_xs.append(x1)
        x1, edge_index, _, batch, _, permutation = self.pool1(x1, edge_index, None, batch)
        pooling_indices.append(permutation)

        # Encoder 2
        x2 = torch.relu(self.conv2(x1, edge_index))
        intermediate_xs.append(x2)
        x2, edge_index, _, batch, _, permutation = self.pool2(x2, edge_index, None, batch)
        pooling_indices.append(permutation)

        # Bottleneck
        x3 = torch.relu(self.bottleneck_conv(x2, edge_index))

        # Decoder 1 (Unpool + Conv)
        x3, edge_index = self.unpool1(x3, edge_index, intermediate_xs[-1].size(0), pooling_indices[-1])
        x3 = torch.relu(self.deconv1(x3, edge_index))

        # Decoder 2 (Unpool + Conv)
        x4, edge_index = self.unpool2(x3, edge_index, intermediate_xs[-2].size(0), pooling_indices[-2])
        x4 = torch.relu(self.deconv2(x4, edge_index))

        # Final convolution
        out = self.final_conv(x4, edge_index)

        return out

class GraphUnpooling(torch.nn.Module):
    def __init__(self):
        super(GraphUnpooling, self).__init__()

    def forward(self, x, edge_index, upsampled_size, pooling_indices):
        # Convert pooling indices to long type if they are not already
        pooling_indices = pooling_indices.to(torch.long)

        # Restore the node features to the previous size before pooling
        upsampled_x = x.new_zeros([upsampled_size, x.size(1)])
        upsampled_x[pooling_indices] = x

        # For simplicity, we assume that the edge structure remains the same after unpooling.
        # In a real scenario, the edges would also need to be restored appropriately.
        upsampled_edge_index = edge_index

        return upsampled_x, upsampled_edge_index

if __name__ == "__main__":
    # Example usage:
    # Define the model
    model = GraphUNet(in_channels=3, hidden_channels=64, out_channels=3, pool_ratios=[0.5, 0.5])
    num_nodes = 1000
    num_edges = 3000
    # Example data
    # 'x' is the node feature matrix
    # 'edge_index' is the edge index matrix
    # 'batch' is a vector that maps each node to its respective graph in the batch
    x = torch.randn((num_nodes, 3))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Forward pass
    out = model(x, edge_index, batch)
    print(out.shape)