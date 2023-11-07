import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GraphConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__(aggr='add')  # 'add' aggregation method

        # Linear transformation for node feature dimensionality reduction
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Compute degree normalization factor
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Apply linear transformation to node features
        x = self.lin(x)

        # Propagate messages and perform aggregation
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # Message function simply passes the normalized features from source nodes (x_j)
        # and multiplies them by the normalization factor (norm)
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # The update function sums the aggregated messages and returns the result
        return aggr_out
