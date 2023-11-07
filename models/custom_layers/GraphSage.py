import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import torch_scatter
import torch.nn as nn

class GraphSage(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = None
        self.lin_r = None

        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        out = None

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        agg = self.propagate(edge_index, x=(x, x))
        out = self.lin_r(agg) + self.lin_l(x)
        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)

        return out

    def message(self, x_j):

        out = None

        out = x_j

        return out

    def aggregate(self, inputs, index, dim_size = None):


        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="mean")

        return out
