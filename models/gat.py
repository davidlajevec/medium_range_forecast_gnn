import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.0):
        super(GAT, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        x = self.lin(x).view(-1, self.heads, self.out_channels)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x_i.size(0))
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = self.softmax(alpha, edge_index)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        return aggr_out

    def softmax(self, alpha, edge_index):
        row, col = edge_index
        alpha = alpha - alpha.max(dim=1, keepdim=True)[0]
        alpha = alpha.exp()
        alpha_sum = degree(row, alpha, dtype=alpha.dtype)
        alpha_sum[alpha_sum == 0] = 1
        return alpha / alpha_sum[row]