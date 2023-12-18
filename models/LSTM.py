import torch
import torch.nn as nn
from .custom_layers.LEL4MBN import CustomGraphLayer

class GraphConvLSTMCell(nn.Module):
    def __init__(self, node_in_features, hidden_channels, edge_in_features, non_linearity=nn.ReLU()):
        super(GraphConvLSTMCell, self).__init__()

        combined_dim = node_in_features + hidden_channels

        self.gcn_i = CustomGraphLayer(combined_dim, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.gcn_f = CustomGraphLayer(combined_dim, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.gcn_c = CustomGraphLayer(combined_dim, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.gcn_o = CustomGraphLayer(combined_dim, edge_in_features, hidden_channels, non_linearity=non_linearity)

    def forward(self, x, edge_index, edge_attr, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)

        i = torch.sigmoid(self.gcn_i(combined, edge_index, edge_attr))
        f = torch.sigmoid(self.gcn_f(combined, edge_index, edge_attr))
        c = f * c_prev + i * torch.tanh(self.gcn_c(combined, edge_index, edge_attr))
        o = torch.sigmoid(self.gcn_o(combined, edge_index, edge_attr))

        h = o * torch.tanh(c)

        return h, c

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features, non_linearity=nn.ReLU()):
        super(GNN, self).__init__()

        self.hidden_channels = hidden_channels  # Add this line
        self.num_layers = out_features  # Assuming out_features represents the number of layers

        self.layers = nn.ModuleList([
            GraphConvLSTMCell(node_in_features if i == 0 else hidden_channels, hidden_channels, edge_in_features, non_linearity=non_linearity)
            for i in range(self.num_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        h, c = [None] * self.num_layers, [None] * self.num_layers

        for i in range(self.num_layers):
            h[i], c[i] = (torch.zeros(x.size(0), self.hidden_channels).to(x.device),
                          torch.zeros(x.size(0), self.hidden_channels).to(x.device))

        for i, layer in enumerate(self.layers):
            h[i], c[i] = layer(x, edge_index, edge_attr, h[i], c[i])
            x = h[i]

        return x

