from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class CustomGraphLayer(MessagePassing):
    def __init__(self, in_channels, edge_in_channels, out_channels, skip_channels=0):
        super(CustomGraphLayer, self).__init__(aggr='add')  # 'add' aggregation.
        self.skip_channels = skip_channels

        # Adjust input dimension to account for skip connections
        adjusted_in_channels = in_channels + skip_channels

        # Neural Network for node feature transformation
        self.node_nn = nn.Sequential(
            nn.Linear(adjusted_in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_channels, out_channels)
        )

        # Neural Network for edge feature transformation
        self.edge_nn = nn.Sequential(
            nn.Linear(out_channels + edge_in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        # Neural Network for second aggregation layer
        self.aggregate_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, skip_feature=None):
        # If skip connections are provided, concatenate them with the input features
        if self.skip_channels > 0 and skip_feature is not None:
            x = torch.cat([x, skip_feature], dim=1)

        # Node feature transformation
        x = self.node_nn(x)

        # Propagate using the message passing mechanism
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        # Apply the second aggregation layer
        out = self.aggregate_nn(out)

        return out

    def message(self, x_i, x_j, edge_attr):
        # Combine node features with edge attributes
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.edge_nn(tmp)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, reduce='sum')

    def update(self, aggr_out):
        return aggr_out
