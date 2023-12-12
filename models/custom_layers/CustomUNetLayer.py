from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class CustomGraphLayer(MessagePassing):
    def __init__(self, in_channels, edge_in_channels, out_channels,skip_channels=0):
        super(CustomGraphLayer, self).__init__(aggr='add')  # 'add' aggregation.
        self.skip_channels = skip_channels

        # Adjust input dimension to account for skip connections
        adjusted_in_channels = in_channels + skip_channels
        #print(f"adjusted_in_channels:{adjusted_in_channels}")
        #print(f"out_channels:{out_channels}")
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
        #print(f"original x shape:{x.shape}")
        if self.skip_channels > 0 and skip_feature is not None:
            x = torch.cat([x, skip_feature], dim=1)
        #print(f"x shape:{x.shape}")
        # Node feature transformation
        weight_matrix_shape = self.node_nn[0].weight.shape
        #print(f"Weight matrix shape: {weight_matrix_shape}")
        x = self.node_nn(x)

        #print(f"x transformed shape:{x.shape}")
        # Propagate using the message passing mechanism
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        #print(f"out shape:{out.shape}")
        # Apply the second aggregation layer
        out = self.aggregate_nn(out)
        #print(f"out transf shape:{out.shape}")
        return out

    def message(self, x_i, x_j, edge_attr):
        # Combine node features with edge attributes
        tmp = torch.cat([x_j, edge_attr], dim=1)
        return self.edge_nn(tmp)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=self.node_dim, reduce='sum')

    def update(self, aggr_out):
        return aggr_out
