from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class CustomGraphLayer(MessagePassing):
    def __init__(self, 
        in_channels, 
        edge_in_channels, 
        out_channels,
        non_linearity=nn.ReLU(),
        dropout_rate=0.2,
        batch_norm=True):
        super(CustomGraphLayer, self).__init__()  

        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(dropout_rate)

        # Neural Network for node feature transformation
        self.node_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2) if self.batch_norm else nn.Identity(),
            non_linearity,
            self.dropout,
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2) if self.batch_norm else nn.Identity(),
            non_linearity,
            self.dropout
        )

        # Neural Network for first aggregation layer
        self.edge_nn = nn.Sequential(
            nn.Linear(out_channels // 2 + edge_in_channels, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2) if self.batch_norm else nn.Identity(),
            non_linearity,
            self.dropout,
            nn.Linear(out_channels // 2, out_channels // 2),
            nn.BatchNorm1d(out_channels // 2) if self.batch_norm else nn.Identity(),
            non_linearity,
            self.dropout
        )

        self.aggregate_nn = nn.Sequential(
            nn.Linear(out_channels//2*4, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: Node features [N, in_channels]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge features [E, edge_in_channels]

        # Step 1: Node feature transformation
        x = self.node_nn(x)

        # Step 2: First aggregation layer
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        # Step 3: Second aggregation layer
        out = self.aggregate_nn(out)

        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i: Source node features [E, out_channels]
        # x_j: Target node features [E, out_channels]
        # edge_attr: Edge features [E, edge_in_channels]
        
        # Combine node features with edge attributes
        tmp = torch.cat([x_j, edge_attr], dim=1) 
        return self.edge_nn(tmp)

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:
        mean_aggr = scatter(inputs, index, dim=self.node_dim, reduce='mean')
        squared_diff = (inputs - mean_aggr[index])**2
        variance_aggr = scatter(squared_diff, index, dim=self.node_dim, reduce='mean')
        std_aggr = torch.sqrt(variance_aggr)
        min_aggr = scatter(inputs, index, dim=self.node_dim, reduce='min')
        max_aggr = scatter(inputs, index, dim=self.node_dim, reduce='max')
        return torch.cat([mean_aggr, max_aggr, min_aggr, std_aggr], dim=1)    

    def update(self, aggr_out: Tensor):
        return aggr_out