# LEL - Local Embedding Layer
from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops

class CustomGraphLayer(MessagePassing):
    def __init__(self, 
        in_channels, 
        edge_in_channels, 
        out_channels,
        non_linearity=nn.ReLU(),
        ):
        super(CustomGraphLayer, self).__init__()  

        # Neural Network for node feature transformation
        self.node_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            non_linearity
        )

        self.node_to_agg_nn = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            non_linearity
        )

        # Neural Network for first aggregation layer
        self.edge_nn = nn.Sequential(
            nn.Linear(out_channels + edge_in_channels, out_channels),
            non_linearity
        )

        self.aggregate_nn = nn.Sequential(
            nn.Linear(out_channels*3+out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):        
        # x: Node features [N, in_channels]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge features [E, edge_in_channels]

        # Step 1: Node feature transformation
        x = self.node_nn(x)

        # Step 2: First aggregation layer
        agg1 = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        # Step 3: Second aggregation layer
        x_agg = torch.cat([x, agg1], dim=1)

        agg2 = self.aggregate_nn(x_agg)

        return agg2

    def message(self, x_i, x_j, edge_attr):
        # x_i: Source node features [E, out_channels]
        # x_j: Target node features [E, out_channels]
        # edge_attr: Edge features [E, edge_in_channels]
        
        # Combine node features with edge attributes
        tmp = torch.cat([x_j, edge_attr], dim=1) 
        return self.edge_nn(tmp)

    def aggregate(self, inputs: Tensor, index: Tensor, ptr: Tensor | None = None, dim_size: int | None = None) -> Tensor:
        mean_aggr = scatter(inputs, index, dim=self.node_dim, reduce='mean')
        min_aggr = scatter(inputs, index, dim=self.node_dim, reduce='min')
        max_aggr = scatter(inputs, index, dim=self.node_dim, reduce='max')
        return torch.cat([mean_aggr, max_aggr, min_aggr], dim=1)    

    def update(self, aggr_out: Tensor):
        return aggr_out