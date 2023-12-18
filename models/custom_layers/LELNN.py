from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import torch.nn.init as init

class CustomGraphLayer(MessagePassing):
    def __init__(self, in_channels, edge_in_channels, out_channels, num_neighbors=20, reduction_multiplier=2, non_linearity=nn.ReLU()):
        super(CustomGraphLayer, self).__init__()
        self.num_neighbors = num_neighbors
        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        self.reduction_multiplier = reduction_multiplier
        
        self.node_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            #nn.Dropout(p=0.3),
            non_linearity,
            #nn.Linear(out_channels, out_channels),
            #nn.BatchNorm1d(out_channels),
            #non_linearity,
        )

        # Neural Network for first aggregation layer
        self.edge_nn = nn.Sequential(
            nn.Linear(out_channels + edge_in_channels, out_channels//reduction_multiplier),
            nn.BatchNorm1d(out_channels//reduction_multiplier),
            #nn.Dropout(p=0.3),
            non_linearity,
            #nn.Linear(out_channels//reduction_multiplier, out_channels//reduction_multiplier),
            #nn.BatchNorm1d(out_channels//reduction_multiplier),
            #non_linearity
        )

        self.aggregate_nn = nn.Sequential(
            nn.Linear(out_channels//reduction_multiplier*num_neighbors, out_channels), 
            nn.BatchNorm1d(out_channels),
            non_linearity,
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            non_linearity,
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        # x: Node features [N, in_channels]
        # edge_index: Graph connectivity [2, E]
        # edge_attr: Edge features [E, edge_in_channels]

        # Step 1: Node feature transformation
        x = self.node_nn(x)

        # Step 2: Propagate the messages
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)

        return out

    def message(self, x_i, x_j, edge_attr):
        # x_i: Source node features [E, out_channels]
        # x_j: Target node features [E, out_channels]
        # edge_attr: Edge features [E, edge_in_channels]
        # Combine node features with edge attributes
        tmp = torch.cat([x_j, edge_attr], dim=1) 
        return self.edge_nn(tmp)
    
    def aggregate(self, inputs, index, dim_size):
        concatenated_size = self.out_channels // self.reduction_multiplier * self.num_neighbors
        sorted_indices = torch.argsort(index)
        inputs_sorted = inputs[sorted_indices]
        grouped_inputs = torch.stack(torch.split(inputs_sorted, self.num_neighbors))
        reshaped_inputs = grouped_inputs.view(-1, self.num_neighbors, self.out_channels // self.reduction_multiplier)
        concatenated = reshaped_inputs.view(dim_size, concatenated_size)
        return self.aggregate_nn(concatenated)


    def update(self, aggr_out: Tensor):
        return aggr_out