from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

class CustomGraphLayer(MessagePassing):
    def __init__(self, in_channels, edge_in_channels, out_channels):
        super(CustomGraphLayer, self).__init__(aggr='add')  # 'add' aggregation.

        # Neural Network for node feature transformation
        self.node_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )

        # Neural Network for first aggregation layer
        self.edge_nn = nn.Sequential(
            nn.Linear(out_channels + edge_in_channels, out_channels),
            nn.ReLU()
        )

        self.aggregate_nn = nn.Sequential(
            nn.Linear(out_channels*3, out_channels)
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
    
    # CORRECT AGGREGATE BUT PROBLEM WITH MESH
    def aggregate(self, inputs: Tensor, index: Tensor):
        node_count = index.max() + 1
        aggr_out = []
        for node_idx in range(node_count):
            # Gather all messages for this node
            messages = inputs[index == node_idx]
            # Ensure that we have exactly 8 messages per node
            if messages.size(0) != 8:
                raise ValueError(f"Node {node_idx} does not have exactly 8 messages. Found {messages.size(0)} messages.")
            # Reshape and concatenate the messages
            concatenated_messages = messages.view(-1)
            aggr_out.append(concatenated_messages)
        # Stack all node message tensors
        aggr_out = torch.stack(aggr_out, dim=0)
        # Pass the concatenated messages through the linear layer
        return self.aggregate_nn(aggr_out)

    def update(self, aggr_out: Tensor):
        return aggr_out