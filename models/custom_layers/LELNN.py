from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
import torch.nn.init as init

class CustomGraphLayer(MessagePassing):
    def __init__(self, in_channels, edge_in_channels, out_channels, num_neighbors=8, non_linearity=nn.ReLU()):
        super(CustomGraphLayer, self).__init__(aggr='add')  # 'add' aggregation.
        self.num_neighbors = num_neighbors
        self.in_channels = in_channels
        self.edge_in_channels = edge_in_channels
        self.out_channels = out_channels
        # Neural Network for node feature transformation
        self.node_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            non_linearity
        )

        # Neural Network for first aggregation layer
        self.edge_nn = nn.Sequential(
            nn.Linear(out_channels + edge_in_channels, out_channels//2),
            non_linearity
        )

        self.aggregate_nn = nn.Sequential(
            nn.Linear(out_channels//2*num_neighbors, out_channels)
        )

        self.init_weights()

    def init_weights(self):
        # Iterate over all modules in the layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use He initialization for ReLU (or similar non-linearities)
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

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
    
    def aggregate2(self, inputs, index, dim_size):
        # Custom aggregation: Efficient concatenation of messages for fixed neighbor count
        concatenated_size = self.out_channels // 2 * self.num_neighbors
        # Create a tensor to gather all neighbor features
        all_neighbors = torch.zeros((dim_size, self.num_neighbors, self.out_channels // 2), device=inputs.device)
        # Use scatter_add to efficiently gather neighbor features for each node
        # Note: This assumes that the inputs are already ordered by the index
        all_neighbors = all_neighbors.view(-1, self.out_channels // 2)
        print(inputs[index==0], "inputs.shape")
        print(inputs.shape, "inputs.shape")
        all_neighbors.scatter_add_(0, index.repeat_interleave(self.out_channels // 2).view(-1, self.out_channels // 2), inputs)
        print(all_neighbors.shape, "all_neighbors.shape")
        print(all_neighbors[index==0], "all_neighbors")
        # Reshape and concatenate the neighbor features for each node
        concatenated = all_neighbors.view(dim_size, concatenated_size)
        print(concatenated.shape, "concatenated.shape")
        print(concatenated[0], "concatenated")
        #return self.aggregate_nn(concatenated)
        return
    
    def aggregate(self, inputs, index, dim_size):
        concatenated_size = self.out_channels // 2 * self.num_neighbors
        
        print(f'Initial inputs shape: {inputs.shape}')
        print(f'Index shape: {index.shape}')
        print(f'Dim size: {dim_size}')

        # Reshape inputs to [E, num_neighbors, out_channels // 2]
        reshaped_inputs = inputs.view(-1, self.num_neighbors, self.out_channels // 2)
        
        print(f'Reshaped inputs shape: {reshaped_inputs.shape}')
        print(reshaped_inputs[0], "reshaped_inputs")
        # Create an expanded index that maps each set of neighbors to the correct node
        #expanded_index = index.unsqueeze(-1).expand(-1, self.num_neighbors).reshape(-1)
        expanded_index = index[::self.num_neighbors] 
        print(f'Expanded index shape: {expanded_index.shape}')
        # Use index_select to gather the reshaped inputs based on the expanded index
        selected_inputs = torch.index_select(reshaped_inputs, 0, expanded_index)
        print(f'Selected inputs shape: {selected_inputs.shape}')
        # Reshape selected_inputs to the size of [dim_size, concatenated_size]
        concatenated = selected_inputs.view(dim_size, concatenated_size)
        print(f'Concatenated shape: {concatenated.shape}')
        print(concatenated[0], "concatenated")
        #return self.aggregate_nn(concatenated)
        return


    def update(self, aggr_out: Tensor):
        return aggr_out