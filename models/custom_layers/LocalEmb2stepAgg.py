import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Linear

class CustomGNN(MessagePassing):
    """
    Custom Graph Neural Network using PyTorch Geometric's MessagePassing class.

    This class implements a graph neural network that performs message passing and aggregation
    to generate node embeddings. It is designed to handle graphs with varying node connectivity
    and can incorporate edge attributes.

    Parameters:
    - node_in_features (int): The number of features (dimensions) for each node in the graph. 
      This represents the size of the input feature vector for each node. 

    - edge_in_features (int): The number of features (dimensions) for each edge in the graph.
      This is the size of the input feature vector for each edge. Edge features represent 
      information about the connection or relationship between two nodes. 

    - out_features (int): The size of the output feature vector for each node after processing
      by the network. This is the dimensionality of the node features in the transformed space.

    The model includes a linear transformation for both node and edge features, and follows 
    a two-step aggregation process to update the node embeddings.
    """
    def __init__(self, node_in_features, edge_in_features, out_features, normalize = True,
                 bias = False):
        super(CustomGNN, self).__init__(aggr='add')  # 'add' aggregation.
        self.node_in_features=node_in_features
        self.edge_in_features=edge_in_features
        self.out_features=out_features
        self.normalize=normalize

        self.lin = Linear(self.node_in_features, self.out_features,bias=bias)
        self.edge_nn = Linear(self.edge_in_features, self.out_features,bias=bias)
        self.aggr_nn = Linear(self.out_features, self.out_features,bias=bias)
        self.out = Linear(self.out_features, self.out_features,bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.edge_nn.reset_parameters()
        self.aggr_nn.reset_parameters()
        self.out.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        out = self.out(out) + self.lin(x)  # Skip connection: adding the original features

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)  # L2 normalization

        return out

    def message(self, x_j, edge_attr):
        # Message: Node feature passed through a linear layer
        msg = self.lin(x_j)
        if edge_attr is not None:
            # Process edge attributes if available
            msg += self.edge_nn(edge_attr)
        return msg

    def update(self, aggr_out, x):
        # Step 2: Aggregation of all neighbor information
        aggr_out = self.aggr_nn(aggr_out)
        return aggr_out + self.lin(x)  # Optional skip-connection
    
#---------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Example node features: [u, v, scalar]
    node_features = torch.tensor([
        [1.0, 0.5, 0.3],
        [0.7, 1.2, 0.4],
        [0.6, 0.9, 0.5],
        [1.1, 0.4, 0.2],
        [0.8, 0.8, 0.3]
    ], dtype=torch.float)

    # Example edge index (5 nodes, some arbitrary connections)
    edge_index = torch.tensor([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 0],
        [1, 3]
    ], dtype=torch.long).t().contiguous()

    # Edge attributes: relative positions (example)
    edge_attributes = torch.tensor([
        [1, 0],
        [1, 1],
        [1, 1],
        [-1, -1],
        [-1, 0],
        [2, 1]
    ], dtype=torch.float)

    # Create a PyTorch Geometric data object
    from torch_geometric.data import Data
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attributes)

    # Initialize your CustomGNN
    model = CustomGNN(node_in_features=3, edge_in_features=2, out_features=3)

    # Forward pass
    output = model(data.x, data.edge_index, data.edge_attr)

    print(output)