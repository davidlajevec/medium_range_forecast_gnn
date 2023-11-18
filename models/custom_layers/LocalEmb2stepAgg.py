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
      This represents the size of the input feature vector for each node. For example, if each 
      node in your graph has 10 attributes (like age, location, etc.), node_in_features should be 10.

    - edge_in_features (int): The number of features (dimensions) for each edge in the graph.
      This is the size of the input feature vector for each edge. Edge features represent 
      information about the connection or relationship between two nodes. For instance, if each 
      edge has 5 attributes (like length, type, etc.), edge_in_features should be 5.

    - out_features (int): The size of the output feature vector for each node after processing
      by the network. This is the dimensionality of the node features in the transformed space, 
      which can be used for downstream tasks like classification. For example, if you want the 
      transformed node features to have 20 dimensions, out_features should be set to 20.

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