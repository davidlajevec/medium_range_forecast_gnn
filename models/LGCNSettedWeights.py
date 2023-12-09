from .custom_layers.LocalEmbeddingLayerM import CustomGraphLayer
import torch
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
        super(GNN, self).__init__()

        self.layer1 = CustomGraphLayer(node_in_features, edge_in_features, hidden_channels)

        self.layer2 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer3 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer4 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer5 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer_last = CustomGraphLayer(hidden_channels, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        
        x1 = self.layer1(x, edge_index, edge_attr)

        x2 = self.layer2(x1, edge_index, edge_attr)

        x3 = self.layer3(x2, edge_index, edge_attr)

        x4 = self.layer4(x3, edge_index, edge_attr)

        x5 = self.layer4(x4, edge_index, edge_attr)
    
        x_skip = x1 + x2/2. + x3/3. + x4/4. + x5/5.

        x = self.layer_last(x_skip, edge_index, edge_attr)

        return x

