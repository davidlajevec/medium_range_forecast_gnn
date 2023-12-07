from .custom_layers.CustomLocalEmbeddingLayer import CustomGraphLayer
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
        super(GNN, self).__init__()

        self.layer1 = CustomGraphLayer(node_in_features, edge_in_features, hidden_channels)
        self.layer2 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer3 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer4 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer5 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer6 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer7 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.concat_layer = CustomGraphLayer(hidden_channels * 7, edge_in_features, hidden_channels)

        self.layer_last = CustomGraphLayer(hidden_channels, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        
        x1 = self.layer1(x, edge_index, edge_attr)

        x2 = self.layer2(x1, edge_index, edge_attr)

        x3 = self.layer3(x2, edge_index, edge_attr)

        x4 = self.layer4(x3, edge_index, edge_attr)

        x5 = self.layer5(x4, edge_index, edge_attr)

        x6 = self.layer6(x5, edge_index, edge_attr)

        x7 = self.layer7(x6, edge_index, edge_attr)
    
        x_concat = torch.cat((x1, x2, x3, x4, x5, x6, x7), dim=1)

        x = self.concat_layer(x_concat, edge_index, edge_attr)

        x = self.layer_last(x, edge_index, edge_attr)

        return x

