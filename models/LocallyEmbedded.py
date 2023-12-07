from .custom_layers.CustomLocalEmbeddingLayer import CustomGraphLayer
import torch
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, node_in_features, edge_in_features, hidden_channels, out_features):
        super(GNN, self).__init__()

        self.layer1 = CustomGraphLayer(node_in_features, edge_in_features, hidden_channels)

        self.layer2 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer3 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer4 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)

        self.layer_last = CustomGraphLayer(hidden_channels, edge_in_features, out_features)

    def forward(self, x, edge_index, edge_attr):
        
        x = self.layer1(x, edge_index, edge_attr)

        x = self.layer2(x, edge_index, edge_attr)

        x = self.layer3(x, edge_index, edge_attr)

        x = self.layer_last(x, edge_index, edge_attr)

        return x

