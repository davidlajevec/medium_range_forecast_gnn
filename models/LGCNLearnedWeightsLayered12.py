from .custom_layers.LEL3M import CustomGraphLayer
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
        self.layer6 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer7 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer8 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels)
        self.layer_last = CustomGraphLayer(hidden_channels, edge_in_features, out_features)

        # Learnable weights

        # 2nd layer
        self.weight2_1 = nn.Parameter(torch.tensor(0.6))
        self.weight2_2 = nn.Parameter(torch.tensor(0.4))

        # 3rd layer
        self.weight3_1 = nn.Parameter(torch.tensor(0.6))
        self.weight3_2 = nn.Parameter(torch.tensor(0.4))
        self.weight3_3 = nn.Parameter(torch.tensor(0.2))

        # 4th layer
        self.weight4_1 = nn.Parameter(torch.tensor(0.6))
        self.weight4_2 = nn.Parameter(torch.tensor(0.4))
        self.weight4_3 = nn.Parameter(torch.tensor(0.2))
        self.weight4_4 = nn.Parameter(torch.tensor(0.2))

        # 5th layer
        self.weight5_1 = nn.Parameter(torch.tensor(0.6))
        self.weight5_2 = nn.Parameter(torch.tensor(0.4))
        self.weight5_3 = nn.Parameter(torch.tensor(0.2))
        self.weight5_4 = nn.Parameter(torch.tensor(0.2))
        self.weight5_5 = nn.Parameter(torch.tensor(0.2))

        # 6th layer
        self.weight6_1 = nn.Parameter(torch.tensor(0.6))
        self.weight6_2 = nn.Parameter(torch.tensor(0.4))
        self.weight6_3 = nn.Parameter(torch.tensor(0.2))
        self.weight6_4 = nn.Parameter(torch.tensor(0.2))
        self.weight6_5 = nn.Parameter(torch.tensor(0.2))
        self.weight6_6 = nn.Parameter(torch.tensor(0.2))

        # 7th layer
        self.weight7_1 = nn.Parameter(torch.tensor(0.6))
        self.weight7_2 = nn.Parameter(torch.tensor(0.4))
        self.weight7_3 = nn.Parameter(torch.tensor(0.2))
        self.weight7_4 = nn.Parameter(torch.tensor(0.2))
        self.weight7_5 = nn.Parameter(torch.tensor(0.2))
        self.weight7_6 = nn.Parameter(torch.tensor(0.2))
        self.weight7_7 = nn.Parameter(torch.tensor(0.2))

        # 8th layer
        self.weight8_1 = nn.Parameter(torch.tensor(0.6))
        self.weight8_2 = nn.Parameter(torch.tensor(0.4))
        self.weight8_3 = nn.Parameter(torch.tensor(0.2))
        self.weight8_4 = nn.Parameter(torch.tensor(0.2))
        self.weight8_5 = nn.Parameter(torch.tensor(0.2))
        self.weight8_6 = nn.Parameter(torch.tensor(0.2))
        self.weight8_7 = nn.Parameter(torch.tensor(0.2))
        self.weight8_8 = nn.Parameter(torch.tensor(0.2))

        # 9th layer
        self.weight9_1 = nn.Parameter(torch.tensor(0.6))
        self.weight9_2 = nn.Parameter(torch.tensor(0.4))
        self.weight9_3 = nn.Parameter(torch.tensor(0.2))
        self.weight9_4 = nn.Parameter(torch.tensor(0.2))
        self.weight9_5 = nn.Parameter(torch.tensor(0.2))
        self.weight9_6 = nn.Parameter(torch.tensor(0.2))
        self.weight9_7 = nn.Parameter(torch.tensor(0.2))
        self.weight9_8 = nn.Parameter(torch.tensor(0.2))
        self.weight9_9 = nn.Parameter(torch.tensor(0.2))

        # 10th layer
        self.weight10_1 = nn.Parameter(torch.tensor(0.6))
        self.weight10_2 = nn.Parameter(torch.tensor(0.4))
        self.weight10_3 = nn.Parameter(torch.tensor(0.2))
        self.weight10_4 = nn.Parameter(torch.tensor(0.2))
        self.weight10_5 = nn.Parameter(torch.tensor(0.2))
        self.weight10_6 = nn.Parameter(torch.tensor(0.2))
        self.weight10_7 = nn.Parameter(torch.tensor(0.2))
        self.weight10_8 = nn.Parameter(torch.tensor(0.2))
        self.weight10_9 = nn.Parameter(torch.tensor(0.2))
        self.weight10_10 = nn.Parameter(torch.tensor(0.2))

        # 11th layer
        self.weight11_1 = nn.Parameter(torch.tensor(0.6))
        self.weight11_2 = nn.Parameter(torch.tensor(0.4))
        self.weight11_3 = nn.Parameter(torch.tensor(0.2))
        self.weight11_4 = nn.Parameter(torch.tensor(0.2))
        self.weight11_5 = nn.Parameter(torch.tensor(0.2))
        self.weight11_6 = nn.Parameter(torch.tensor(0.2))
        self.weight11_7 = nn.Parameter(torch.tensor(0.2))
        self.weight11_8 = nn.Parameter(torch.tensor(0.2))
        self.weight11_9 = nn.Parameter(torch.tensor(0.2))
        self.weight11_10 = nn.Parameter(torch.tensor(0.2))
        self.weight11_11 = nn.Parameter(torch.tensor(0.2))

        # 12th layer
        self.weight12_1 = nn.Parameter(torch.tensor(0.6))
        self.weight12_2 = nn.Parameter(torch.tensor(0.4))
        self.weight12_3 = nn.Parameter(torch.tensor(0.2))
        self.weight12_4 = nn.Parameter(torch.tensor(0.2))
        self.weight12_5 = nn.Parameter(torch.tensor(0.2))
        self.weight12_6 = nn.Parameter(torch.tensor(0.2))
        self.weight12_7 = nn.Parameter(torch.tensor(0.2))
        self.weight12_8 = nn.Parameter(torch.tensor(0.2))
        self.weight12_9 = nn.Parameter(torch.tensor(0.2))
        self.weight12_10 = nn.Parameter(torch.tensor(0.2))
        self.weight12_11 = nn.Parameter(torch.tensor(0.2))
        self.weight12_12 = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, edge_index, edge_attr):
        
        x1 = self.layer1(x, edge_index, edge_attr)

        x2 = self.layer2(x1, edge_index, edge_attr)

        x2_weighted = x1 * self.weight2_1 + x2 * self.weight2_2

        x3 = self.layer3(x2_weighted, edge_index, edge_attr)

        x3_weighted = x1 * self.weight3_1 + x2_weighted * self.weight3_2 + x3 * self.weight3_3

        x4 = self.layer4(x3_weighted, edge_index, edge_attr)

        x4_weighted = x1 * self.weight4_1 + x2_weighted * self.weight4_2 + x3_weighted * self.weight4_3 + x4 * self.weight4_4

        x5 = self.layer4(x4_weighted, edge_index, edge_attr)
    
        x5_weighted = x1 * self.weight5_1 + x2_weighted * self.weight5_2 + x3_weighted * self.weight5_3 + x4_weighted * self.weight5_4 + x5 * self.weight5_5

        x6 = self.layer5(x5_weighted, edge_index, edge_attr)

        x6_weighted = x1 * self.weight6_1 + x2_weighted * self.weight6_2 + x3_weighted * self.weight6_3 + x4_weighted * self.weight6_4 + x5_weighted * self.weight6_5 + x6 * self.weight6_6

        x7 = self.layer6(x6_weighted, edge_index, edge_attr)

        x7_weighted = x1 * self.weight7_1 + x2_weighted * self.weight7_2 + x3_weighted * self.weight7_3 + x4_weighted * self.weight7_4 + x5_weighted * self.weight7_5 + x6_weighted * self.weight7_6 + x7 * self.weight7_7

        x8 = self.layer7(x7_weighted, edge_index, edge_attr)

        x8_weighted = x1 * self.weight8_1 + x2_weighted * self.weight8_2 + x3_weighted * self.weight8_3 + x4_weighted * self.weight8_4 + x5_weighted * self.weight8_5 + x6_weighted * self.weight8_6 + x7_weighted * self.weight8_7 + x8 * self.weight8_8

        x9 = self.layer8(x8_weighted, edge_index, edge_attr)

        x9_weighted = x1 * self.weight9_1 + x2_weighted * self.weight9_2 + x3_weighted * self.weight9_3 + x4_weighted * self.weight9_4 + x5_weighted * self.weight9_5 + x6_weighted * self.weight9_6 + x7_weighted * self.weight9_7 + x8_weighted * self.weight9_8 + x9 * self.weight9_9

        x10 = self.layer9(x9_weighted, edge_index, edge_attr)

        x10_weighted = x1 * self.weight10_1 + x2_weighted * self.weight10_2 + x3_weighted * self.weight10_3 + x4_weighted * self.weight10_4 + x5_weighted * self.weight10_5 + x6_weighted * self.weight10_6 + x7_weighted * self.weight10_7 + x8_weighted * self.weight10_8 + x9_weighted * self.weight10_9 + x10 * self.weight10_10

        x11 = self.layer10(x10_weighted, edge_index, edge_attr)

        x11_weighted = x1 * self.weight11_1 + x2_weighted * self.weight11_2 + x3_weighted * self.weight11_3 + x4_weighted * self.weight11_4 + x5_weighted * self.weight11_5 + x6_weighted * self.weight11_6 + x7_weighted * self.weight11_7 + x8_weighted * self.weight11_8 + x9_weighted * self.weight11_9 + x10_weighted * self.weight11_10 + x11 * self.weight11_11

        x12 = self.layer11(x11_weighted, edge_index, edge_attr)

        x12_weighted = x1 * self.weight12_1 + x2_weighted * self.weight12_2 + x3_weighted * self.weight12_3 + x4_weighted * self.weight12_4 + x5_weighted * self.weight12_5 + x6_weighted * self.weight12_6 + x7_weighted * self.weight12_7 + x8_weighted * self.weight12_8 + x9_weighted * self.weight12_9 + x10_weighted * self.weight12_10 + x11_weighted * self.weight12_11 + x12 * self.weight12_12

        x = self.layer_last(x12_weighted, edge_index, edge_attr)

        return x

