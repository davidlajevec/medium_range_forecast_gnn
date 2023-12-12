from .custom_layers.LEL3M import CustomGraphLayer
#from .custom_layers.LEL3MSelfConnection import CustomGraphLayer
#from .custom_layers.LEL4M import CustomGraphLayer
#from .custom_layers.LELNN import CustomGraphLayer
import torch
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, 
            node_in_features, 
            edge_in_features, 
            hidden_channels, 
            out_features,
            non_linearity=nn.ReLU(),):
        super(GNN, self).__init__()

        self.layer1 = CustomGraphLayer(node_in_features, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.layer2 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.layer3 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.layer4 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.layer5 = CustomGraphLayer(hidden_channels, edge_in_features, hidden_channels, non_linearity=non_linearity)
        self.layer_last = CustomGraphLayer(hidden_channels, edge_in_features, out_features, non_linearity=non_linearity)

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

    def forward(self, x, edge_index, edge_attr):
        
        x1 = self.layer1(x, edge_index, edge_attr)

        x2 = self.layer2(x1, edge_index, edge_attr)
        
        weights2_sum = self.weight2_1 + self.weight2_2

        x2_weighted = x1 * self.weight2_1/weights2_sum + x2 * self.weight2_2/weights2_sum

        x3 = self.layer3(x2_weighted, edge_index, edge_attr)

        weights3_sum = self.weight3_1 + self.weight3_2 + self.weight3_3

        x3_weighted = x1 * self.weight3_1/weights3_sum + x2_weighted * self.weight3_2/weights3_sum + x3 * self.weight3_3/weights3_sum

        x4 = self.layer4(x3_weighted, edge_index, edge_attr)

        weights4_sum = self.weight4_1 + self.weight4_2 + self.weight4_3 + self.weight4_4

        x4_weighted = x1 * self.weight4_1/weights4_sum + x2_weighted * self.weight4_2/weights4_sum + x3_weighted * self.weight4_3/weights4_sum + x4 * self.weight4_4/weights4_sum

        x5 = self.layer4(x4_weighted, edge_index, edge_attr)

        weights5_sum = self.weight5_1 + self.weight5_2 + self.weight5_3 + self.weight5_4 + self.weight5_5
    
        x5_weighted = x1 * self.weight5_1/weights5_sum + x2_weighted * self.weight5_2/weights5_sum + x3_weighted * self.weight5_3/weights5_sum + x4_weighted * self.weight5_4/weights5_sum + x5 * self.weight5_5/weights5_sum

        x = self.layer_last(x5_weighted, edge_index, edge_attr)

        return x

