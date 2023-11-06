import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling, SAGPooling, global_max_pool, global_add_pool, DenseGCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.bn4 = nn.BatchNorm1d(hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)
        self.pool1 = TopKPooling(hidden_channels, ratio=0.75)
        self.pool2 = SAGPooling(hidden_channels, ratio=0.75)
        self.pool3 = global_mean_pool
        self.pool4 = global_max_pool
        self.pool5 = global_add_pool
        self.unpool1 = TopKPooling(hidden_channels, ratio=0.75)
        self.unpool2 = SAGPooling(hidden_channels, ratio=0.75)
        self.unpool3 = global_mean_pool
        self.unpool4 = global_max_pool
        self.unpool5 = global_add_pool

    def forward(self, x, edge_index):
        # first layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.05, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index)
        
        # second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.05, training=self.training)
        
        # third layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.05, training=self.training)
        
        # fourth layer
        #x = self.conv4(x, edge_index)
        #x = self.bn4(x)
        #x = F.relu(x)
        #x = F.dropout(x, p = 0.05, training=self.training)
        #x = self.pool4(x, batch)
        
        # output layer
        #x = self.conv5(x, edge_index)
        #x = self.pool5(x, batch)
        #x = self.unpool5(x, batch)
        #x = self.conv4(x, edge_index)
        #x = self.unpool4(x, batch)
        #x = F.relu(x)
        #x = F.dropout(x, p = 0.05, training=self.training)
        #x = self.conv3(x, edge_index)
        #x = self.unpool3(x, batch)
        #x = F.relu(x)
        #x = F.dropout(x, p = 0.05, training=self.training)
        #x, edge_index, _, batch, _, _ = self.unpool2(x, edge_index, batch)
        #x = self.conv2(x, edge_index)
        #x = self.bn2(x)
        #x = F.relu(x)
        #x = F.dropout(x, p = 0.05, training=self.training)
        x, edge_index, _, batch, _, _ = self.unpool1(x, edge_index, batch)
        #x = self.conv1(x, edge_index)
        #x = self.bn1(x)
        #x = F.relu(x)
        #x = F.dropout(x, p = 0.05, training=self.training)
        #x = self.conv5(x, edge_index)
        return x

if __name__ == "__main__":
    x = torch.randn(1000, 3)  # 10 nodes, 16 features per node

    edge_index = torch.tensor([[i for i in range(200)],
                               [i for i in range(200, 400)]], dtype=torch.long)

    model = GCN(in_channels=3, hidden_channels=16, out_channels=3)

    output = model(x, edge_index)
    print(output.shape)  # torch.Size([10, 3])