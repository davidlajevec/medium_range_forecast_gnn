import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import train_test_split_edges

from models.gcn import GCN
from models.gat import GAT
from datasets.atmospheric_dataset import load_data
from train import train

# Load and preprocess data
dataset = 'Cora'
data = load_data(dataset)
data = train_test_split_edges(data)

# Set up model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(data.num_features, 16, data.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
train(model, optimizer, data, device)

# Evaluate model on test set
model.eval()
_, pred = model(data.x.to(device), data.train_pos_edge_index.to(device))
correct = pred.eq(data.y.to(device)).sum().item()
acc = correct / data.num_nodes
print(f'Test Accuracy: {acc:.4f}')