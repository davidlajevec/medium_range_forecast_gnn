import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

def train(model, optimizer, criterion, train_loader, device):
    model.train()

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_nodes
    return correct / total