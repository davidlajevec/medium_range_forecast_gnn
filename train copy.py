import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datasets.atmospheric_dataset import AtmosphericDataset
from torch_geometric.nn import GCNConv, GATConv
from utils.mesh_creation import create_k_nearest_neighboors_edges
import os
import sys

MODEL_NAME = "gcn"
MODEL_PATH = "models/gcn.py"

sys.path.insert(0, MODEL_PATH)
import Model

if not os.path.exists("trained_models/" + MODEL_NAME):
    os.makedirs("trained_models/" + MODEL_NAME)

with open(f"trained_models/{MODEL_NAME}/model_code.py", "w") as f:
    with open(MODEL_PATH, "r") as model_file:
        f.write(model_file.read())

BATCH_SIZE = 32
EPOCHS = 5
VARIABLES = ["geopotential_500"]
NUM_VARIABLES = len(VARIABLES)

model = gcn.GCN(in_channels=NUM_VARIABLES, 
    hidden_channels=16, 
    out_channels=NUM_VARIABLES)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
edge_index = torch.tensor(edge_index, dtype=torch.long)

training_dataset = AtmosphericDataset(edge_index=edge_index, atmosphere_variables=VARIABLES, start_year=1950, end_year=1950)
validation_dataset = AtmosphericDataset(edge_index=edge_index, atmosphere_variables=VARIABLES, start_year=2003, end_year=2008)
training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_val_loss = float("inf")


for epoch in range(EPOCHS):
    print("-"*50)
    print(f"Epoch: {epoch+1:03d}")
    training_loss = 0
    model.train()
    for i, data in enumerate(training_dataloader):
        print(f)
        data.to(device)
        optimizer.zero_grad()
        
        y_pred = model(data.x, data.edge_index) 
        y = data.y

        loss = criterion(y_pred, y)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f"Batch: {i+1}/{len(training_dataloader)}, Loss: {training_loss/(i+1):.4f}")

    model.eval()
    for i, data in enumerate(validation_dataloader):
        data.to(device)
        # make prediction
        y_pred = model(data.x, data.edge_index) 
        y = data.y
        loss += criterion(y_pred, y).item()
    val_loss = loss/len(validation_dataloader)
    print(f"Validation Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Trace model
        traced_model = torch.jit.trace(model, (data.x, data.edge_index))
        torch.jit.save(traced_model, 'checkpoints/traced_model.pt')
        #torch.save(model.state_dict(), "checkpoints/model.pt")
        print("Model saved!")
