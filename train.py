import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datasets.atmospheric_dataset import AtmosphericDataset
import matplotlib.pyplot as plt
from utils.mesh_creation import create_k_nearest_neighboors_edges

from models import gcn
import os
import csv

TRAINING_NAME = ""

MODEL_NAME = "gcn"

saving_path = "trained_models/" + MODEL_NAME + TRAINING_NAME
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

BATCH_SIZE = 64
EPOCHS = 5
VARIABLES = ["geopotential_500"]
NUM_VARIABLES = len(VARIABLES)
HIDDEN_CHANNELS = 32

model = gcn.GCN(in_channels=NUM_VARIABLES, 
    hidden_channels=HIDDEN_CHANNELS, 
    out_channels=NUM_VARIABLES)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
edge_index = torch.tensor(edge_index, dtype=torch.long)

training_dataset = AtmosphericDataset(edge_index=edge_index, atmosphere_variables=VARIABLES, start_year=1950, end_year=1970)
validation_dataset = AtmosphericDataset(edge_index=edge_index, atmosphere_variables=VARIABLES, start_year=2003, end_year=2008)
training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
best_val_loss = float("inf")

training_losses = []
validation_losses = []

for epoch in range(EPOCHS):
    print("-"*50)
    print(f"Epoch: {epoch+1:03d}")
    training_loss = 0
    model.train()
    for i, data in enumerate(training_dataloader):
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
    loss = 0
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
        torch.jit.save(traced_model, f'{saving_path}/traced_model.pt')
        print("Model saved!")
    
    # Save training and validation losses
    training_losses.append(training_loss/len(training_dataloader))
    validation_losses.append(val_loss)
    with open(f"{saving_path}/losses.csv", mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "training_loss", "validation_loss"])
        for i in range(epoch+1):
            writer.writerow([i+1, training_losses[i], validation_losses[i]])
    
    # Plot training and validation losses
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig(f"{saving_path}/training_losses_plot.png")
