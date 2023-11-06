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

# Define constants
TRAINING_NAME = "gcn_1"
BATCH_SIZE = 32
EPOCHS = 15
VARIABLES = ["geopotential_500", "u_500", "v_500"]
NUM_VARIABLES = len(VARIABLES)
HIDDEN_CHANNELS = 16
LR = 0.01
GAMMA = 0.7
START_YEAR = 1950
END_YEAR = 1970

# Define the model
model = gcn.GCN(
    in_channels=NUM_VARIABLES,
    hidden_channels=HIDDEN_CHANNELS,
    out_channels=NUM_VARIABLES,
)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR,
    )
    
criterion = torch.nn.MSELoss()

# Create edges and points
edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=24)
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

# Load the training and validation datasets
training_dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=START_YEAR,
    end_year=END_YEAR,
)
validation_dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=2003,
    end_year=2006,
)
training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train(
    model=model,
    device=device,
    epochs=EPOCHS,
    training_dataloader=training_dataloader,
    validation_dataloader=validation_dataloader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    training_name=TRAINING_NAME,
)