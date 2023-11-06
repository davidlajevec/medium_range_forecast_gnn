import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datasets.atmospheric_dataset import AtmosphericDataset
import matplotlib.pyplot as plt
from utils.mesh_creation import create_k_nearest_neighboors_edges
from models import gcn
from train import train
from predict import predict
import os
import csv
import json

# Define constants
TRAINING_NAME = "gcn2"
BATCH_SIZE = 16
EPOCHS = 1
VARIABLES = ["geopotential_500", "u_500", "v_500"]
NUM_VARIABLES = len(VARIABLES)
HIDDEN_CHANNELS = 32
LR = 0.001
GAMMA = 0.99
PATIENCE = 3

PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]

START_YEAR_TRAINING = 1950
END_YEAR_TRAINING = 1950

START_YEAR_VALIDATION = 2003
END_YEAR_VALIDATION = 2006

START_YEAR_TEST = 2022
END_YEAR_TEST = 2022

PLOT = False

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
edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
edge_index = torch.tensor(edge_index, dtype=torch.long)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

# Load the training and validation datasets
training_dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=START_YEAR_TRAINING,
    end_year=END_YEAR_TRAINING,
)
validation_dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=START_YEAR_VALIDATION,
    end_year=END_YEAR_VALIDATION,
)

test_dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=START_YEAR_TEST,
    end_year=END_YEAR_TEST,
)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
train(
    model=model,
    device=device,
    epochs=EPOCHS,
    training_dataset=training_dataset,
    validation_dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    training_name=TRAINING_NAME,
    patience=PATIENCE
)


training_parameters = {
    'epoch': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'gamma': GAMMA,
    'patience': PATIENCE,
    'training_name': TRAINING_NAME,
    'variables': VARIABLES,
    'start_year_training': START_YEAR_TRAINING,
    'end_year_training': END_YEAR_TRAINING,
    'start_year_validation': START_YEAR_VALIDATION,
    'end_year_validation': END_YEAR_VALIDATION,
    'start_year_test': START_YEAR_TEST,
    'end_year_test': END_YEAR_TEST,
}

with open(f"trained_models/{TRAINING_NAME}/training_parameters.json", "w") as f:
    json.dump(training_parameters, f)

predict(
    TRAINING_NAME,
    plot=PLOT,
    variables=VARIABLES,
    projections=PROJECTIONS,
    device=device,
    dataset=test_dataset,
    forecast_length=10,
    plot_index=0
)