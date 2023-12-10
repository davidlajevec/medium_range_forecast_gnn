"""
This script trains a Graph Convolutional Network (GCN) to perform medium-range weather forecasting.

The GCN is trained on atmospheric data from the ERA5 reanalysis dataset, and the training and validation datasets are loaded using the AtmosphericDataset class.

The GCN is defined in the gcn module, and the training is performed using the train function.

The trained model is saved to disk, and the predict function is used to generate weather forecasts using the trained model.

The forecasts are plotted using the matplotlib library.
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datasets.atmospheric_dataset import AtmosphericDataset
#from datasets.atmospheric_dataset_steps import AtmosphericDataset
import matplotlib.pyplot as plt
from utils.mesh_creation import create_neighbooring_edges
from train import train
#from train_multiple_steps import train
from predict import predict
import os
import csv
import json

# CHECK IF RUNNING CORRECT MODEL
from models.LGCNLearnedWeightsLayeredMore import GNN

# Define constants
TRAINING_NAME = "layerd_64_12_lay"
BATCH_SIZE = 4
EPOCHS = 10
VARIABLES = ["geopotential_500", "u_500", "v_500"]
STATIC_FIELDS = ["land_sea_mask", "surface_topography"]
NUM_ATMOSPHERIC_VARIABLES = len(VARIABLES) 
NUM_STATIC_FIELDS = len(STATIC_FIELDS)
HIDDEN_CHANNELS = 64
LR = 0.001
GAMMA = 0.99
PATIENCE = 3

INPUT_GRAPH_ATTRIBUTES = ["x", "edge_index", "edge_attr"]

START_YEAR_TRAINING = 1950
END_YEAR_TRAINING = 1970

START_YEAR_VALIDATION = 2009
END_YEAR_VALIDATION = 2015

START_YEAR_TEST = 2022
END_YEAR_TEST = 2022

PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]
PLOT = True
NUM_PREDICTIONS = 20

# Define the model
model = GNN(
    node_in_features=NUM_ATMOSPHERIC_VARIABLES + NUM_STATIC_FIELDS, 
    edge_in_features=3, 
    hidden_channels=HIDDEN_CHANNELS, 
    out_features=NUM_ATMOSPHERIC_VARIABLES,
)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR,
    )
    
criterion = torch.nn.MSELoss()

# Create edges and points
edge_index, edge_attrs, _, _ = create_neighbooring_edges(k=1)
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

# Define the scheduler
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

# Load the training and validation datasets
training_dataset = AtmosphericDataset(
    edge_index=edge_index,
    edge_attributes=edge_attrs,
    atmosphere_variables=VARIABLES,
    static_fields=STATIC_FIELDS,
    start_year=START_YEAR_TRAINING,
    end_year=END_YEAR_TRAINING,
)
validation_dataset = AtmosphericDataset(
    edge_index=edge_index,
    edge_attributes=edge_attrs,
    atmosphere_variables=VARIABLES,
    static_fields=STATIC_FIELDS,
    start_year=START_YEAR_VALIDATION,
    end_year=END_YEAR_VALIDATION,
)

test_dataset = AtmosphericDataset(
    edge_index=edge_index,
    edge_attributes=edge_attrs,
    atmosphere_variables=VARIABLES,
    static_fields=STATIC_FIELDS,
    start_year=START_YEAR_TEST,
    end_year=END_YEAR_TEST,
)

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_parameters = {
    'epoch': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'gamma': GAMMA,
    'patience': PATIENCE,
    'input_graph_attributes': INPUT_GRAPH_ATTRIBUTES,
    'training_name': TRAINING_NAME,
    'variables': VARIABLES,
    'start_year_training': START_YEAR_TRAINING,
    'end_year_training': END_YEAR_TRAINING,
    'start_year_validation': START_YEAR_VALIDATION,
    'end_year_validation': END_YEAR_VALIDATION,
    'start_year_test': START_YEAR_TEST,
    'end_year_test': END_YEAR_TEST,
}

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
    patience=PATIENCE,
    input_graph_attributes=INPUT_GRAPH_ATTRIBUTES,
)

# Save training parameters to disk
with open(f"trained_models/{TRAINING_NAME}/training_parameters.json", "w") as f:
    json.dump(training_parameters, f)

# Generate weather forecasts using the trained model
predict(
    TRAINING_NAME,
    plot=PLOT,
    variables=VARIABLES,
    static_fields=STATIC_FIELDS,
    projections=PROJECTIONS,
    device=device,
    dataset=test_dataset,
    forecast_length=10,
    plot_index=0,
    num_predictions=NUM_PREDICTIONS,
    input_graph_attributes=INPUT_GRAPH_ATTRIBUTES,
)