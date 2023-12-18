import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datasets.atmospheric_dataset_multiple_steps import AtmosphericDataset
#from datasets.atmospheric_dataset_steps import AtmosphericDataset
import matplotlib.pyplot as plt
from utils.mesh_creation import create_neighbooring_edges
#from train_multiple_steps import train
#from train_multiple_steps_2_cores import train
from train_multiple_steps import train
from predict import predict
import os
import csv
import json

from utils.variables_sets import set1, set2, set3, set4

# CHECK IF RUNNING CORRECT MODEL
from models.UNettest_Custom import GNN
#from models.LGCNLearnedWeightsLayered5 import GNN

# Define constants
TRAINING_NAME = "camelot2_4steps"
BATCH_SIZE = 2
EPOCHS = 10
VARIABLES = set1
STATIC_FIELDS = ["land_sea_mask", "surface_topography"]
NUM_ATMOSPHERIC_VARIABLES = len(VARIABLES) 
NUM_STATIC_FIELDS = len(STATIC_FIELDS)
HIDDEN_CHANNELS = 128
LR = 0.0005
GAMMA = 0.99
PATIENCE = 3
NON_LINEARITY = nn.LeakyReLU()
K = 2
NUM_OF_PREDICTION_STEPS = 4

INPUT_GRAPH_ATTRIBUTES = ["x", "edge_index", "edge_attr"]

START_YEAR_TRAINING = 1990
END_YEAR_TRAINING = 2015

START_YEAR_VALIDATION = 2015
END_YEAR_VALIDATION = 2020

START_YEAR_TEST = 2022
END_YEAR_TEST = 2022

PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]
PLOT = True
NUM_PREDICTIONS = 20

FORECAST_LENGTH = 20 # days

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = torch.load("trained_models/unet_depth3_sum_05_drop_256_dropout_01/model.pth")
#model = torch.load("trained_models/final_1step_lr_001_gamma_90/model.pth")

model = torch.load("trained_models/old_models_2/unet_depth3_sum_05_drop_128hid_camelot2/model.pth")
#model = torch.load("trained_models/old_models_2/layerd8_128_nn_k2/model.pth")

#model = GNN(
#   in_channels=NUM_ATMOSPHERIC_VARIABLES + NUM_STATIC_FIELDS, 
#   hidden_channels=HIDDEN_CHANNELS, 
#   out_channels=NUM_ATMOSPHERIC_VARIABLES,
#   depth=3,
#   sum_res=True
#)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR,
    )
    
criterion = torch.nn.MSELoss()

# Create edges and points
edge_index, edge_attrs, _, _ = create_neighbooring_edges(k=K)
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
    num_of_prediction_steps=NUM_OF_PREDICTION_STEPS
)
validation_dataset = AtmosphericDataset(
    edge_index=edge_index,
    edge_attributes=edge_attrs,
    atmosphere_variables=VARIABLES,
    static_fields=STATIC_FIELDS,
    start_year=START_YEAR_VALIDATION,
    end_year=END_YEAR_VALIDATION,
    num_of_prediction_steps=NUM_OF_PREDICTION_STEPS
)

test_dataset = AtmosphericDataset(
    edge_index=edge_index,
    edge_attributes=edge_attrs,
    atmosphere_variables=VARIABLES,
    static_fields=STATIC_FIELDS,
    start_year=START_YEAR_TEST,
    end_year=END_YEAR_TEST,
    num_of_prediction_steps=NUM_OF_PREDICTION_STEPS
)


training_parameters = {
    'epoch': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LR,
    'gamma': GAMMA,
    'patience': PATIENCE,
    'non_linearity': str(NON_LINEARITY),
    'k': K,
    'hidden_channels': HIDDEN_CHANNELS,
    'input_graph_attributes': INPUT_GRAPH_ATTRIBUTES,
    'training_name': TRAINING_NAME,
    'variables': VARIABLES,
    'static_fields': STATIC_FIELDS,
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
    number_of_prediction_steps=NUM_OF_PREDICTION_STEPS  
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
    forecast_length=FORECAST_LENGTH,
    plot_index=0,
    num_predictions=NUM_PREDICTIONS,
    input_graph_attributes=INPUT_GRAPH_ATTRIBUTES,
)