import torch
import torch.nn as nn
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
import json

from utils.variables_sets import set1, set2, set3, set4

# CHECK IF RUNNING CORRECT MODEL
from models.LGCNLearnedWeightsLayered7  import GNN 

# Define constants
TRAINING_NAME = "layerd7_128_nnagg2_k2"
BATCH_SIZE = 4
EPOCHS = 15
#VARIABLES = ["geopotential_500", "u_500", "v_500"]
VARIABLES = set1
STATIC_FIELDS = ["land_sea_mask", "surface_topography"]
NUM_ATMOSPHERIC_VARIABLES = len(VARIABLES) 
NUM_STATIC_FIELDS = len(STATIC_FIELDS)
HIDDEN_CHANNELS = 128
LR = 0.001
GAMMA = 0.99
PATIENCE = 5
NON_LINEARITY = nn.LeakyReLU()
K = 2

# Set the device to GPU if available, otherwise CPU
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_GRAPH_ATTRIBUTES = ["x", "edge_index", "edge_attr"]

START_YEAR_TRAINING = 1950
END_YEAR_TRAINING = 1980

START_YEAR_VALIDATION = 2009
END_YEAR_VALIDATION = 2012

START_YEAR_TEST = 2022
END_YEAR_TEST = 2022

PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]
PLOT = True
NUM_PREDICTIONS = 20

FORECAST_LENGTH = 20 # days

def mae_loss(input, target):
    return torch.mean(torch.abs(input - target))

criterion = torch.nn.MSELoss()
#criterion = mae_loss

# Define the model
model = GNN(
    node_in_features=NUM_ATMOSPHERIC_VARIABLES + NUM_STATIC_FIELDS, 
    edge_in_features=3, 
    hidden_channels=HIDDEN_CHANNELS, 
    out_features=NUM_ATMOSPHERIC_VARIABLES,
    non_linearity=NON_LINEARITY,
)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=LR,
    )

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
    device=DEVICE,
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
    device=DEVICE,
    dataset=test_dataset,
    forecast_length=FORECAST_LENGTH,
    plot_index=0,
    num_predictions=NUM_PREDICTIONS,
    input_graph_attributes=INPUT_GRAPH_ATTRIBUTES,
)