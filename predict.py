import torch

# use trained model to predict rmse and acc in future and plot one forecast for each variable
# plot forecast graph with climatology, persistence, and model prediction with acc and rmse on top of each other for each variable

from models import gcn
from datasets.atmospheric_dataset import AtmosphericDataset
from utils.mesh_creation import create_k_nearest_neighboors_edges
from utils.plot_atmospheric_field import plot_true_and_predicted_atomspheric_field
import os
import random
from cartopy import crs as ccrs

FORECAST_LENGTH = 5  # days
PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]

# load trained model
VARIABLES = ["geopotential_500"]
NUM_VARIABLES = 1
MODEL_NAME = "gcn"
model = torch.jit.load(f'trained_models/{MODEL_NAME}/traced_model.pt')

edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
edge_index = torch.tensor(edge_index, dtype=torch.long)
# load data to be predicted
dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=2009,
    end_year=2009,
)

index = random.randint(0, len(dataset)-FORECAST_LENGTH*2)
index = 10

if not os.path.exists(f"trained_models/{MODEL_NAME}/forecast_plot"):
    os.makedirs(f"trained_models/{MODEL_NAME}/forecast_plot")

for variable in VARIABLES:
    for projection in PROJECTIONS:
        projection = projection.split("(")[0].split(".")[1]
        if not os.path.exists(f"trained_models/{MODEL_NAME}/forecast_plot/{variable}/{projection}"):
            os.makedirs(f"trained_models/{MODEL_NAME}/forecast_plot/{variable}/{projection}")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
with torch.no_grad():
    model.eval()
    for i in range(FORECAST_LENGTH*2):
        data = dataset[index+i]
        data.to(device)
        if i == 0:
            y_pred = model(data.x, data.edge_index)
        else:
            y_pred = model(y_pred, data.edge_index)
        y_pred_grid = dataset.unstandardize(y_pred.view(60, 120, NUM_VARIABLES))[:, :, 0]
        y_grid = dataset.unstandardize(data.y.view(60, 120, NUM_VARIABLES))[:, :, 0]
        for variable in VARIABLES:
            for projection in PROJECTIONS:
                projection_name = projection.split("(")[0].split(".")[1]
                plot_true_and_predicted_atomspheric_field(
                    y_grid,
                    y_pred_grid,
                    show=False,
                    save=True,
                    save_path=f"trained_models/{MODEL_NAME}/forecast_plot/{variable}/{projection_name}/prediction_{i}.png",
                    left_title="True",
                    right_title="Predicted",
                    projection=eval(projection),
                    title=dataset.file_names[0][i][-17:-4] + f" {variable} {12*(i+1)}h forecast",
                )