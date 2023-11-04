import torch

# use trained model to predict rmse and acc in future and plot one forecast for each variable
# plot forecast graph with climatology, persistence, and model prediction with acc and rmse on top of each other for each variable

from models import gcn
from datasets.atmospheric_dataset import AtmosphericDataset
from utils.mesh_creation import create_k_nearest_neighboors_edges

# load trained model
VARIABLES = ["geopotential_500"]
NUM_VARIABLES = 1
model = gcn.GCN(
    in_channels=NUM_VARIABLES, hidden_channels=16, out_channels=NUM_VARIABLES
)
model.load_state_dict(torch.load("checkpoints/model.pt"))
edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
edge_index = torch.tensor(edge_index, dtype=torch.long)
# load data to be predicted
dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=2009,
    end_year=2009,
)

# make prediction
with torch.no_grad():
    model.eval()
    data = dataset[10]  # replace with your own way of getting inputs
    y_pred = model(data.x, data.edge_index)
    y_pred = dataset.unstandardize(y_pred.view(60, 120, NUM_VARIABLES))[:, :, 0]
    y = dataset.unstandardize(data.y.view(60, 120, NUM_VARIABLES))[:, :, 0]

from utils.plot_atmospheric_field import plot_true_and_predicted_atomspheric_field

plot_true_and_predicted_atomspheric_field(
    y,
    y_pred,
    show=True,
    
)
