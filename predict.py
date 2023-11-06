import torch
from datasets.atmospheric_dataset import AtmosphericDataset
from utils.mesh_creation import create_k_nearest_neighboors_edges
from utils.plot_atmospheric_field import plot_true_and_predicted_atomspheric_field
from utils.utils import filename_to_climatology, spherical_weighted_rmse, spherical_weighted_acc
import os
import random
from cartopy import crs as ccrs
import numpy as np
from matplotlib import pyplot as plt

FORECAST_LENGTH = 5  # days
PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]
PLOT = False

# load trained model
VARIABLES = ["geopotential_500", "u_500", "v_500"]
NUM_VARIABLES = len(VARIABLES)
MODEL_NAME = "gcn_1"
model = torch.jit.load(f'trained_models/{MODEL_NAME}/traced_model.pt')

edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
edge_index = torch.tensor(edge_index, dtype=torch.long)
# load data to be predicted
dataset = AtmosphericDataset(
    edge_index=edge_index,
    atmosphere_variables=VARIABLES,
    start_year=2009,
    end_year=2022,  
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model_name, plot, variables, projections, device, dataset, forecast_length, plot_index = 0):
    
    if not os.path.exists(f"trained_models/{model_name}/forecast_plot"):
        os.makedirs(f"trained_models/{model_name}/forecast_plot")

    if plot:
        for variable in variables:
            for projection in projections:
                projection = projection.split("(")[0].split(".")[1]
                if not os.path.exists(f"trained_models/{MODEL_NAME}/forecast_plot/{variable}/{projection}"):
                    os.makedirs(f"trained_models/{MODEL_NAME}/forecast_plot/{variable}/{projection}")


    index = random.randint(0, len(dataset)-FORECAST_LENGTH*2)
    index = 10
    indices = [10, 12]

    with torch.no_grad():
        model.to(device)
        model.eval()
        rmse, rmse_climatology, rmse_persistence = {}, {}, {}
        acc, acc_climatology, acc_persistence = {}, {}, {}
        for variable in VARIABLES:
            rmse[variable] = np.zeros((len(indices), len(forecast_length*2)))
            rmse_climatology[variable] = np.zeros((len(indices), len(forecast_length*2)))
            rmse_persistence[variable] = np.zeros((len(indices), len(forecast_length*2)))

            acc[variable] = np.zeros((len(indices), len(forecast_length*2)))
            acc_climatology[variable] = np.zeros((len(indices), len(forecast_length*2)))
            acc_persistence[variable] = np.zeros((len(indices), len(forecast_length*2)))
        for index in indices:
            for i in range(forecast_length*2):
                data = dataset[index+i]
                data.to(device)
                if i == 0:
                    y_pred = model(data.x, data.edge_index)
                else:
                    y_pred = model(y_pred, data.edge_index)
                y_pred_grid = y_pred.view(60, 120, NUM_VARIABLES).cpu()
                y_pred_grid = dataset.unstandardize(y_pred_grid).numpy()
                y_true_grid = data.y.view(60, 120, NUM_VARIABLES).cpu()
                y_true_grid = dataset.unstandardize(y_true_grid).numpy()
                for j, variable in enumerate(VARIABLES):
                    climatology_field = filename_to_climatology(dataset.file_names[j][i])
                    # calculate rmse and acc using prediction
                    #rmse[variable].append(spherical_weighted_rmse(y_true_grid[:,:,j], y_pred_grid[:,:,j], dataset.lats[1:-1,:-1]))
                    rmse[variable][index,i] = spherical_weighted_rmse(y_true_grid[:,:,j], y_pred_grid[:,:,j], dataset.lats[1:-1,:-1])
                    #acc[variable].append(spherical_weighted_acc(y_true_grid[:,:,j], y_pred_grid[:,:,j], climatology_field, dataset.lats[1:-1,:-1]))
                    acc[variable][index,i] = spherical_weighted_acc(y_true_grid[:,:,j], y_pred_grid[:,:,j], climatology_field, dataset.lats[1:-1,:-1])
                    # calculate rmse and acc using climatology
                    rmse_climatology[variable][index,i] = spherical_weighted_rmse(y_true_grid[:,:,j], climatology_field, dataset.lats[1:-1,:-1])
                    acc_climatology[variable][index,i] = 0
                    # calculate rmse and acc using persistence
                    if i == 0:
                        rmse_persistence[variable][index,i] = spherical_weighted_rmse(y_true_grid[:,:,j], y_true_grid[:,:,j], dataset.lats[1:-1,:-1])
                        acc_persistence[variable][index,i] = spherical_weighted_acc(y_true_grid[:,:,j], y_true_grid[:,:,j], climatology_field, dataset.lats[1:-1,:-1])
                    else:
                        rmse_persistence[variable][index,i] = spherical_weighted_rmse(y_true_grid[:,:,j], y_true_grid[:,:,j-1], dataset.lats[1:-1,:-1])
                        acc_persistence[variable][index,i] = spherical_weighted_acc(y_true_grid[:,:,j], y_true_grid[:,:,j-1], climatology_field, dataset.lats[1:-1,:-1])
                    if PLOT and index == plot_index:
                        for projection in PROJECTIONS:
                            projection_name = projection.split("(")[0].split(".")[1]
                            plot_true_and_predicted_atomspheric_field(
                                y_true_grid[:,:,j],
                                y_pred_grid[:,:,j],
                                show=False,
                                save=True,
                                save_path=f"trained_models/{MODEL_NAME}/forecast_plot/{variable}/{projection_name}/prediction_{i}.png",
                                left_title="True",
                                right_title="Predicted",
                                projection=eval(projection),
                                title=dataset.file_names[0][i][-17:-4] + f" {variable} {12*(i+1)}h forecast",
                            )

        days = [i/2 for i in range(FORECAST_LENGTH*2)]
        for variable in VARIABLES:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

            acc_mean = np.mean(acc[variable], axis=0)
            ax1.plot(days, acc_mean, label="prediction")
            acc_climatology_mean = np.mean(acc_climatology[variable], axis=0)
            ax1.plot(days, acc_climatology_mean, label=f"climatology", linestyle='--')
            acc_persistence_mean = np.mean(acc_persistence[variable], axis=0)
            ax1.plot(days, acc_persistence_mean, label=f"persistence", linestyle='dotted', color='black', linewidth=2)

            ax2.plot(days, rmse[variable], label="prediction")
            ax2.plot(days, rmse_climatology[variable], label="climatology", linestyle='--')
            ax2.plot(days, rmse_persistence[variable], label="persistence", linestyle='dotted', color='black', linewidth=2)

            ax1.set_ylabel('ACC')
            ax1.legend()
            ax1.grid(True)
            ax1.set_title(f"{variable}", fontweight='bold')

            ax2.set_xlabel('Days')
            ax2.set_ylabel('RMSE')
            ax2.grid(True)

            plt.savefig(f"trained_models/{MODEL_NAME}/forecast_plot/acc_rmse_{variable}.png")
