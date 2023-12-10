import torch
from datasets.atmospheric_dataset import AtmosphericDataset
from utils.mesh_creation import create_neighbooring_edges
from utils.plot_atmospheric_field import plot_true_and_predicted_atomspheric_field
from utils.utils import filename_to_climatology
from utils.metrics import spherical_weighted_rmse, spherical_weighted_acc
import os
import random
from cartopy import crs as ccrs
import numpy as np
from matplotlib import pyplot as plt
import numpy as np


def predict(
    model_name,
    plot,
    variables,
    projections,
    device,
    dataset,
    forecast_length,
    plot_index=0,
    num_predictions=None,
    input_graph_attributes=["x", "edge_index", "edge_attributes", "batch"],
):
    """
    Predicts atmospheric variables using a trained GNN model.

    Args:
        model_name (str): Name of the trained model.
        plot (bool): Whether to plot the predicted atmospheric fields.
        variables (list): List of atmospheric variables to predict.
        projections (list): List of projections to use for plotting.
        device (str): Device to use for prediction.
        dataset (AtmosphericFieldDataset): Dataset to use for prediction.
        forecast_length (int): Length of the forecast in hours.
        plot_index (int, optional): Index of the prediction to plot. Defaults to 0.
        num_predictions (int, optional): Number of predictions to make. Defaults to None.
        input_graph_attributes (list, optional): List of input graph attributes. Defaults to ["x", "edge_index", "edge_attributes", "batch"].

    Returns:
        None
    """
def predict(
    model_name,
    plot,
    variables,
    static_fields,
    projections,
    device,
    dataset,
    forecast_length,
    plot_index=0,
    num_predictions=None,
    input_graph_attributes=["x", "edge_index", "edge_attributes", "batch"],
):
    num_variables = len(variables)
    num_static_fields = len(static_fields)

    if not os.path.exists(f"trained_models/{model_name}/forecast_plot"):
        os.makedirs(f"trained_models/{model_name}/forecast_plot")

    if plot:
        for variable in variables:
            for projection in projections:
                projection = projection.split("(")[0].split(".")[1]
                if not os.path.exists(
                    f"trained_models/{model_name}/forecast_plot/{variable}/{projection}"
                ):
                    os.makedirs(
                        f"trained_models/{model_name}/forecast_plot/{variable}/{projection}"
                    )
    if num_predictions and num_predictions < len(dataset):
        indices = list(range(num_predictions))
    elif num_predictions:
        print("Num_predictions must be less than the length of the dataset!")
    else:
        indices = list(range(len(dataset) - forecast_length * 2))
    #model = torch.jit.load(f"trained_models/{model_name}/traced_model.pt")
    model = torch.load(f"trained_models/{model_name}/model.pth")
    with torch.no_grad():
        model.to(device)
        model.eval()
        rmse, rmse_climatology, rmse_persistence = {}, {}, {}
        acc, acc_climatology, acc_persistence = {}, {}, {}
        for variable in variables:
            rmse[variable] = np.zeros((len(indices), forecast_length * 2 + 1))
            rmse[variable][:, 0] = 0
            rmse_climatology[variable] = np.zeros(
                (len(indices), forecast_length * 2 + 1)
            )
            rmse_persistence[variable] = np.zeros(
                (len(indices), forecast_length * 2 + 1)
            )
            rmse_persistence[variable][:, 0] = 0

            acc[variable] = np.zeros((len(indices), forecast_length * 2 + 1))
            acc[variable][:, 0] = 1
            acc_climatology[variable] = np.zeros(
                (len(indices), forecast_length * 2 + 1)
            )
            acc_climatology[variable][:, 0] = 0
            acc_persistence[variable] = np.zeros(
                (len(indices), forecast_length * 2 + 1)
            )
            acc_persistence[variable][:, 0] = 1
        for index in indices:
            if index % 5 == 0:
                print(f"Predicting {index}/{len(indices)}", end="\r")
            for i in range(1, forecast_length * 2 + 1):
                data = dataset[index + i]
                data.to(device)
                if i == 1:
                    data_mapping = {attr:getattr(data, attr) for attr in input_graph_attributes}
                    y_pred = model(**data_mapping)
                    y_persistence_grid = data.x.view(60, 120, num_variables+num_static_fields).cpu()[:, :, :num_variables]
                    y_persistence_grid = dataset.unstandardize(
                        y_persistence_grid
                    ).numpy()
                else:
                    prediction_data = data
                    prediction_data.x[:,:num_variables] = y_pred
                    data_mapping = {attr:getattr(prediction_data, attr) for attr in input_graph_attributes}
                    y_pred = model(**data_mapping)
                y_pred_grid = y_pred.view(60, 120, num_variables).cpu()
                y_pred_grid = dataset.unstandardize(y_pred_grid).numpy()
                y_true_grid = data.y.view(60, 120, num_variables).cpu()
                y_true_grid = dataset.unstandardize(y_true_grid).numpy()

                for j, variable in enumerate(variables):
                    climatology_field = filename_to_climatology(
                        dataset.file_names[j][i]
                    )

                    rmse[variable][index, i] = spherical_weighted_rmse(
                        y_true_grid[:, :, j],
                        y_pred_grid[:, :, j],
                        dataset.lats[1:-1, :-1],
                    )
                    acc[variable][index, i] = spherical_weighted_acc(
                        y_true_grid[:, :, j],
                        y_pred_grid[:, :, j],
                        climatology_field,
                        dataset.lats[1:-1, :-1],
                    )
                    # calculate rmse and acc using climatology
                    rmse_climatology[variable][index, i] = spherical_weighted_rmse(
                        y_true_grid[:, :, j], climatology_field, dataset.lats[1:-1, :-1]
                    )
                    if i == 1:
                        rmse_climatology[variable][index, 0] = spherical_weighted_rmse(
                            y_true_grid[:, :, j],
                            climatology_field,
                            dataset.lats[1:-1, :-1],
                        )
                    acc_climatology[variable][index, i] = 0
                    # calculate rmse and acc using persistence
                    rmse_persistence[variable][index, i] = spherical_weighted_rmse(
                        y_true_grid[:, :, j],
                        y_persistence_grid[:, :, j],
                        dataset.lats[1:-1, :-1],
                    )
                    acc_persistence[variable][index, i] = spherical_weighted_acc(
                        y_true_grid[:, :, j],
                        y_persistence_grid[:, :, j],
                        climatology_field,
                        dataset.lats[1:-1, :-1],
                    )
                    if plot and index == plot_index:
                        for projection in projections:
                            projection_name = projection.split("(")[0].split(".")[1]
                            plot_true_and_predicted_atomspheric_field(
                                y_true_grid[:, :, j],
                                y_pred_grid[:, :, j],
                                show=False,
                                save=True,
                                save_path=f"trained_models/{model_name}/forecast_plot/{variable}/{projection_name}/prediction_{i}.png",
                                left_title="True",
                                right_title="Predicted",
                                projection=eval(projection),
                                title=dataset.file_names[0][i][-17:-4]
                                + f" {variable} {12*i}h forecast",
                            )

        days = [i / 2 for i in range(forecast_length * 2 + 1)]
        for variable in variables:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
            acc_mean = np.mean(acc[variable], axis=0)
            ax1.plot(days, acc_mean, label="prediction")
            acc_climatology_mean = np.mean(acc_climatology[variable], axis=0)
            ax1.plot(days, acc_climatology_mean, label=f"climatology", linestyle="--")
            acc_persistence_mean = np.mean(acc_persistence[variable], axis=0)
            ax1.plot(
                days,
                acc_persistence_mean,
                label=f"persistence",
                linestyle="dotted",
                color="black",
                linewidth=2,
            )

            rmse_mean = np.mean(rmse[variable], axis=0)
            ax2.plot(days, rmse_mean, label="prediction")
            rmse_climatology_mean = np.mean(rmse_climatology[variable], axis=0)
            ax2.plot(days, rmse_climatology_mean, label="climatology", linestyle="--")
            rmse_persistence_mean = np.mean(rmse_persistence[variable], axis=0)
            ax2.plot(
                days,
                rmse_persistence_mean,
                label="persistence",
                linestyle="dotted",
                color="black",
                linewidth=2,
            )

            ax1.set_ylabel("ACC")
            ax1.legend()
            ax1.grid(True)
            ax1.set_title(f"{variable}", fontweight="bold")

            ax2.set_xlabel("Days")
            ax2.set_ylabel("RMSE")
            ax2.grid(True)

            plt.savefig(
                f"trained_models/{model_name}/forecast_plot/acc_rmse_{variable}.png"
            )


if __name__ == "__main__":
    FORECAST_LENGTH = 14  # days
    PROJECTIONS = ["ccrs.Orthographic(-10, 62)", "ccrs.Robinson()"]
    PLOT = True
    PLOT_INDEX = 0
    NUM_PREDICTIONS = 20
    INPUT_GRAPH_ATTRIBUTES = ["x", "edge_index", "edge_attr"]
    # load trained model
    VARIABLES = ["geopotential_500", "u_500", "v_500"]
    MODEL_NAME = "locally_embedded"

    edge_index, edge_attrs, _, _ = create_neighbooring_edges(k=1)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs.T, dtype=torch.float)

    # load data to be predicted
    dataset = AtmosphericDataset(
        edge_index=edge_index,
        edge_attributes=edge_attrs,
        atmosphere_variables=VARIABLES,
        start_year=2019,
        end_year=2019,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict(
        MODEL_NAME,
        PLOT,
        VARIABLES,
        PROJECTIONS,
        device,
        dataset,
        FORECAST_LENGTH,
        plot_index=PLOT_INDEX,
        num_predictions=NUM_PREDICTIONS,
        input_graph_attributes=INPUT_GRAPH_ATTRIBUTES,
    )
