import numpy as np


def spherical_weights(lat):
    """
    Computes the spherical weight for a given latitude.

    Args:
        lat (numpy.ndarray): Array of latitudes in degrees.

    Returns:
        numpy.ndarray: Array of spherical weights.
    """
    cos_lat = np.cos(np.deg2rad(lat))
    normalization = np.sum(cos_lat[:, 0]) / lat.shape[0]
    return cos_lat / normalization 


def spherical_weighted_rmse(y_true, y_pred, lat):
    """
    Calculate the spherical weighted root mean squared error (RMSE) between two arrays of atmospheric variables.
    """
    weights = spherical_weights(lat)
    weights = weights.flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    rmse = (
        np.sqrt(np.mean(weights * (y_true - y_pred) ** 2)) / 10
    )  # divide by 10 to convert to decameters
    return rmse


def spherical_weighted_acc(y_true, y_pred, climatology_field, lat):
    """
    Calculate the spherical weighted anomaly correlation coefficient (ACC) between two arrays of atmospheric variables.
    """
    weights = spherical_weights(lat)
    weights = weights.flatten()
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    climatology_field = climatology_field.flatten()

    cov = np.mean(weights * (y_true - climatology_field) * (y_pred - climatology_field))
    var_true = np.mean(weights * (y_true - climatology_field) ** 2)
    var_pred = np.mean(weights * (y_pred - climatology_field) ** 2)

    # Check for near-zero variance to prevent division by zero
    if np.isclose(var_true, 0) or np.isclose(var_pred, 0):
        return 0.0  # or some other appropriate value or warning

    acc = cov / np.sqrt(var_true * var_pred)
    return acc


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.getcwd())
    from datasets.atmospheric_dataset import AtmosphericDataset
    import numpy as np
    import torch
    from mesh_creation import create_k_nearest_neighboors_edges
    from utils import filename_to_climatology

    edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=8)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    # load data to be predicted
    dataset = AtmosphericDataset(
        edge_index=edge_index,
        atmosphere_variables=["geopotential_500", "u_500", "v_500"],
        start_year=2019,
        end_year=2019,
    )
    x = dataset.__getitem__(0).x.view(60, 120, 3)
    y = dataset.__getitem__(210).y.view(60, 120, 3)
    x = dataset.unstandardize(x)[:,:,0].numpy()
    y = dataset.unstandardize(y)[:,:,0].numpy()
    lats = dataset.lats[1:-1, 1:]
    climatology_field = filename_to_climatology(
        dataset.file_names[0][0],
    )
    rmse = np.sqrt(np.mean((x - y) ** 2))
    #print(rmse)
    #print(spherical_weighted_rmse(x, y, lats))
    print(spherical_weighted_acc(y, x, climatology_field, lats))
