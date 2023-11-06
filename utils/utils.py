import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out

import numpy as np

def filename_to_climatology(data_filename, data_root="data_12hr"):
    """
    Given a data filename, returns the corresponding climatology data.

    Args:
        data_filename (str): The filename of the data file.
        data_root (str): The root directory of the data files. Defaults to "data_12hr".

    Returns:
        numpy.ndarray: The climatology data corresponding to the given data filename.
    """
    data_filename = data_filename.split("/")
    variable = data_filename[0]
    date_str = data_filename[-1].split("_")[-1][:-4]
    year, month, day, hour = map(int, date_str.split("-"))
    #days_in_month = [31, 28 if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0) else 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_so_far = sum(days_in_month[:month-1]) + day 
    if month == 2 and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) and day == 29:
        days_so_far -= 1
    climatology_path = f"{data_root}/{variable}/climatology_{days_so_far}_{hour}.npy"
    climatology = np.load(climatology_path)
    return climatology


def spherical_weight(lat):
    """
    Computes the spherical weight for a given latitude.

    Args:
        lat (numpy.ndarray): Array of latitudes in degrees.

    Returns:
        numpy.ndarray: Array of spherical weights.
    """
    cos_lat = np.cos(np.radians(lat)) 
    normalization = np.sum(cos_lat[:,0])*lat.shape[0]
    return cos_lat/normalization

def spherical_weighted_rmse(y_true, y_pred, lat):
    """
    Calculate the spherical weighted root mean squared error (RMSE) between two arrays of atmospheric variables.
    """
    weights = spherical_weight(lat)
    rmse = np.sqrt(np.mean(weights * (y_true - y_pred) ** 2))
    return rmse

def spherical_weighted_acc(y_true, y_pred, climatology_field, lat):
    """
    Calculate the spherical weighted anomaly correlation coefficient (ACC) between two arrays of atmospheric variables.
    """
    weights = spherical_weight(lat)
    numerator = np.sum(weights * (y_true - climatology_field) * (y_pred - climatology_field))
    denominator = np.sqrt(np.sum(weights * (y_true - climatology_field) ** 2) * np.sum(weights * (y_pred - climatology_field) ** 2))
    acc = numerator / denominator
    return acc