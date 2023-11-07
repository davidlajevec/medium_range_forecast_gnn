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

