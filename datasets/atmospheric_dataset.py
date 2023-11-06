import os
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeScale
import numpy as np

class AtmosphericDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        edge_index,
        edge_attributes=None,
        root="data_12hr",
        atmosphere_variables=["geopotential_500"],
        start_year=1950,
        end_year=2008,
    ):
        self.root = root
        self.edge_index = edge_index
        self.edge_attributes = edge_attributes

        with open(f"{root}/annotation_file.txt", "r") as f:
            annotations = f.readlines()

        self.atmosphere_variables = atmosphere_variables

        phi = np.rad2deg(np.arange(0, 2 * np.pi + np.deg2rad(3), np.deg2rad(3)))
        theta = 88.5 - np.arange(0, 180, 3)
        theta = np.hstack((90, theta, -90))

        self.lons, self.lats = np.meshgrid(phi, theta)

        mean_fields = []
        std_fields = []
        for variable in atmosphere_variables:
            mean = np.load(f"{root}/{variable}/{variable}_mean.npy")
            std = np.load(f"{root}/{variable}/{variable}_std.npy")
            mean_fields.append(torch.from_numpy(mean).to(torch.float))
            std_fields.append(torch.from_numpy(std).to(torch.float))
        self.mean_fields = torch.stack(mean_fields, dim=2)
        self.std_fields = torch.stack(std_fields, dim=2)

        self.file_names = [[] for variable in atmosphere_variables]

        years = list(range(start_year, end_year + 1))
        for line in annotations:
            year = int(line[2:6])
            if year in years:
                for i, variable in enumerate(atmosphere_variables):
                    self.file_names[i].append(
                        f"{variable}/{variable}_{year}/{variable}{line[1:-1]}"
                    )
    
    def __len__(self):
        return len(self.file_names[0])-1
    
    def standardize(self, x):
        return (x - self.mean_fields) / self.std_fields
    
    def unstandardize(self, x):
        return x * self.std_fields + self.mean_fields

    def process(self, idx):
        data_x, data_y = [], []
        for i, variable in enumerate(self.atmosphere_variables):
            variable_data_x = np.load(f"{self.root}/{self.file_names[i][idx]}")
            variable_data_y = np.load(f"{self.root}/{self.file_names[i][idx+1]}")
            data_x.append(torch.from_numpy(variable_data_x).to(torch.float))
            data_y.append(torch.from_numpy(variable_data_y).to(torch.float))
        
        x = torch.stack(data_x, dim=2)
        x = self.standardize(x).view(-1, len(self.atmosphere_variables))
        y = torch.stack(data_y, dim=2)
        y = self.standardize(y).view(-1, len(self.atmosphere_variables))
        return Data(x=x, edge_index=self.edge_index, y=y)
        # return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attributes, y=y)

    def __getitem__(self, idx):
        return self.process(idx)

if __name__ == "__main__":
    # lets try to load the data
    import os
    import sys

    sys.path.append(os.getcwd())
    from utils.mesh_creation import create_k_nearest_neighboors_edges
    from utils.plot_atmospheric_field import (
        plot_atmospheric_field,
        plot_true_and_predicted_atomspheric_field,
    )
    import cartopy.crs as ccrs

    edge_index, edge_attr, points = create_k_nearest_neighboors_edges(radius=1, k=8)

    dataset = AtmosphericDataset(
        root="data_12hr",
        edge_index=edge_index,
        edge_attributes=edge_attr,
        atmosphere_variables=["geopotential_500", "u_500", "v_500"],
        start_year=1950,
        end_year=2008,
    )
    #x = dataset.__getitem__(0).x.view(60, 120, 3)[:, :, 0]
    y = dataset.unstandardize(dataset.__getitem__(0).y.view(60, 120, 3))[:, :, 0]
    x = dataset.unstandardize(dataset.__getitem__(1).x.view(60, 120, 3))[:, :, 0]

    plot_true_and_predicted_atomspheric_field(
        x,
        y,
        show=True,
        show_colorbar=False,
        x_title="Longitude",
        y_title="Latitude",
        title="Geopotential 500 hPa",
        projection=ccrs.Robinson(),
        cmap="nipy_spectral",
    )
    print(dataset.__getitem__(0).x.shape)