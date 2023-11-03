import os
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import NormalizeScale
import numpy as np


class AtmosphereDatasets(torch.utils.data.Dataset):
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

        self.file_names = [[] for variable in atmosphere_variables]

        years = list(range(start_year, end_year + 1))
        for line in annotations:
            year = int(line[2:6])
            if year in years:
                for i, variable in enumerate(atmosphere_variables):
                    self.file_names[i].append(
                        f"{variable}/{variable}_{year}/{variable}{line[1:-1]}"
                    )
        if end_year == 2022:
            for i, variable in enumerate(atmosphere_variables):
                self.file_names[i].pop(-1)

    def standardize(self):
        pass

    def process(self, idx):
        # data_list = []
        # for raw_path in self.raw_paths:
        #    raw_data = np.load(raw_path)
        #    x = torch.from_numpy(raw_data).to(torch.float)
        # return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attr)
        data_x, data_y = [], []
        for i, variable in enumerate(self.atmosphere_variables):
            variable_data_x = np.load(f"{self.root}/{self.file_names[i][idx]}")
            variable_data_y = np.load(f"{self.root}/{self.file_names[i][idx+1]}")
            data_x.append(torch.from_numpy(variable_data_x).to(torch.float))
            data_y.append(torch.from_numpy(variable_data_y).to(torch.float))
        x = torch.stack(data_x, dim=2).view(-1, 2)
        y = torch.stack(data_y, dim=2).view(-1, 2)
        return Data(x=x, y=y)
        # return Data(x=x, edge_index=self.edge_index, edge_attr=self.edge_attributes, y=y)

    def len(self):
        return len(self.file_names[0])

    def get(self, idx):
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

    dataset = AtmosphereDatasets(
        root="data_12hr",
        edge_index=edge_index,
        edge_attributes=edge_attr,
        atmosphere_variables=["geopotential_500", "u_500", "v_500"],
        start_year=2022,
        end_year=2022,
    )

    x = dataset.get(0).x.view(60, 120, 3)[:, :, 0]
    y = dataset.get(0).y.view(60, 120, 3)[:, :, 0]

    # plot_atmospheric_field(x, projection=ccrs.Robinson(), show=True)
    # plot_true_and_predicted_atomspheric_field(x, y, projection=ccrs.Robinson(), show=True)
    plot_true_and_predicted_atomspheric_field(
        x,
        y,
        show=True,
        x_title="Longitude",
        y_title="Latitude",
        title="Geopotential 500 hPa",
        projection=ccrs.Robinson(),
    )
