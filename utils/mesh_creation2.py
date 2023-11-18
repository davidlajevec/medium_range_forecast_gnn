import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
import plotly.graph_objs as go
from plotly.offline import plot

def spherical_distance(point1, point2, radius=1):
    """
    Computes the spherical distance between two points on a sphere with given radius.
    """
    phi1, theta1 = point1
    phi2, theta2 = point2
    delta_phi = phi2 - phi1
    delta_theta = theta2 - theta1
    a = np.sin(delta_theta / 2) ** 2 + np.cos(theta1) * np.cos(theta2) * np.sin(delta_phi / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

spherical_distance = np.vectorize(spherical_distance)

def create_8_neighboors_edges(radius=1):
    # Create points on a sphere with 3-degree resolution
    phi = np.arange(0, 2 * np.pi, np.deg2rad(3))
    theta = np.deg2rad(np.arange(0.5, 180, 3))
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Compute pairwise distances between points
    dists = cdist(points, points)

    num_points = len(points)
    num_phi = len(np.unique(phi))

    # Find the indices of eight neighbors for each point
    indices = []
    for i in range(len(points)):
        neighbors = []

        # Diagonal neighbors
        if i % num_phi == 0:  # First phi column
            neighbors.extend([i + num_phi - 1, i + 1, i + num_points - num_phi - 1, i + num_points - num_phi, i + num_points - 1, i + num_points])
        elif (i + 1) % num_phi == 0:  # Last phi column
            neighbors.extend([i - 1, i - num_phi + 1, i - num_points + num_phi - 1, i - num_points + num_phi, i - num_points - 1, i - num_points])
        else:
            neighbors.extend([i - 1, i + 1, i - num_phi + 1, i - num_phi, i + num_phi - 1, i + num_phi])

        # Up and down neighbors
        if i >= num_phi:
            neighbors.extend([i - num_phi, i + num_phi])

        indices.append(neighbors)

    # Create edge_index and edge_attrs
    row = []
    col = []
    for i, neighbor_indices in enumerate(indices):
        for neighbor_index in neighbor_indices:
            if neighbor_index < num_points:
                row.append(i)
                col.append(neighbor_index)

    edge_index = np.vstack((row, col))
    edge_attrs = dists[row, col]

    return edge_index, edge_attrs, points


if __name__ == "__main__":
    edge_index, edge_attrs, points = create_8_neighboors_edges(radius=1)
    print(edge_index.shape)