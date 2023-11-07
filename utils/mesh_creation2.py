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

    # Find the indices of eight neighbors for each point
    indices = []
    for i in range(len(points)):
        neighbors = []

        # Diagonal neighbors
        if i % 2 == 0:  # Even rows
            neighbors.extend([i - 1, i + 1, i - 12, i - 11, i + 12, i + 13])
        else:  # Odd rows
            neighbors.extend([i - 1, i + 1, i - 12, i - 13, i + 12, i + 11])

        # Up and down neighbors
        neighbors.extend([i - 12, i + 12])

        # Remove invalid neighbors (out of bounds)
        valid_neighbors = [n for n in neighbors if 0 <= n < len(points)]

        indices.append(valid_neighbors)

    # Create edge_index and edge_attrs
    row = []
    col = []
    for i, neighbor_indices in enumerate(indices):
        row.extend([i] * len(neighbor_indices))
        col.extend(neighbor_indices)

    edge_index = np.vstack((row, col))
    edge_attrs = dists[row, col]

    return edge_index, edge_attrs, points


if __name__ == "__main__":
    edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=4)
    print(edge_index.shape)