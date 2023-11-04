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

def create_k_nearest_neighboors_edges(radius=1, k=12):
    # Create points on sphere with 3 degree resolution
    phi = np.arange(0, 2 * np.pi, np.deg2rad(3))
    theta = np.deg2rad(-88.5 + np.arange(0, 180, 3))
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    # Compute pairwise distances between points
    dists = cdist(points, points)# metric=spherical_distance)
    # Find k nearest neighbors for each point
    indices = np.argsort(dists, axis=1)[:, 1:k+1]
    # Compute edge attributes as spherical distance between points
    edge_attrs = dists[indices]
    # Create edge_index sparse matrix
    row = np.repeat(np.arange(len(points)), k)
    col = indices.flatten()
    edge_index = np.vstack((row, col))
    return edge_index, edge_attrs, points

if __name__ == "__main__":
    edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=4)
    print(edge_index.shape)