import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mesh_creation import create_neighbooring_edges

PLOT_EDGE_INDEX = 60*120-1
K = 2
SHOW = True

edge_index, edge_attrs, points, points_theta_phi = create_neighbooring_edges(k=K)

x_data = points[:, 0]
y_data = points[:, 1]
z_data = points[:, 2]

# Replace the Plotly code with Matplotlib code
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a sphere
phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 20)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Plot the sphere
#ax.plot_surface(x, y, z, color='b', alpha=0.2)

# Filter and plot the edges
def filter_pairs(adj_matrix, index=0):
    filtered_pairs = [[adj_matrix[0][i], adj_matrix[1][i]] for i in range(len(adj_matrix[0])) if adj_matrix[0][i] == index]
    return np.array(filtered_pairs).T

filtered_edge_index = filter_pairs(edge_index, index=PLOT_EDGE_INDEX)   
print(' '.join(map(str, filtered_edge_index[0, :])))
print(' '.join(map(str, filtered_edge_index[1, :])))


for i, j in zip(filtered_edge_index[0, :], filtered_edge_index[1, :]):
    ax.plot([x_data[i], x_data[j]], [y_data[i], y_data[j]], [z_data[i], z_data[j]], color='black')

# Plot the points
ax.scatter(x_data, y_data, z_data, color='black', s=3)  # s is the marker size

# Set the title and aspect ratio
ax.set_title('Graph Embedded on Sphere')
ax.set_box_aspect([1, 1, 1])  # Matplotlib < 3.2.0 may not support set_box_aspect

if SHOW:
    plt.show()

