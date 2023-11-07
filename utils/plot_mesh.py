import plotly.graph_objects as go
import numpy as np
import scipy.sparse as sp
from mesh_creation import create_k_nearest_neighboors_edges
#from mesh_creation2 import create_8_neighboors_edges

edge_index, edge_attrs, points = create_k_nearest_neighboors_edges(radius=1, k=12)
#edge_index, edge_attrs, points = create_8_neighboors_edges(radius=1)


# Create a sphere
phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2*np.pi, 20)
phi, theta = np.meshgrid(phi, theta)
x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

# Create a 3D plot
fig = go.Figure()

# Add sphere trace
fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.2))

# Add graph trace
x_data = points[:, 0]
y_data = points[:, 1]
z_data = points[:, 2]
print(edge_index.shape)


# Create edge index sparse matrix
#edge_index = sp.coo_matrix(edge_index)

edge_x = []
edge_y = []
edge_z = []


#for i, j in zip(edge_index.row, edge_index.col):
for i, j in zip(edge_index[0,:], edge_index[1,:]):
    #print("i:{},j:{}".format(i,j))
    edge_x += [x_data[i], x_data[j], None]
    edge_y += [y_data[i], y_data[j], None]
    edge_z += [z_data[i], z_data[j], None]

# Add edges to graph trace
fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color='black', width=4)))

# Modify marker size of nodes
fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data, mode='markers', marker=dict(color='black', size=3)))

# Set layout
fig.update_layout(title='Graph Embedded on Sphere', scene=dict(aspectmode='data', aspectratio=dict(x=1, y=1, z=1)))

fig.show()

