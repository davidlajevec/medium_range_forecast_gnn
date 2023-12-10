import numpy as np

def spherical_distance(point1, point2, radius=1):
    phi1, theta1 = point1
    phi2, theta2 = point2
    delta_phi = phi2 - phi1
    delta_theta = theta2 - theta1
    a = np.sin(delta_theta / 2) ** 2 + np.cos(theta1) * np.cos(theta2) * np.sin(delta_phi / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c

def phi_angle_difference(phi1, phi2):
    # phi1 is from the main node
    # phi2 is from the neighboring node
    return - (phi2 - phi1 + 180) % 360 - 180

def theta_angle_difference(theta1, theta2):
    # theta1 is from the main node
    # theta2 is from the neighboring node
    return (theta1 - theta2)

def over_the_top_neighbors(i, n, xdim=120):
    return (np.linspace(0, xdim-1, n+2, dtype=int)[1:-1] + i ) % (xdim-1)

def calculate_edge_attribute(main_node_ind, neighbor_node_ind, points_theta_phi):
    d1 = spherical_distance(points_theta_phi[main_node_ind], points_theta_phi[neighbor_node_ind])
    d2 = phi_angle_difference(points_theta_phi[main_node_ind][1], points_theta_phi[neighbor_node_ind][1])
    d3 = theta_angle_difference(points_theta_phi[main_node_ind][0], points_theta_phi[neighbor_node_ind][0])
    return [d1, d2, d3]

def add_k1_north_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):

        ## North pole 

        # Latitudinal neighbors

        edge_index.append([i, (i-1)%120])
        edge_attrs.append(edge_attribute_assignment_fun(i, (i-1)%120))
    
        edge_index.append([i, (i+1)%120])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+1))

        # Southern neighbors

        # LeftSouth
        if i == 0:
            edge_index.append([i, i+2*x_dim-1])
            edge_attrs.append(edge_attribute_assignment_fun(i, i+2*x_dim-1))
        else:
            edge_index.append([i, i+x_dim-1])
            edge_attrs.append(edge_attribute_assignment_fun(i, i+x_dim-1))

        # MiddleSouth
        edge_index.append([i, i+x_dim])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+x_dim))

        # RightSouth
        if i == x_dim-1:
            edge_index.append([i, (i+x_dim+1)%x_dim+x_dim])
            edge_attrs.append(edge_attribute_assignment_fun(i, (i+x_dim+1)%x_dim+x_dim))
        else:
            edge_index.append([i, i+x_dim+1])
            edge_attrs.append(edge_attribute_assignment_fun(i, i+x_dim+1))

        # Over the top neighbors
        ith_point_over_the_top_neighbors = over_the_top_neighbors(i, 3)
        for j in ith_point_over_the_top_neighbors:
            edge_index.append([i, j])
            edge_attrs.append(edge_attribute_assignment_fun(i, j))

    return edge_index, edge_attrs

def add_k1_south_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        ## South pole

        # Latitudinal neighbors
        if i == 0:
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim)-1])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim)-1))
        else:
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)-1])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)-1))
        
        if i == x_dim-1:
            edge_index.append([i+x_dim*(y_dim-1), (i+x_dim*(y_dim-1)+1)%x_dim + x_dim*(y_dim-1)])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), ((i+1)%x_dim+x_dim*(y_dim-1))%x_dim+x_dim*(y_dim-1)))
        else:
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)+1])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), (i+1)%x_dim+x_dim*(y_dim-1)))

        # NorthLeft
        if i == 0:
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)-1])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)-1))
        else:
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)-1])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)-1))

        # NorthMiddle
        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)))

        # NorthRight
        if i == x_dim-1:
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+1-x_dim])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+1-x_dim))
        else: 
            edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+1])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+1))


        ith_point_under_the_bottom_neighbors = over_the_top_neighbors(i, 3)
        for j in ith_point_under_the_bottom_neighbors:
            
            edge_index.append([i+x_dim*(y_dim-1), j + x_dim*(y_dim-1)])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), j + x_dim*(y_dim-1)))
    
    return edge_index, edge_attrs

def add_k1_middle_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    ## Middle 
    for i in range(1, y_dim-1):
        for j in range(x_dim):
            node_index = i * x_dim + j

            # Neighbors: left, right, up, down, diagonals
            neighbors = [
                (i, (j - 1) % x_dim),  # Left
                (i, (j + 1) % x_dim),  # Right
                ((i - 1) % y_dim, j),  # Up
                ((i + 1) % y_dim, j),  # Down
                ((i - 1) % y_dim, (j - 1) % x_dim),  # Left Up
                ((i - 1) % y_dim, (j + 1) % x_dim),  # Right Up
                ((i + 1) % y_dim, (j - 1) % x_dim),  # Left Down
                ((i + 1) % y_dim, (j + 1) % x_dim)   # Right Down
            ]

            for neighbor in neighbors:
                neighbor_index = neighbor[0] * x_dim + neighbor[1]
                edge_index.append([node_index, neighbor_index])
                edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))
    return edge_index, edge_attrs

def add_k2_north_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        ## North pole 

        # Souther neighbors
        edge_index.append([i, i+x_dim+2])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+x_dim+2))

        edge_index.append([i, i+x_dim-2])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+2*x_dim-2))

        edge_index.append([i, i+2*x_dim-1])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+3*x_dim-1))

        edge_index.append([i, i+2*x_dim+1])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+2*x_dim+1))

        edge_index.append([i, i+2*x_dim])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+2*x_dim))

        # Latitudinal neighbors
        edge_index.append([i, i+2])
        edge_attrs.append(edge_attribute_assignment_fun(i, i+1))

        edge_index.append([i, (i-2)%x_dim])
        edge_attrs.append(edge_attribute_assignment_fun(i, i-1))

        ith_point_over_the_top_neighbors = over_the_top_neighbors(i, 8)
        for j in ith_point_over_the_top_neighbors:
            edge_index.append([i, j])
            edge_attrs.append(edge_attribute_assignment_fun(i, j))
    return edge_index, edge_attrs

def add_k2_south_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        ## South pole

        # Latitudinal neighbors
        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)-2])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)-2))

        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-1)+2])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), (i+2)%x_dim+x_dim*(y_dim-1)))

        # NorthLeftLeft

        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)-2])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)-2))

        # NorthRightRight

        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+2])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+2))

        # NorthNorthLeft

        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)-1])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)-1))

        # NorthNorthMiddle

        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)))

        # NorthNorthRight

        edge_index.append([i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+1])
        edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), i+x_dim*(y_dim-2)+1))

        ith_point_under_the_bottom_neighbors = over_the_top_neighbors(i+x_dim*(y_dim-1), 8)
            
        for j in ith_point_under_the_bottom_neighbors:
            edge_index.append([i+x_dim*(y_dim-1), j + x_dim*(y_dim-1)])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), j + x_dim*(y_dim-1)))
    return edge_index, edge_attrs

def create_neighbooring_edges(k = 1):
    # Create points on a sphere with 3-degree resolution
    phi = np.arange(0, 2 * np.pi, np.deg2rad(3))
    theta = np.deg2rad(90-np.arange(0.5, 180, 3))

    xyz_calculation = lambda phi, theta: (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta))

    x_dim, y_dim = 120, 60 # hardcoded dimensions

    edge_index, edge_attrs, points_xyz, points_theta_phi = [], [], [], []
    #### Add points
    for j in range(y_dim):
        for i in range(x_dim):
            points_xyz.append(xyz_calculation(phi[i], theta[j]))
            points_theta_phi.append((theta[j], phi[i]))

    points_xyz = np.array(points_xyz)
    points_theta_phi = np.array(points_theta_phi)

    def lambda_edge_attribute(main_node_ind, neighbor_node_ind):
        return calculate_edge_attribute(main_node_ind, neighbor_node_ind, points_theta_phi)

    if k == 1:
        edge_index_north_k1, edge_attrs_north_k1 = add_k1_north_pole_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_north_k1
        edge_attrs += edge_attrs_north_k1

        edge_index_middle_k1, edge_attrs_middle_k1 = add_k1_middle_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_middle_k1
        edge_attrs += edge_attrs_middle_k1

        edge_index_south_k1, edge_attrs_south_k1 = add_k1_south_pole_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_south_k1
        edge_attrs += edge_attrs_south_k1

    if k == 2:
        edge_index_north_k2, edge_attrs_north_k2 = add_k2_north_pole_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_north_k2
        edge_attrs += edge_attrs_north_k2
        #print("Not implemented yet!")

    edge_index = np.array(edge_index).T
    edge_attrs = np.array(edge_attrs)
    return edge_index, edge_attrs, points_xyz, points_theta_phi


if __name__ == "__main__": 
    edge_index, edge_attrs, points_xyz, points_theta_phi = create_neighbooring_edges(k=1)
    for i in range(60*120-1):
        count = np.count_nonzero(edge_index[0] == i)
        if count != 8:
            print(i, count, 0)

        count = np.count_nonzero(edge_index[1] == i)
        if count != 8:
            print(i, count, 1)