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
    return (np.linspace(0, xdim-1, n+2, dtype=int)[1:-1] + i ) % xdim

def calculate_edge_attribute(main_node_ind, neighbor_node_ind, points_theta_phi):
    d1 = spherical_distance(points_theta_phi[main_node_ind], points_theta_phi[neighbor_node_ind])
    d2 = phi_angle_difference(points_theta_phi[main_node_ind][1], points_theta_phi[neighbor_node_ind][1])
    d3 = theta_angle_difference(points_theta_phi[main_node_ind][0], points_theta_phi[neighbor_node_ind][0])
    return [d1, d2, d3]

def add_k1_north_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        node_index = i

        neighbors = [
            # Latitudinal neighbors
            (i, (i - 1) % x_dim),  # Left
            (i, (i + 1) % x_dim),  # Right

            # Southern neighbors
            (i, x_dim + (i - 1) % x_dim),  # LeftSouth
            (i, i + x_dim),  # MiddleSouth
            (i, x_dim + (i + 1) % x_dim),  # RightSouth
        ]

        for neighbor in neighbors:
            neighbor_index = neighbor[1]
            edge_index.append([node_index, neighbor_index])
            edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))

        # Over the top neighbors
        ith_point_over_the_top_neighbors = over_the_top_neighbors(i, 3)

        for j in ith_point_over_the_top_neighbors:
            edge_index.append([i, j])
            edge_attrs.append(edge_attribute_assignment_fun(i, j))

    return edge_index, edge_attrs

def add_k1_south_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        node_index = i + x_dim * (y_dim - 1)
        
        neighbors = [
            # Latitudinal neighbors
            (i + x_dim * (y_dim - 1), (i - 1) % x_dim + x_dim * (y_dim - 1)),  # Left
            (i + x_dim * (y_dim - 1), (i + 1) % x_dim + x_dim * (y_dim - 1)),  # Right

            # Northern neighbors
            (i + x_dim * (y_dim - 1), (i - 1) % x_dim + x_dim * (y_dim - 2)),  # LeftNorth
            (i + x_dim * (y_dim - 1), i + x_dim * (y_dim - 2)),  # MiddleNorth
            (i + x_dim * (y_dim - 1), (i + 1) % x_dim + x_dim * (y_dim - 2)),  # RightNorth
        ]

        for neighbor in neighbors:
            neighbor_index = neighbor[1]
            edge_index.append([node_index, neighbor_index])
            edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))

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
        node_index = i

        neighbors = [
            # Latitudinal neighbors
            (i, (i - 1) % x_dim),  # Left
            (i, (i - 2) % x_dim),  # LeftLeft
            (i, (i + 1) % x_dim),  # Right
            (i, (i + 2) % x_dim),  # RightRight

            # Southern neighbors
            (i, x_dim + (i - 1) % x_dim),  # LeftSouth
            (i, x_dim + (i - 2) % x_dim),  # LeftLeftSouth
            (i, x_dim * 2 + (i - 1) % x_dim),  # LeftSouthSouth
            (i, i + x_dim),  # MiddleSouth
            (i, i + x_dim * 2),  # MiddleSouthSouth
            (i, x_dim + (i + 1) % x_dim),  # RightSouth
            (i, x_dim + (i + 2) % x_dim),  # RightRightSouth
            (i, x_dim * 2 + (i + 1) % x_dim),  # RightSouthSouth
        ]

        for neighbor in neighbors:
            neighbor_index = neighbor[1]
            edge_index.append([node_index, neighbor_index])
            edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))

        # Over the top neighbors
        ith_point_over_the_top_neighbors = over_the_top_neighbors(i, 8)

        for j in ith_point_over_the_top_neighbors:
            edge_index.append([i, j])
            edge_attrs.append(edge_attribute_assignment_fun(i, j))

    return edge_index, edge_attrs

def add_k2_north_2nd_row_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        node_index = i + x_dim

        neighbors = [
            # Latitudinal neighbors
            (i + x_dim, (i - 1) % x_dim + x_dim),  # Left
            (i + x_dim, (i - 2) % x_dim + x_dim),  # LeftLeft
            (i + x_dim, (i + 1) % x_dim + x_dim),  # Right
            (i + x_dim, (i + 2) % x_dim + x_dim),  # RightRight

            # Southern neighbors
            (node_index, (i - 1) % x_dim + x_dim * 2),  # LeftSouth
            (node_index, (i - 1) % x_dim + x_dim * 3),  # LeftSouthSouth
            (node_index, (i - 2) % x_dim + x_dim * 2),  # LeftLeftSouth
            (node_index, i % x_dim + x_dim * 2),  # MiddleSouth
            (node_index, i % x_dim + x_dim * 3),  # MiddleSouthSouth
            (node_index, (i + 1) % x_dim + x_dim * 2),  # RightSouth
            (node_index, (i + 1) % x_dim + x_dim * 3),  # RightSouthSouth
            (node_index, (i + 2) % x_dim + x_dim * 2),  # RightRightSouth

            # Northern neighbors
            (i + x_dim, (i - 1) % x_dim),  # LeftNorth
            (i + x_dim, (i - 2) % x_dim),  # LeftLeftNorth
            (i + x_dim, (i + 1) % x_dim),  # RightNorth
            (i + x_dim, (i + 2) % x_dim),  # RightRightNorth
            (i + x_dim, i),  # MiddleNorth
        ]

        for neighbor in neighbors:
            neighbor_index = neighbor[1]
            edge_index.append([node_index, neighbor_index])
            edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))

        # Over the top neighbors
        ith_point_over_the_top_neighbors = over_the_top_neighbors(i, 3)

        for j in ith_point_over_the_top_neighbors:
            edge_index.append([i+x_dim, j + x_dim])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim, j + x_dim))

    return edge_index, edge_attrs

def add_k2_middle_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(2, y_dim-2):
        for j in range(x_dim):
            node_index = i * x_dim + j

            # Neighbors: left, right, up, down, diagonals
            neighbors = [
                (i, (j - 1) % x_dim),  # Left
                (i, (j - 2) % x_dim),  # LeftLeft

                (i, (j + 1) % x_dim),  # Right
                (i, (j + 2) % x_dim),  # RightRight

                ((i - 1) % y_dim, j),  # Up
                ((i - 2) % y_dim, j),  # UpUp

                ((i + 1) % y_dim, j),  # Down
                ((i + 2) % y_dim, j),  # DownDown

                ((i - 1) % y_dim, (j - 1) % x_dim),  # Left Up
                ((i - 1) % y_dim, (j - 2) % x_dim),  # LeftLeft Up
                ((i - 2) % y_dim, (j - 1) % x_dim),  # Left UpUp

                ((i - 1) % y_dim, (j + 1) % x_dim),  # Right Up
                ((i - 1) % y_dim, (j + 2) % x_dim),  # RightRight Up
                ((i - 2) % y_dim, (j + 1) % x_dim),  # Right UpUp

                ((i + 1) % y_dim, (j - 1) % x_dim),  # Left Down
                ((i + 1) % y_dim, (j - 2) % x_dim),  # LeftLeft Down
                ((i + 2) % y_dim, (j - 1) % x_dim),  # Left DownDown

                ((i + 1) % y_dim, (j + 1) % x_dim),   # Right Down
                ((i + 1) % y_dim, (j + 2) % x_dim),   # RightRight Down
                ((i + 2) % y_dim, (j + 1) % x_dim),   # Right DownDown
            ]

            for neighbor in neighbors:
                neighbor_index = neighbor[0] * x_dim + neighbor[1]
                edge_index.append([node_index, neighbor_index])
                edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))
    return edge_index, edge_attrs

def add_k2_south_2nd_row_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        node_index = i + x_dim * (y_dim - 2)
        neighboors = [
            # Latitudinal neighbors
            (i + x_dim * (y_dim - 2), (i - 1) % x_dim + x_dim * (y_dim - 2)),  # Left
            (i + x_dim * (y_dim - 2), (i - 2) % x_dim + x_dim * (y_dim - 2)),  # LeftLeft
            (i + x_dim * (y_dim - 2), (i + 1) % x_dim + x_dim * (y_dim - 2)),  # Right
            (i + x_dim * (y_dim - 2), (i + 2) % x_dim + x_dim * (y_dim - 2)),  # RightRight

            # Southern neighbors
            (i + x_dim * (y_dim - 2), x_dim + (i - 1) % x_dim + x_dim * (y_dim - 2)),  # LeftSouth
            (i + x_dim * (y_dim - 2), x_dim + (i - 2) % x_dim + x_dim * (y_dim - 2)),  # LeftLeftSouth
            (i + x_dim * (y_dim - 2), i + x_dim + x_dim * (y_dim - 2)),  # MiddleSouth
            (i + x_dim * (y_dim - 2), x_dim + (i + 1) % x_dim + x_dim * (y_dim - 2)),  # RightSouth
            (i + x_dim * (y_dim - 2), x_dim + (i + 2) % x_dim + x_dim * (y_dim - 2)),  # RightRightSouth

            # Northern neighbors
            (i + x_dim * (y_dim - 2), (i - 1) % x_dim + x_dim * (y_dim - 3)),  # LeftNorth
            (i + x_dim * (y_dim - 2), (i - 1) % x_dim + x_dim * (y_dim - 4)),  # LeftNorthNorth
            (i + x_dim * (y_dim - 2), (i - 2) % x_dim + x_dim * (y_dim - 3)),  # LeftLeftNorth
            (i + x_dim * (y_dim - 2), (i + 1) % x_dim + x_dim * (y_dim - 3)),  # RightNorth
            (i + x_dim * (y_dim - 2), (i + 2) % x_dim + x_dim * (y_dim - 3)),  # RightRightNorth
            (i + x_dim * (y_dim - 2), (i + 1) % x_dim + x_dim * (y_dim - 4)),  # RightNorthNorth
            (i + x_dim * (y_dim - 2), i % x_dim + x_dim * (y_dim - 3)),  # MiddleNorth          
            (i + x_dim * (y_dim - 2), i % x_dim + x_dim * (y_dim - 4)),  # MiddleNorthNorth
        ]

        for neighbor in neighboors:
            neighbor_index = neighbor[1]
            edge_index.append([node_index, neighbor_index])
            edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))

        ith_point_over_the_top_neighbors = over_the_top_neighbors(i, 3)

        for j in ith_point_over_the_top_neighbors:
            edge_index.append([i+x_dim*(y_dim-2), j + x_dim*(y_dim-2)])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-2), j + x_dim*(y_dim-2)))
    
    return edge_index, edge_attrs

def add_k2_south_pole_connections(x_dim, y_dim, edge_attribute_assignment_fun):
    edge_index, edge_attrs = [], []
    for i in range(x_dim):
        node_index = i + x_dim * (y_dim - 1)
        
        neighbors = [
            # Latitudinal neighbors
            (i + x_dim * (y_dim - 1), (i - 1) % x_dim + x_dim * (y_dim - 1)),  # Left
            (i + x_dim * (y_dim - 1), (i - 2) % x_dim + x_dim * (y_dim - 1)),  # LeftLeft
            (i + x_dim * (y_dim - 1), (i + 1) % x_dim + x_dim * (y_dim - 1)),  # Right
            (i + x_dim * (y_dim - 1), (i + 2) % x_dim + x_dim * (y_dim - 1)),  # RightRight

            # Northern neighbors
            (i + x_dim * (y_dim - 1), (i - 1) % x_dim + x_dim * (y_dim - 2)),  # LeftNorth
            (i + x_dim * (y_dim - 1), (i - 2) % x_dim + x_dim * (y_dim - 2)),  # LeftLeftNorth
            (i + x_dim * (y_dim - 1), (i - 1) % x_dim + x_dim * (y_dim - 3)),  # LeftNorthNorth
            (i + x_dim * (y_dim - 1), (i + 1) % x_dim + x_dim * (y_dim - 2)),  # RightNorth
            (i + x_dim * (y_dim - 1), (i + 2) % x_dim + x_dim * (y_dim - 2)),  # RightRightNorth
            (i + x_dim * (y_dim - 1), (i + 1) % x_dim + x_dim * (y_dim - 3)),  # RightNorthNorth
            (i + x_dim * (y_dim - 1), i % x_dim + x_dim * (y_dim - 2)),  # MiddleNorth
            (i + x_dim * (y_dim - 1), i % x_dim + x_dim * (y_dim - 3)),  # MiddleNorthNorth
        ]

        for neighbor in neighbors:
            neighbor_index = neighbor[1]
            edge_index.append([node_index, neighbor_index])
            edge_attrs.append(edge_attribute_assignment_fun(node_index, neighbor_index))
        
        ith_point_under_the_bottom_neighbors = over_the_top_neighbors(i, 8)

        for j in ith_point_under_the_bottom_neighbors:
            edge_index.append([i+x_dim*(y_dim-1), j + x_dim*(y_dim-1)])
            edge_attrs.append(edge_attribute_assignment_fun(i+x_dim*(y_dim-1), j + x_dim*(y_dim-1)))
        
    return edge_index, edge_attrs



def create_neighbooring_edges(k = 1, x_dim=120, y_dim=60):
    # Create points on a sphere with 3-degree resolution

    edge_index, edge_attrs, points_xyz, points_theta_phi = [], [], [], []

    phi = np.arange(0, 2 * np.pi, np.deg2rad(3))
    theta = np.deg2rad(90-np.arange(0.5, 180, 3))

    xyz_calculation = lambda phi, theta: (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta))

    ## Add points
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
        # North pole 1st row
        edge_index_north_k2, edge_attrs_north_k2 = add_k2_north_pole_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_north_k2
        edge_attrs += edge_attrs_north_k2

        # North pole 2nd row
        edge_index_north_2nd_row_k2, edge_attrs_north_2nd_row_k2 = add_k2_north_2nd_row_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_north_2nd_row_k2
        edge_attrs += edge_attrs_north_2nd_row_k2

        # Middle neighbors
        edge_index_middle_k2, edge_attrs_middle_k2 = add_k2_middle_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_middle_k2
        edge_attrs += edge_attrs_middle_k2

        # South pole 2nd row
        edge_index_south_2nd_row_k2, edge_attrs_south_2nd_row_k2 = add_k2_south_2nd_row_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_south_2nd_row_k2
        edge_attrs += edge_attrs_south_2nd_row_k2

        # South pole 1st row
        edge_index_south_k2, edge_attrs_south_k2 = add_k2_south_pole_connections(x_dim, y_dim, lambda_edge_attribute)
        edge_index += edge_index_south_k2
        edge_attrs += edge_attrs_south_k2

    edge_index = np.array(edge_index).T
    edge_attrs = np.array(edge_attrs)
    return edge_index, edge_attrs, points_xyz, points_theta_phi


if __name__ == "__main__": 
    edge_index, edge_attrs, points_xyz, points_theta_phi = create_neighbooring_edges(k=2)

    for i in range(60*120-1):
        count = np.count_nonzero(edge_index[0] == i)
        if count != 20:
            print(i, count, 0)
        count = np.count_nonzero(edge_index[1] == i)
        if count != 20:
            print(i, count, 1)