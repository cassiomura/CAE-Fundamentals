# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                           Title: FiniteElement.py
# ==============================================================================
import numpy as np
import pandas as pd

class FiniteElement:

    DOF_PER_NODE_MAP = {
        'CTRIA3': 2,
        'CQUAD4': 2,
        'CROD': 2,
        'CBAR': 6,
        'CTETRA': 3,
        'CHEXA': 3
    }

    def __init__(self, row, df_nodes, df_properties, df_materials):
        self.row = row
        self.element_type = row['Element Type']
        self.element_id = row['Element ID']
        self.element_property_id = row['Property']
        self.connectivity = np.array(list(map(int, row['Connectivity'].split())), dtype=int) - 1 # Ajusta para um array NumPy
        self.node_coordinates = self.get_node_coordinates(df_nodes)

        # Get the DOF per node from the class dictionary
        self.dof_per_node = self.DOF_PER_NODE_MAP.get(self.element_type)

        # Return the vector that contains the index (at the np.array vector) in the global:
        self.global_index_dofs = self.get_global_index_dofs(df_nodes)

    
    def get_node_coordinates(self, df_nodes: pd.DataFrame) -> np.array:
        # Extract the connectivity of the element as a list of node IDs
        connectivity_string = self.row['Connectivity']
        connected_node_ids = list(map(int, connectivity_string.split()))

        # Initialize a list to hold the coordinates of the connected nodes
        connected_node_coordinates = []

        # Iterate over the connected node IDs and retrieve their coordinates
        for node_id in connected_node_ids:
            node_coordinates = df_nodes[df_nodes['Node ID'] == node_id][['X', 'Y', 'Z']].values[0]
            connected_node_coordinates.append(node_coordinates)

        # Convert the list of coordinates to a NumPy array for consistency
        return np.array(connected_node_coordinates)
    
    def get_global_index_dofs(self, df_nodes: pd.DataFrame) -> np.array:
        # Get the nodes index in the df_nodes dataframe: 
        node_ids = self.connectivity + 1
        global_index_dofs_nodes = np.array([df_nodes[df_nodes['Node ID'] == node_id].index[0] for node_id in node_ids])

        num_nodes = df_nodes.shape[0] 
        # Initialize global_index_dofs as an empty list to collect results
        global_index_dofs = []

        for k in range(self.dof_per_node):
            # Append the indices for each DOF to the list
            global_index_dofs.extend(global_index_dofs_nodes + k * num_nodes)

        return global_index_dofs