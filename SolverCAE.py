# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                            Title: SolverCAE.py
# ==============================================================================
import numpy as np
import pandas as pd
from ElementLibrary.Unidimensional.CBAR import CBAR
from ElementLibrary.Bidimensional.CTRIA3 import CTRIA3
from ElementLibrary.Bidimensional.CQUAD4 import CQUAD4
from ElementLibrary.Tridimensional.CTETRA import CTETRA
from ElementLibrary.Tridimensional.CHEXA import CHEXA
from ElementLibrary.FiniteElement import FiniteElement

def solve_system(stiffness_matrix_bcs: np.array, force_vector_bcs: np.array, stiffness_matrix: np.array) -> tuple:
    # Solve for the displacement vector using the stiffness matrix and force vector
    displacement_vector = np.linalg.solve(stiffness_matrix_bcs, force_vector_bcs)
    displacement_vector = displacement_vector.flatten()
    
    # Solve for the force vector:
    force_vector = np.matmul(stiffness_matrix, displacement_vector)
     
    return displacement_vector, force_vector

def assemble_global_stiffness_matrix(df_elements: pd.DataFrame, df_nodes: pd.DataFrame, df_materials: pd.DataFrame, df_properties:pd.DataFrame) -> np.array:
    # Number of nodes in the mesh
    num_nodes = df_nodes.shape[0] 
    
    # Determine number of DOFs based on the first element type
    num_dofs_per_node = FiniteElement.DOF_PER_NODE_MAP.get(df_elements.iloc[0, 0])  
    num_dofs = num_dofs_per_node * num_nodes

    # Initialize the global stiffness matrix
    K_global = np.zeros((num_dofs, num_dofs))

    # Iterate through each element in the mesh
    for _, row in df_elements.iterrows():
        # Create element object based on its type
        element_type = row['Element Type']
        if element_type == 'CTRIA3':
            element = CTRIA3(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CQUAD4':
            element = CQUAD4(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CBAR':
            element = CBAR(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CTETRA':
            element = CTETRA(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CHEXA':
            element = CHEXA(row, df_nodes, df_properties, df_materials)

        # Get global DOF indices and element stiffness matrix
        global_index_dofs = element.global_index_dofs
        K_element = element.stiffness_matrix

        # Assemble local stiffness matrix into the global matrix
        for i_local, i_global in enumerate(global_index_dofs):
            for j_local, j_global in enumerate(global_index_dofs):
                K_global[i_global][j_global] += K_element[i_local][j_local]
                
    return K_global

def assemble_global_load_vector(simulation_case: str, df_nodes: pd.DataFrame, df_loads: pd.DataFrame) -> np.array:
    # Number of nodes in the mesh
    num_nodes = df_nodes.shape[0] 
    
    # Determine number of DOFs based on the first element type
    num_dofs_per_node = FiniteElement.DOF_PER_NODE_MAP.get(simulation_case)  
    num_dofs = num_dofs_per_node * num_nodes

    # Initialize the global force vector
    F_global = np.zeros((num_dofs, 1))
    # Loop for every force in the loads dataframe:
    for _, row in df_loads.iterrows():
        load_type = row['Type']
        node_id = int(row['G'])
        force_magnitude = float(row['F'])
        normal = np.array([float(row['N1']), float(row['N2']), float(row['N3'])])

        # Find node index and corresponding DOFs
        node_index = df_nodes[df_nodes['Node ID'] == node_id].index[0]
        dofs = np.arange(node_index, node_index + num_dofs_per_node * num_nodes, num_nodes)

        # Select DOFs based on load type
        if load_type == 'MOMENT':
            selected_dofs = dofs[-3:]  # Last 3 rotational DOFs
        elif load_type == 'FORCE':
            selected_dofs = dofs[:3] if num_dofs_per_node == 6 else dofs[:num_dofs_per_node]

        # Apply the force or moment
        for i, dof in enumerate(selected_dofs):
            F_global[dof] += force_magnitude * normal[i]
    
    return F_global
    
def impose_boundary_conditions_penalty_method(simulation_case: str, df_nodes: pd.DataFrame, df_boundary_conditions: pd.DataFrame, stiffness_matrix: np.array, force_vector: np.array) -> tuple:
    # Retrieve the list of degrees of freedom that are fixed based on boundary conditions
    fixed_dofs = get_boundary_conditions_dofs(simulation_case, df_nodes, df_boundary_conditions)
    
    # Set the fixed boundary conditions in the stiffness matrix and force vector
    updated_stiffness_matrix, updated_force_vector = set_fixed_boundary_conditions(fixed_dofs, stiffness_matrix, force_vector)

    return updated_stiffness_matrix, updated_force_vector

def get_boundary_conditions_dofs(simulation_case: str, df_nodes: pd.DataFrame, df_boundary_conditions: pd.DataFrame) -> np.array:
    # Number of nodes in the mesh
    num_nodes = df_nodes.shape[0] 

    # Determine number of DOFs based on the first element type
    num_dofs_per_node = FiniteElement.DOF_PER_NODE_MAP.get(simulation_case) 

    # List to store fixed DOFs
    fixed_dofs = []

    # Iterate over each boundary condition
    for _, row in df_boundary_conditions.iterrows():
        dof_types = row['C'] # Extract DOF types
        nodes = row['G'].split() # Extract node IDs

        # Process each node and its associated DOFs
        for node_id in nodes:
            node_index = df_nodes.loc[df_nodes['Node ID'] == int(node_id)].index[0]
            
            for dof_type in dof_types:
                dof_index = int(dof_type) - 1
                 # Check if dof_index is within the valid range for num_dofs_per_node
                if dof_index < num_dofs_per_node:  # Ensure dof_index does not surpass num_dofs_per_node
                    fixed_dofs.append(node_index + dof_index*num_nodes)

    # Return fixed DOFs as a NumPy array
    return np.array(fixed_dofs) 

def set_fixed_boundary_conditions(fixed_dofs: np.array, stiffness_matrix: np.array, force_vector: np.array) -> tuple:
    # Create a copy of the global stiffness matrix and global force vector 
    stiffness_matrix_ = stiffness_matrix.copy()
    force_vector_ = force_vector.copy()

    # Iterate over each fixed degree of freedom
    for fixed_dof in fixed_dofs:
        # Set the corresponding row and column in the stiffness matrix to zero (except for the diagonal element)
        stiffness_matrix_[fixed_dof, :] = 0
        stiffness_matrix_[:, fixed_dof] = 0
        stiffness_matrix_[fixed_dof, fixed_dof] = 1  # Set the diagonal element to 1

        # Set the corresponding force in the force vector to zero
        force_vector_[fixed_dof] = 0

    return stiffness_matrix_, force_vector_