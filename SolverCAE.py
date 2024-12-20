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

def solve_linear_static_analysis(method: str, simulation_case: str, df_nodes: pd.DataFrame, df_boundary_conditions: pd.DataFrame, stiffness_matrix: np.array, force_vector: np.array):
    fixed_dofs = get_boundary_conditions_dofs(simulation_case, df_nodes, df_boundary_conditions)

    if method == "Penalty Method":
        displacement_vector, force_vector = solve_system_penalty_method(stiffness_matrix, force_vector, fixed_dofs)
    elif method == "Reduction Method":
        displacement_vector, force_vector = solve_system_reduction_method(stiffness_matrix, force_vector, fixed_dofs)
    else:
        raise ValueError(f"Unsupported method: '{method}'. Please use 'Penalty Method' or 'Reduction Method'.")

    return displacement_vector, force_vector

def solve_system_penalty_method(stiffness_matrix: np.array, force_vector: np.array, fixed_dofs: np.array) -> tuple:
    stiffness_matrix_bcs = apply_penalty_boundary_conditions_stiffness(fixed_dofs, stiffness_matrix)
    force_vector_bcs = apply_penalty_boundary_conditions_force(fixed_dofs, force_vector)

    # Solve for the displacements
    displacement_vector = np.linalg.solve(stiffness_matrix_bcs, force_vector_bcs).flatten()

    # Solve for the forces
    force_vector = np.matmul(stiffness_matrix, displacement_vector)

    return displacement_vector, force_vector

def solve_system_reduction_method(stiffness_matrix: np.array, force_vector: np.array, fixed_dofs: np.array) -> tuple:
    free_dofs = [i for i in range(len(force_vector)) if i not in fixed_dofs]

    # Stiffness matrix decompostion
    K_I_I = stiffness_matrix[np.ix_(fixed_dofs, fixed_dofs)]  
    K_I_II = stiffness_matrix[np.ix_(fixed_dofs, free_dofs)]  
    K_II_I = stiffness_matrix[np.ix_(free_dofs, fixed_dofs)]  
    K_II_II = stiffness_matrix[np.ix_(free_dofs, free_dofs)]  
    
    # Force vector decomposition
    F_I = force_vector[fixed_dofs]
    F_II = force_vector[free_dofs]

    # Initialization of the solution
    displacement_vector = np.zeros_like(force_vector)
    force_vector_ = np.zeros_like(force_vector)

    # Solve displacements at free dofs
    U_II = np.linalg.solve(K_II_II, F_II)
    
    # Solve forces at fixed dofs
    F_I_updated = np.matmul(K_I_II, U_II)

    # Reconstruct the displacement vector
    displacement_vector[free_dofs] = U_II  
    displacement_vector[fixed_dofs] = 0  # Fixed dofs remain zero

    # Reconstruct the force vector
    force_vector_[fixed_dofs] = F_I_updated
    force_vector_[free_dofs] = F_II  # Free dofs forces remain unchanged

    return displacement_vector.flatten(), force_vector_.flatten()

def assemble_global_stiffness_matrix(df_elements: pd.DataFrame, df_nodes: pd.DataFrame, df_materials: pd.DataFrame, df_properties:pd.DataFrame) -> np.array:
    num_nodes = df_nodes.shape[0] 
    num_dofs_per_node = FiniteElement.DOF_PER_NODE_MAP.get(df_elements.iloc[0, 0])  
    num_dofs = num_dofs_per_node * num_nodes

    # Initialize the global stiffness matrix
    K_global = np.zeros((num_dofs, num_dofs))

    # Iterate through each element in the mesh
    for _, row in df_elements.iterrows():
        element_type = row['Element Type']
        if element_type == 'CBAR':
            element = CBAR(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CTRIA3':
            element = CTRIA3(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CQUAD4':
            element = CQUAD4(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CTETRA':
            element = CTETRA(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CHEXA':
            element = CHEXA(row, df_nodes, df_properties, df_materials)

        # Get global DOF indices and element stiffness matrix
        global_index_dofs = element.global_index_dofs
        #K_element = element.stiffness_matrix
        K_element = element.assemble_stiffness_matrix()

        # Assemble local stiffness matrix into the global matrix
        for i_local, i_global in enumerate(global_index_dofs):
            for j_local, j_global in enumerate(global_index_dofs):
                K_global[i_global][j_global] += K_element[i_local][j_local]
                
    return K_global

def assemble_global_load_vector(simulation_case: str, df_nodes: pd.DataFrame, df_loads: pd.DataFrame) -> np.array:
    num_nodes = df_nodes.shape[0] 
    num_dofs_per_node = FiniteElement.DOF_PER_NODE_MAP.get(simulation_case)  
    num_dofs = num_dofs_per_node * num_nodes

    # Initialize the global force vector
    F_global = np.zeros((num_dofs, 1))

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

def apply_penalty_boundary_conditions_stiffness(fixed_dofs: np.array, stiffness_matrix: np.array) -> tuple:
    stiffness_matrix_ = stiffness_matrix.copy()

    for fixed_dof in fixed_dofs:
        stiffness_matrix_[fixed_dof, :] = 0
        stiffness_matrix_[:, fixed_dof] = 0
        stiffness_matrix_[fixed_dof, fixed_dof] = 1

    return stiffness_matrix_

def apply_penalty_boundary_conditions_force(fixed_dofs: np.array, force_vector: np.array) -> tuple:
    force_vector_ = force_vector.copy()

    for fixed_dof in fixed_dofs:
        force_vector_[fixed_dof] = 0
    
    return force_vector_

def get_boundary_conditions_dofs(simulation_case: str, df_nodes: pd.DataFrame, df_boundary_conditions: pd.DataFrame) -> np.array:
    num_nodes = df_nodes.shape[0] 
    num_dofs_per_node = FiniteElement.DOF_PER_NODE_MAP.get(simulation_case) 

    fixed_dofs = []
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

    return np.array(fixed_dofs)
