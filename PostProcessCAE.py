# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: PostProcessCAE.py
# ==============================================================================
import pandas as pd
import numpy as np
from ElementLibrary.Unidimensional.CBAR import CBAR
from ElementLibrary.Bidimensional.CTRIA3 import CTRIA3
from ElementLibrary.Bidimensional.CQUAD4 import CQUAD4
from ElementLibrary.Tridimensional.CTETRA import CTETRA
from ElementLibrary.Tridimensional.CHEXA import CHEXA
from ElementLibrary.FiniteElement import FiniteElement

def get_post_processing_output(displacements: np.array, forces: np.array, df_elements: pd.DataFrame, 
                               df_nodes: pd.DataFrame, df_materials: pd.DataFrame, df_properties: pd.DataFrame) -> tuple:
    
    # Create a DataFrame for node displacements
    df_displacement = create_displacement_dataframe(displacements, df_nodes, df_elements)

    # Create a DataFrame for node forces
    df_forces = create_forces_dataframe(forces, df_nodes, df_elements)

    # Create a DataFrame for element strain
    df_strain = create_strain_dataframe(displacements, df_elements, df_nodes, df_materials, df_properties)

    # Create a DataFrame for element stress
    df_stress = create_stress_dataframe(displacements, df_elements, df_nodes, df_materials, df_properties)

    return df_displacement, df_forces, df_stress, df_strain

def create_displacement_dataframe(displacements: np.array, df_nodes: pd.DataFrame, df_elements: pd.DataFrame) -> pd.DataFrame:
    # Number of nodes in the mesh
    num_nodes = df_nodes.shape[0]
    
    # Element type of the first element in the df_elements DataFrame
    element_type = df_elements.iloc[0, 0]
    
    # Case 1: For 2D elements (CTRIA3, CQUAD4) and 1D elements (CROD), which have only ux, uy displacements
    if element_type in ['CTRIA3', 'CQUAD4', 'CROD']:
        # Extract x and y displacements from the solution displacement array
        ux_displacements = displacements[:num_nodes]               # x displacements
        uy_displacements = displacements[num_nodes:2*num_nodes]    # y displacements

        # Create a DataFrame for displacements (ux, uy)
        df_displacements = pd.DataFrame({
            'ux': ux_displacements,
            'uy': uy_displacements,
        })
    
    # Case 2: For 3D elements (CTETRA, CHEXA), which have ux, uy, uz displacements
    elif element_type == 'CTETRA' or element_type == 'CHEXA':
        # Extract x and y displacements from the solution displacement array
        ux_displacements = displacements[:num_nodes]               # x displacements
        uy_displacements = displacements[num_nodes:2*num_nodes]    # y displacements
        uz_displacements = displacements[2*num_nodes:3*num_nodes]  # z displacements

        # Create a DataFrame for displacements (ux, uy)
        df_displacements = pd.DataFrame({
            'ux': ux_displacements,
            'uy': uy_displacements,
            'uz': uz_displacements,
        })
    
    # Case 3: For beam elements (CBAR), which have translations and rotations (ux, uy, uz, thetax, thetay, thetaz)
    elif element_type == 'CBAR':
        # Extract x and y displacements from the solution displacement array
        ux_displacements = displacements[:num_nodes]                     # x displacements
        uy_displacements = displacements[num_nodes:2*num_nodes]          # y displacements
        uz_displacements = displacements[2*num_nodes:3*num_nodes]        # z displacements
        thetax_displacements = displacements[3*num_nodes:4*num_nodes]    # theta x rotation
        thetay_displacements = displacements[4*num_nodes:5*num_nodes]    # theta y rotation
        thetaz_displacements = displacements[5*num_nodes:6*num_nodes]    # theta z rotation

        # Create a DataFrame for displacements (ux, uy, uz, thetax, thetay, thetaz)
        df_displacements = pd.DataFrame({
            'ux': ux_displacements,
            'uy': uy_displacements,
            'uz': uz_displacements,
            'thetax': thetax_displacements,
            'thetay': thetay_displacements,
            'thetaz': thetaz_displacements
        })
    
    # Combine the Node ID from df_nodes with the displacement DataFrame
    df_node_displacements = pd.concat([df_nodes['Node ID'], df_displacements], axis=1)

    return df_node_displacements

def create_forces_dataframe(forces: np.array, df_nodes: pd.DataFrame, df_elements: pd.DataFrame) -> pd.DataFrame:
    # Number of nodes in the mesh
    num_nodes = df_nodes.shape[0]
    
    # Element type of the first element in the df_elements DataFrame
    element_type = df_elements.iloc[0, 0]
    
    # Case 1: For 2D elements (CTRIA3, CQUAD4) and 1D elements (CROD), which have only fx, fy forces
    if element_type in ['CTRIA3', 'CQUAD4', 'CROD']:
        # Extract x and y forces from the solution force array
        force_x = forces[:num_nodes]               # x force
        force_y = forces[num_nodes:2*num_nodes]    # y force

        # Create a DataFrame for forces (fx, fy)
        df_forces = pd.DataFrame({
            'fx': force_x,
            'fy': force_y,
        })
    
    # Case 2: For 3D elements (CTETRA, CHEXA), which have fx, fy, fz forces
    elif element_type in ['CTETRA', 'CHEXA']:
        # Extract x and y displacements from the solution displacement array
        force_x = forces[:num_nodes]               # x force
        force_y = forces[num_nodes:2*num_nodes]    # y force
        force_z = forces[2*num_nodes:3*num_nodes]  # z force

        # Create a DataFrame for forces (fx, fy, fz)
        df_forces = pd.DataFrame({
            'fx': force_x,
            'fy': force_y,
            'fz': force_z,
        })
    
    # Case 3: For beam elements (CBAR), which have forces and moments (fx, fy, fz, Mx, My, Mz)
    elif element_type == 'CBAR':
        # Extract x and y displacements from the solution displacement array
        force_x = forces[:num_nodes]                     # x displacements
        force_y = forces[num_nodes:2*num_nodes]          # y displacements
        force_z = forces[2*num_nodes:3*num_nodes]        # z displacements
        moment_x = forces[3*num_nodes:4*num_nodes]    # theta x distorcion
        moment_y = forces[4*num_nodes:5*num_nodes]    # theta y distorcion
        moment_z = forces[5*num_nodes:6*num_nodes]    # theta z distorcion

        # Create a DataFrame for displacements (ux, uy, uz, thetax, thetay, thetaz)
        df_forces = pd.DataFrame({
            'ux': force_x,
            'uy': force_y,
            'uz': force_z,
            'thetax': moment_x,
            'thetay': moment_y,
            'thetaz': moment_z
        })
    
    # Combine the Node ID from df_nodes with the displacement DataFrame
    df_node_forces = pd.concat([df_nodes['Node ID'], df_forces], axis=1)

    return df_node_forces

def create_stress_dataframe(displacement:np.array, df_elements: pd.DataFrame, df_nodes: pd.DataFrame, df_materials: pd.DataFrame, df_properties:pd.DataFrame) -> pd.DataFrame:
    # Initialize stress data storage
    stresses = {'stress_x': [], 'stress_y': [], 'stress_z': [], 'stress_xy': [], 'stress_xz': [], 'stress_yz': [], 'von_mises': []}
    
    # Iterate through each element in the mesh
    for _, row in df_elements.iterrows():
        # Create element object based on its type
        element_type = row['Element Type']
        if element_type == 'CTRIA3':
            element = CTRIA3(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CQUAD4':
            element = CQUAD4(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CTETRA':
            element = CTETRA(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CHEXA':
            element = CHEXA(row, df_nodes, df_properties, df_materials)
        else:  
            return pd.DataFrame() # Return empty DataFrame if element type is unsupported
        
        # Define a reference point within the element to evaluate stress
        POINT = 0, 0, 0

        # Calculate the strain for the element based on nodal displacements
        element_strain = compute_element_strain(element, displacement, df_nodes, POINT)
        # Calculate the stress based on the computed strain and material properties
        element_stress, stress_tensor = compute_element_stress(element, element_strain)
        
        # 2D case:
        if element_stress.size == 3:
             # Append calculated stress components to the corresponding lists
            stresses['stress_x'].append(element_stress[0])
            stresses['stress_y'].append(element_stress[1])
            stresses['stress_xy'].append(element_stress[2])
            stresses['von_mises'].append(compute_von_mises_stress(stress_tensor))
            stresses['stress_z'].append(None)
            stresses['stress_xz'].append(None)
            stresses['stress_yz'].append(None)

        # 3D case:
        elif element_stress.size == 6:
            # Append calculated strain components to the corresponding lists
            stresses['stress_x'].append(element_stress[0])
            stresses['stress_y'].append(element_stress[1])
            stresses['stress_z'].append(element_stress[2])
            stresses['stress_xy'].append(element_stress[3])
            stresses['stress_xz'].append(element_stress[4])
            stresses['stress_yz'].append(element_stress[5])
            stresses['von_mises'].append(compute_von_mises_stress(stress_tensor))

    # Create a DataFrame with the stress results
    df_element_stress = pd.DataFrame(stresses)
    #df_element_stress['Element ID'] = df_elements['Element ID']
    df_element_stress.insert(0, 'Element ID', df_elements['Element ID'])
    
    # Drop columns that contain only None values
    df_element_stress.dropna(axis=1, how='all', inplace=True)

    return df_element_stress

def create_strain_dataframe(displacement:np.array, df_elements: pd.DataFrame, df_nodes: pd.DataFrame, df_materials: pd.DataFrame, df_properties: pd.DataFrame) -> pd.DataFrame:
    # Initialize strain data storage
    strains = {'strain_x': [], 'strain_y': [], 'strain_z': [], 'strain_xy': [], 'strain_xz': [], 'strain_yz': []}

    # Iterate through each element in the mesh
    for _, row in df_elements.iterrows():
        # Create element object based on its type
        element_type = row['Element Type']
        if element_type == 'CTRIA3':
            element = CTRIA3(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CQUAD4':
            element = CQUAD4(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CTETRA':
            element = CTETRA(row, df_nodes, df_properties, df_materials)
        elif element_type == 'CHEXA':
            element = CHEXA(row, df_nodes, df_properties, df_materials)
        else:  
            return pd.DataFrame()
        
        # Define a reference point within the element to evaluate strain
        POINT = 0, 0, 0

        # Calculate the strain for the element based on nodal displacements
        element_strain = compute_element_strain(element, displacement, df_nodes, POINT)

        # 2D case:
        if element_strain.size == 3:
            # Append calculated strain components to the corresponding lists
            strains['strain_x'].append(element_strain[0])
            strains['strain_y'].append(element_strain[1])
            strains['strain_xy'].append(element_strain[2])
            strains['strain_z'].append(None)
            strains['strain_xz'].append(None)
            strains['strain_yz'].append(None)

        # 3D case:
        elif element_strain.size == 6:
            # Append calculated strain components to the corresponding lists
            strains['strain_x'].append(element_strain[0])
            strains['strain_y'].append(element_strain[1])
            strains['strain_z'].append(element_strain[2])
            strains['strain_xy'].append(element_strain[3])
            strains['strain_xz'].append(element_strain[4])
            strains['strain_yz'].append(element_strain[5])

    # Create a DataFrame with the strain results
    df_element_strain = pd.DataFrame(strains)
    #df_element_strain['Element ID'] = df_elements['Element ID']
    df_element_strain.insert(0, 'Element ID', df_elements['Element ID'])
    # Drop columns that contain only None values
    df_element_strain.dropna(axis=1, how='all', inplace=True)

    return df_element_strain

def compute_element_stress(element: FiniteElement, element_strain: np.array) -> tuple:
    # Assemble the stress-strain matrix [D]
    if element.type in ['CTRIA3', 'CQUAD4']:
        D = element.assemble_D_matrix("plane_stress")
    elif element.type in ['CTETRA', 'CHEXA']:
        D = element.assemble_D_matrix()

    # Compute the element stresses by performing the matrix multiplication of D and strain
    element_stress = np.matmul(D, element_strain)

    # Case for 2D stress components
    if element_stress.size == 3:
        sigma_x, sigma_y, tau_xy = element_stress

        # Construct the 2D stress tensor for plane stress (2x2 matrix)
        element_stress_tensor = np.array([[sigma_x, tau_xy],
                                          [tau_xy, sigma_y]])

    # Case for 3D stress components
    elif element_stress.size == 6:
        sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz = element_stress

        # Construct the 3D stress tensor for solid elements (3x3 matrix)
        element_stress_tensor = np.array([[sigma_x, tau_xy,  tau_xz],
                                          [tau_xy,  sigma_y, tau_yz],
                                          [tau_xz,  tau_yz,  sigma_z]])
        
    return element_stress, element_stress_tensor    

def compute_element_strain(element: FiniteElement, displacement: np.array, df_nodes: pd.DataFrame, point: tuple) -> np.array:
    # Retrieve the global degrees of freedom (DOFs) for the element based on the node data
    element_global_dofs = element.get_global_index_dofs(df_nodes)
    # Extract the displacement vector [U] corresponding to the element's global DOFs
    element_displacement = displacement[element_global_dofs]

    # Extract the coordinates (r, s, t) corresponding to the evaluation point within the element natural coordinate
    r, s, t = point

    # Depending on the element type, assemble the strain-displacement matrix [B]
    if element.type in ['CTRIA3', 'CQUAD4']:
        B = element.assemble_B_matrix(r, s)
    elif element.type in ['CTETRA', 'CHEXA']:
        B = element.assemble_B_matrix(r, s, t)

    # Compute the element strain by multiplying the B matrix with the element displacement vector [U]
    element_strain = np.matmul(B, element_displacement)

    return element_strain

def compute_von_mises_stress(stress_tensor: np.array) -> float:
    # Calculate the principal stresses:
    principal_stresses, principal_directions = np.linalg.eig(stress_tensor)

    # Case for 2D stress components
    if principal_stresses.size == 2:
        von_mises_stress = np.sqrt((np.power(principal_stresses[0], 2) + 
                                    np.power(principal_stresses[1], 2) - 
                                    principal_stresses[0]*principal_stresses[1]))
    
    # Case for 3D stress components
    elif principal_stresses.size == 3:
        von_mises_stress = np.sqrt(0.5*(np.power(principal_stresses[0] - principal_stresses[1], 2) + 
                                        np.power(principal_stresses[1] - principal_stresses[2], 2) + 
                                        np.power(principal_stresses[2] - principal_stresses[0], 2)))
    
    return von_mises_stress