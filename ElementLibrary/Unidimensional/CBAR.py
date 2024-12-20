# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                               Title: CBAR.py
# ==============================================================================
import numpy as np
from ElementLibrary.Unidimensional.Beam_Axial import Beam_Axial
from ElementLibrary.Unidimensional.Beam_Flexural import Beam_Flexural
from ElementLibrary.Unidimensional.Beam_Torsional import Beam_Torsional
from ElementLibrary.FiniteElement import FiniteElement

class CBAR(FiniteElement):
    
    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)

        (x1, y1, z1), (x2, y2, z2) = self.node_coordinates[:2]

        # Read the property data frame information:
        self.element_material_id = int(df_properties.set_index('Property ID').at[self.element_property_id, 'Material ID'])

        self.A = float(df_properties.set_index('Property ID').at[self.element_property_id, 'Area'])
        self.Iz = float(df_properties.set_index('Property ID').at[self.element_property_id, 'I11'])
        self.Iy = float(df_properties.set_index('Property ID').at[self.element_property_id, 'I22'])
        self.J = float(df_properties.set_index('Property ID').at[self.element_property_id, 'J'])
  
        # Read the material data frame information:
        self.E = float(df_materials.set_index('Material ID').at[self.element_material_id, 'E'])
        self.nu = float(df_materials.set_index('Material ID').at[self.element_material_id, 'nu'])

        self.G = self.E/(2*(1 + self.nu))

        # Geometric characteristics of the bar:
        l_xy = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Length of the projected xy element
        self.L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)  # Length of the element

        self.alpha = np.arctan2(y2 - y1, x2 - x1)
        self.beta = np.arctan2(z2 - z1, l_xy)
    
    def assemble_stiffness_matrix(self) -> np.array:

        K_local = self.compute_local_stiffness_matrix_cbar()
        T = self.compute_transformation_matrix()

        K_global = np.matmul(T.T, np.matmul(K_local, T))

        # New order of indices based on the new Delta vector
        new_order = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        # Reordering columns
        K_global = K_global[:, new_order]
        # Reordering rows
        K_global = K_global[new_order, :]

        return K_global
    
    def compute_local_stiffness_matrix_cbar(self) -> np.array:
        # Initialize a 12x12 zero matrix to store the complete local stiffness matrix.
        complete_local_stiffness_matrix = np.zeros((12, 12))

        # Axial stiffness matrix and its corresponding DOFs (degrees of freedom).
        axial_stiffness_matrix = Beam_Axial(self.E, self.A, self.L).assemble_stiffness_matrix()
        dofs_axial = np.array([0, 6])
        
        # Flexural stiffness matrix for bending about the local y-axis and its DOFs.
        flexural_y_stiffness_matrix = Beam_Flexural(self.E, self.Iz, self.L).assemble_flexural_stiffness_matrix()
        dofs_flexural_y = np.array([1, 5, 7, 11])

        # Flexural stiffness matrix for bending about the local z-axis and its DOFs.
        flexural_z_stiffness_matrix = Beam_Flexural(self.E, self.Iy, self.L).assemble_flexural_stiffness_matrix()
        dofs_flexural_z = np.array([2, 4, 8, 10])

        # Torsional stiffness matrix and its corresponding DOFs.
        torsional_stiffness_matrix = Beam_Torsional(self.G, self.J, self.L).assemble_stiffness_matrix()
        dofs_torsional = np.array([3, 9])

        # Assemble each component stiffness matrix into the complete matrix.
        self.assemble_local_to_global_matrix(complete_local_stiffness_matrix, axial_stiffness_matrix, dofs_axial)
        self.assemble_local_to_global_matrix(complete_local_stiffness_matrix, flexural_y_stiffness_matrix, dofs_flexural_y)
        self.assemble_local_to_global_matrix(complete_local_stiffness_matrix, flexural_z_stiffness_matrix, dofs_flexural_z)
        self.assemble_local_to_global_matrix(complete_local_stiffness_matrix, torsional_stiffness_matrix, dofs_torsional)

        # Return the assembled complete local stiffness matrix.
        return complete_local_stiffness_matrix

    def assemble_local_to_global_matrix(self, global_matrix: np.array, local_matrix: np.array, dofs:np.array) -> None:
        # Iterate through each combination of local and corresponding global DOFs.
        for index_local_i, index_global_i in enumerate(dofs):
            for index_local_j, index_global_j in enumerate(dofs):
                # Add the local matrix value to the appropriate position in the global matrix.
                global_matrix[index_global_i][index_global_j] += local_matrix[index_local_i][index_local_j]
    
        return None

    def compute_transformation_matrix(self) -> np.array:
        # Rotation matrix around the Z-axis by angle alpha
        T_alpha = np.array([[np.cos(self.alpha), np.sin(self.alpha), 0],
                            [-np.sin(self.alpha), np.cos(self.alpha), 0],
                            [0, 0, 1]])

        # Rotation matrix around the Y-axis by angle beta
        T_beta = np.array([[np.cos(self.beta), 0, np.sin(self.beta)],
                           [0 , 1, 0],
                           [-np.sin(self.beta), 0,np.cos(self.beta)]])

        # Combined transformation matrix: Lambda = T_alpha * T_beta
        Lambda = np.matmul(T_alpha, T_beta)

        # Matrix of zeros with the same shape as Lambda
        zeros_matrix = np.zeros_like(Lambda)

        # Block matrix with Lambda along the diagonal
        T = np.block([
            [Lambda, zeros_matrix, zeros_matrix, zeros_matrix],
            [zeros_matrix, Lambda, zeros_matrix, zeros_matrix],
            [zeros_matrix, zeros_matrix, Lambda, zeros_matrix],
            [zeros_matrix, zeros_matrix, zeros_matrix, Lambda]])

        return T
    