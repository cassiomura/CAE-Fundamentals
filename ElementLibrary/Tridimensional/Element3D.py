# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                            Title: Element3D.py
# ==============================================================================
import numpy as np
from ElementLibrary.FiniteElement import FiniteElement

class Element3D(FiniteElement):

    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)

        # Read the property data frame information:
        self.element_material_id = int(df_properties.set_index('Property ID').at[self.element_property_id, 'Material ID'])
  
        # Read the material data frame information:
        self.E = float(df_materials.set_index('Material ID').at[self.element_material_id, 'E'])
        self.nu = float(df_materials.set_index('Material ID').at[self.element_material_id, 'nu'])

        # Assemble the stiffness matrix:
        self.stiffness_matrix = self.assemble_stiffness_matrix()

    def assemble_jacobian_matrix(self, r: float, s: float, t: float) -> np.array:
        # Compute natural derivatives:
        dN_dr, dN_ds, dN_dt = self.compute_shape_function_derivatives(r, s, t)

        # Compute the Jacobian matrix using matrix multiplication
        JacobianMatrix = np.array([
            np.dot(dN_dr, self.node_coordinates),  # [J11, J12, J13]
            np.dot(dN_ds, self.node_coordinates),  # [J21, J22, J23]
            np.dot(dN_dt, self.node_coordinates)   # [J31, J32, J33]
        ])

        # Compute the determinant of the Jacobian matrix:
        JacobianDeterminant = np.linalg.det(JacobianMatrix)

        return JacobianMatrix, JacobianDeterminant

    def assemble_B_matrix(self, r: float, s: float, t: float) -> np.array:
        # [dN] - Shape Functions Natural Derivatives dr, ds:
        dN_dr, dN_ds, dN_dt  = self.compute_shape_function_derivatives(r, s, t)

        # [P] - Transformation matrix: Shape Functions Natural Derivatives - Displacement:
        P = self.assemble_P_matrix(dN_dr, dN_ds, dN_dt)

        # [J] - Jacobian Matrix:
        Jmatrix, _ = self.assemble_jacobian_matrix(r, s, t)

        # [G] - Transformation matrix: Strain - Shape Functions Natural Derivatives:
        G = self.assemble_G_matrix(Jmatrix)

        # [B] - Strain-Displacement matrix:
        B = np.matmul(G, P)

        return B

    def assemble_D_matrix(self) -> np.array:
        # Construction of the Stress-Strain matrix [D]:
        D_matrix = self.E*(1 - self.nu)/((1+self.nu)*(1 - 2*self.nu))*np.array([[1, self.nu/(1 - self.nu),  self.nu/(1 - self.nu), 0, 0, 0],
                                                                                [self.nu/(1 - self.nu), 1,  self.nu/(1 - self.nu), 0, 0, 0],
                                                                                [self.nu/(1 - self.nu),  self.nu/(1 - self.nu), 1, 0, 0, 0],
                                                                                [0, 0, 0, (1 - 2*self.nu)/(2*(1 - self.nu)), 0, 0],
                                                                                [0, 0, 0, 0, (1 - 2*self.nu)/(2*(1 - self.nu)), 0],
                                                                                [0, 0, 0, 0, 0, (1 - 2*self.nu)/(2*(1 - self.nu))]])
    
        return D_matrix

    def assemble_G_matrix(self, jacobian_matrix: np.array) -> np.array:
        
        # Calculate the inverse of the Jacobian matrix inv([J])
        inverse_jacobian_matrix = np.linalg.inv(jacobian_matrix)

        # Create a zeros row
        zeros_1x3 = np.zeros_like(inverse_jacobian_matrix[0])

        # Construct the G_matrix using block matrix construction
        G_matrix = np.block([[inverse_jacobian_matrix[0], zeros_1x3, zeros_1x3],
                             [zeros_1x3, inverse_jacobian_matrix[1], zeros_1x3],
                             [zeros_1x3, zeros_1x3, inverse_jacobian_matrix[2]],
                             [inverse_jacobian_matrix[1], inverse_jacobian_matrix[0], zeros_1x3                 ],
                             [inverse_jacobian_matrix[2], zeros_1x3                 , inverse_jacobian_matrix[0]],
                             [zeros_1x3                 , inverse_jacobian_matrix[2], inverse_jacobian_matrix[1]]])
        
        return G_matrix
 
    def assemble_P_matrix(self, dN_dr: np.array, dN_ds: np.array, dN_dt: np.array) -> np.array:
        # Block matrix of natural derivatives:
        derivatives_block = np.array([dN_dr.T, dN_ds.T, dN_dt.T])

        # Create a zero block with the same shape as derivative_block:
        zero_block = np.zeros_like(derivatives_block)

        # Assemble the P matrix by stacking and concatenating blocks:
        P_matrix = np.vstack([np.hstack([derivatives_block, zero_block, zero_block]),
                              np.hstack([zero_block, derivatives_block, zero_block]),
                              np.hstack([zero_block, zero_block, derivatives_block])])
    
        return P_matrix