# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                            Title: Element2D.py
# ==============================================================================
import numpy as np
from ElementLibrary.FiniteElement import FiniteElement

class Element2D(FiniteElement):

    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)

        # Read the property data frame information:
        self.element_material_id = int(df_properties.set_index('Property ID').at[self.element_property_id, 'Material ID'])
        self.thickness = float(df_properties.set_index('Property ID').at[self.element_property_id, 'Thickness'])
  
        # Read the material data frame information:
        self.E = float(df_materials.set_index('Material ID').at[self.element_material_id, 'E'])
        self.nu = float(df_materials.set_index('Material ID').at[self.element_material_id, 'nu'])

        # Assemble the stiffness matrix:
        self.stiffness_matrix = self.assemble_stiffness_matrix()

    def assemble_stiffness_matrix(self) -> np.array:
        stiffness_matrix = np.zeros((self.num_dofs, self.num_dofs))

        # Read Gauss Quadrature data:
        quadrature_points, quadrature_weights = self.compute_quadrature()

        # [D] - Stress - Strain matrix (Plane Stress):
        D_matrix = self.assemble_D_matrix("plane_stress")              
    
        # Looping through Gaussian quadrature points and weights to construct the stiffness matrix:
        for (r, s), w in zip(quadrature_points, quadrature_weights):

            # [J] - Jacobian Matrix:
            _, Jdet = self.assemble_jacobian_matrix(r, s)

            # [B] - Strain-Displacement matrix:
            B_matrix = self.assemble_B_matrix(r, s)

            # [K] - Stiffness Matrix:
            stiffness_matrix += self.thickness*Jdet*w*np.matmul(B_matrix.T, np.matmul(D_matrix, B_matrix))

        return stiffness_matrix
    
    def assemble_jacobian_matrix(self, r: float, s: float) -> tuple:
        # Remove the z coordinate of the node_coordinates array.
        node_coordinates_2D = self.node_coordinates[:, :2]

        dN_dr, dN_ds = self.compute_shape_function_derivatives(r, s)

        jacobian_matrix = np.array([
            np.dot(dN_dr, node_coordinates_2D),  # [J11, J12]
            np.dot(dN_ds, node_coordinates_2D),  # [J21, J22]
        ])

        jacobian_determinant = np.linalg.det(jacobian_matrix)

        return jacobian_matrix, jacobian_determinant

    def assemble_B_matrix(self, r: float, s: float) -> np.array:
        # [dN] - Shape Functions Natural Derivatives dr, ds:
        dN_dr, dN_ds  = self.compute_shape_function_derivatives(r, s)

        # [P] - Transformation matrix: Shape Functions Natural Derivatives - Displacement:
        P_matrix = self.assemble_P_matrix(dN_dr, dN_ds)

        # [J] - Jacobian Matrix:
        jacobian_matrix, _ = self.assemble_jacobian_matrix(r, s)

        # [G] - Transformation matrix: Strain - Shape Functions Natural Derivatives:
        G_matrix = self.assemble_G_matrix(jacobian_matrix)

        # [B] - Strain-Displacement matrix:
        B_matrix = np.matmul(G_matrix, P_matrix)

        return B_matrix

    def assemble_D_matrix(self, material_type: str) -> np.array:
        if material_type == "plane_stress":
            D_matrix = self.E/(1 - np.power(self.nu ,2))*np.array([[1, self.nu,  0], 
                                                                   [self.nu, 1 , 0],
                                                                   [0, 0, (1 - self.nu)/2]])
        elif material_type == "plane_strain":
            D_matrix = self.E/((1+self.nu)*(1-2*self.nu))*np.array([[1 - self.nu, self.nu,  0], 
                                                                    [self.nu, 1 - self.nu , 0],
                                                                    [0, 0, (1 - 2*self.nu)/2]])
        else:
            raise ValueError("Invalid type: material_type. Select 'plane_strain' or 'plane_stress' for matrix [D].")
    
        return D_matrix

    def assemble_G_matrix(self, jacobian_matrix: np.array) -> np.array:
        inverse_jacobian_matrix = np.linalg.inv(jacobian_matrix)

        zeros_block_1x2 = np.zeros_like(inverse_jacobian_matrix[0])
        
        G_matrix = np.block([[inverse_jacobian_matrix[0], zeros_block_1x2],
                             [zeros_block_1x2, inverse_jacobian_matrix[1]],
                             [inverse_jacobian_matrix[1], inverse_jacobian_matrix[0]]])
        
        return G_matrix
 
    def assemble_P_matrix(self, dN_dr: np.array, dN_ds: np.array) -> np.array:
        derivatives_block = np.array([dN_dr.T, dN_ds.T])

        zero_block = np.zeros_like(derivatives_block)

        P_matrix = np.vstack([np.hstack([derivatives_block, zero_block]),
                              np.hstack([zero_block, derivatives_block])])
    
        return P_matrix