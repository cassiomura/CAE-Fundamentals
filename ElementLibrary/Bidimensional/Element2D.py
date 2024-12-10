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
        # Stiffness matrix initialization:
        K = np.zeros((self.num_dofs, self.num_dofs))

        # Read Gauss Quadrature data:
        quadrature_points, quadrature_weights = self.compute_quadrature()

        # [D] - Stress - Strain matrix (Plane Stress):
        D = self.assemble_D_matrix("plane_stress")              
    
        # Looping through Gaussian quadrature points and weights to construct the stiffness matrix:
        for (r, s), w in zip(quadrature_points, quadrature_weights):

            # [J] - Jacobian Matrix:
            _, Jdet = self.assemble_jacobian_matrix(r, s)

            # [B] - Strain-Displacement matrix:
            B = self.assemble_B_matrix(r, s)

            # [K] - Stiffness Matrix:
            K += self.thickness*Jdet*w*np.matmul(B.T, np.matmul(D, B))
        return K
    
    def assemble_jacobian_matrix(self, r: float, s: float) -> np.array:
        # Coordinates of the nodes:
        x = np.array([coord[0] for coord in self.node_coordinates])
        y = np.array([coord[1] for coord in self.node_coordinates])

        # Compute natural derivatives:
        dN_dr, dN_ds = self.compute_shape_function_derivatives(r, s)

        # Jacobian matrix calculation (dot product of shape function derivatives and coordinates):
        J11 = np.dot(dN_dr, x)  # dx/dr
        J12 = np.dot(dN_dr, y)  # dy/dr
        J21 = np.dot(dN_ds, x)  # dx/ds
        J22 = np.dot(dN_ds, y)  # dy/ds

        # Form the Jacobian matrix:
        JacobianMatrix = np.array([[J11, J12], [J21, J22]])

        # Compute the determinant of the Jacobian matrix:
        JacobianDeterminant = np.linalg.det(JacobianMatrix)

        return JacobianMatrix, JacobianDeterminant

    def assemble_B_matrix(self, r: float, s: float) -> np.array:
        # [dN] - Shape Functions Natural Derivatives dr, ds:
        dN_dr, dN_ds  = self.compute_shape_function_derivatives(r, s)

        # [P] - Transformation matrix: Shape Functions Natural Derivatives - Displacement:
        P = self.assemble_P_matrix(dN_dr, dN_ds)

        # [J] - Jacobian Matrix:
        Jmatrix, Jdet = self.assemble_jacobian_matrix(r, s)

        # [G] - Transformation matrix: Strain - Shape Functions Natural Derivatives:
        G = self.assemble_G_matrix(Jmatrix, Jdet)

        # [B] - Strain-Displacement matrix:
        B = np.matmul(G, P)

        return B

    def assemble_D_matrix(self, material_type: str) -> np.array:
        # Construction of the Stress-Strain matrix [D]:
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

    def assemble_G_matrix(self, jacobian_matrix: np.array, jacobian_determinant: float) -> np.array:
        # Extract individual elements of the Jacobian matrix:s
        J11, J12 = jacobian_matrix[0, 0], jacobian_matrix[0, 1]
        J21, J22 = jacobian_matrix[1, 0], jacobian_matrix[1, 1]
    
        # Create the G transformation matrix
        G_transformation_matrix = (1/jacobian_determinant) * np.array([
            [J22, -J12,  0,    0   ],
            [  0,   0,  -J21, J11  ],
            [-J21, J11,  J22, -J12 ]])
    
        return G_transformation_matrix
 
    def assemble_P_matrix(self, dN_dr: np.array, dN_ds: np.array) -> np.array:
        # Block matrix of natural derivatives:
        derivatives_block = np.array([dN_dr.T, dN_ds.T])

        # Create a zero block with the same shape as derivative_block:
        zero_block = np.zeros_like(derivatives_block)

        # Assemble the P matrix by stacking and concatenating blocks:
        P_matrix = np.vstack([np.hstack([derivatives_block, zero_block]),
                              np.hstack([zero_block, derivatives_block])])
    
        return P_matrix