# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                            Title: CQUAD4.py
# ==============================================================================
import numpy as np
from ElementLibrary.Bidimensional.Element2D import Element2D

class CQUAD4(Element2D):

    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)

        self.type = "CQUAD4"

        # Assemble the stiffness matrix:
        self.stiffness_matrix = self.assemble_stiffness_matrix()

    def assemble_stiffness_matrix(self) -> np.array:
        # Stiffness matrix initialization:
        K = np.zeros((8, 8))

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

    def compute_quadrature(self) -> tuple:
        # Quadrature points (r, s coordinates for Gauss points)
        quadrature_points = np.array([[-np.sqrt(3)/3, -np.sqrt(3)/3], 
                                      [np.sqrt(3)/3, -np.sqrt(3)/3],
                                      [np.sqrt(3)/3, np.sqrt(3)/3],
                                      [-np.sqrt(3)/3, np.sqrt(3)/3]])
        quadrature_weights = np.array([1, 1, 1, 1])

        return quadrature_points, quadrature_weights

    def compute_shape_functions(self, r: float, s: float) -> np.array:
        # Calculate shape functions for the QUAD4 element
        N1 = 0.25 * (1 - r) * (1 - s)
        N2 = 0.25 * (1 + r) * (1 - s)
        N3 = 0.25 * (1 + r) * (1 + s)
        N4 = 0.25 * (1 - r) * (1 + s)
    
        # Return the shape functions as an array
        shape_functions = np.array([N1, N2, N3, N4])

        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float) -> tuple:
        # Calculate the derivatives of the shape functions with respect to r (dN/dr)
        dN_dr_1 = -0.25 * (1 - s)
        dN_dr_2 =  0.25 * (1 - s)
        dN_dr_3 =  0.25 * (1 + s)
        dN_dr_4 = -0.25 * (1 + s)
    
        # Calculate the derivatives of the shape functions with respect to s (dN/ds)
        dN_ds_1 = -0.25 * (1 - r)
        dN_ds_2 = -0.25 * (1 + r)
        dN_ds_3 =  0.25 * (1 + r)
        dN_ds_4 =  0.25 * (1 - r)

        # Return the derivatives as two arrays
        dN_dr = np.array([dN_dr_1, dN_dr_2, dN_dr_3, dN_dr_4])
        dN_ds = np.array([dN_ds_1, dN_ds_2, dN_ds_3, dN_ds_4])

        return dN_dr, dN_ds