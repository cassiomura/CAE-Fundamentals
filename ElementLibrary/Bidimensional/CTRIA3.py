# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                            Title: CTRIA3.py
# ==============================================================================
import numpy as np
from ElementLibrary.Bidimensional.Element2D import Element2D

class CTRIA3(Element2D):

    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)
        
        self.type = "CTRIA3"

        # Assemble the stiffness matrix:
        self.stiffness_matrix = self.assemble_stiffness_matrix()
    
    def assemble_stiffness_matrix(self) -> np.array:
        # Stiffness matrix initialization:
        K = np.zeros((6, 6))

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

    def compute_quadrature(self):
        # 3 Points Gauss Quadrature:
        quadrature_points = np.array([[0, 1/2], 
                                      [1/2, 0],
                                      [1/2, 1/2]])
        
        quadrature_weights = np.array([1/6, 1/6, 1/6])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float, s: float) -> np.array:
        # Calculate shape functions for the QUAD4 element
        N1 = 1 - r - s
        N2 = r
        N3 = s
    
        # Return the shape functions as an array
        shape_functions = np.array([N1, N2, N3])

        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float) -> np.array:
        # Calculate the derivatives of the shape functions with respect to r (dN/dr)
        dN_dr_1 = -1
        dN_dr_2 =  1
        dN_dr_3 =  0
    
        # Calculate the derivatives of the shape functions with respect to s (dN/ds)
        dN_ds_1 = -1
        dN_ds_2 =  0
        dN_ds_3 =  1

        # Return the derivatives as two arrays
        dN_dr = np.array([dN_dr_1, dN_dr_2, dN_dr_3])
        dN_ds = np.array([dN_ds_1, dN_ds_2, dN_ds_3])

        return dN_dr, dN_ds
