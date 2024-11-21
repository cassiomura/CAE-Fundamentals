# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                             Title: CHEXA.py
# ==============================================================================
import numpy as np
from ElementLibrary.Tridimensional.Element3D import Element3D

class CHEXA(Element3D):

    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)

        self.type = "CHEXA"

        # Assemble the stiffness matrix:
        self.stiffness_matrix = self.assemble_stiffness_matrix()
    
    def assemble_stiffness_matrix(self):
        # Stiffness matrix initialization:
        K = np.zeros((24, 24))

        # Read Gauss Quadrature data:
        quadrature_points, quadrature_weights = self.compute_quadrature()

        # [D] - Stress - Strain matrix (Plane Stress):
        D = self.assemble_D_matrix()              
    
        # Looping through Gaussian quadrature points and weights to construct the stiffness matrix:
        for (r, s, t), w in zip(quadrature_points, quadrature_weights):

            # [J] - Jacobian Matrix:
            _, Jdet = self.assemble_jacobian_matrix(r, s, t)

            # [B] - Strain-Displacement matrix:
            B = self.assemble_B_matrix(r, s, t)

            # [K] - Stiffness Matrix:
            K += Jdet*w*np.matmul(B.T, np.matmul(D, B))
        return K

    def compute_quadrature(self):
        # 2 Points Gauss Quadrature:
        quadrature_points = np.sqrt(3)/3*np.array([[-1, -1, -1],
                                                   [+1, -1, -1],
                                                   [+1, +1, -1],
                                                   [-1, +1, -1],
                                                   [-1, -1, +1],
                                                   [+1, -1, +1],
                                                   [+1, +1, +1],
                                                   [-1, +1, +1]])
        
        quadrature_weights = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float, s: float, t: float) -> np.array:
        # Calculate shape functions for the CHEXA element
        shape_functions = 1/8*np.array([(1 - r)*(1 - s)*(1 - t),  #N1
                                        (1 + r)*(1 - s)*(1 - t),  #N2
                                        (1 + r)*(1 + s)*(1 - t),  #N3
                                        (1 - r)*(1 + s)*(1 - t),  #N4
                                        (1 - r)*(1 - s)*(1 + t),  #N5
                                        (1 + r)*(1 - s)*(1 + t),  #N6
                                        (1 + r)*(1 + s)*(1 + t),  #N7
                                        (1 - r)*(1 + s)*(1 + t)]) #N8
        
        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float, t: float) -> np.array:
        
        # Calculate shape functions derivatives for the CHEXA element
        dN_dr = 1/8*np.array([- (1 - s) * (1 - t),  #dN1_dr
                              + (1 - s) * (1 - t),  #dN2_dr
                              + (1 + s) * (1 - t),  #dN3_dr
                              - (1 + s) * (1 - t),  #dN4_dr
                              - (1 - s) * (1 + t),  #dN5_dr
                              + (1 - s) * (1 + t),  #dN6_dr
                              + (1 + s) * (1 + t),  #dN7_dr
                              - (1 + s) * (1 + t)]) #dN8_dr
    
        dN_ds = 1/8*np.array([- (1 - r) * (1 - t),  #dN1_ds
                              - (1 + r) * (1 - t),  #dN2_ds
                              + (1 + r) * (1 - t),  #dN3_ds
                              + (1 - r) * (1 - t),  #dN4_ds
                              - (1 - r) * (1 + t),  #dN5_ds
                              - (1 + r) * (1 + t),  #dN6_ds
                              + (1 + r) * (1 + t),  #dN7_ds
                              + (1 - r) * (1 + t)]) #dN8_ds
    
        dN_dt = 1/8*np.array([- (1 - r) * (1 - s),  #dN1_dt
                              - (1 + r) * (1 - s),  #dN2_dt
                              - (1 + r) * (1 + s),  #dN3_dt
                              - (1 - r) * (1 + s),  #dN4_dt
                              + (1 - r) * (1 - s),  #dN5_dt
                              + (1 + r) * (1 - s),  #dN6_dt
                              + (1 + r) * (1 + s),  #dN7_dt
                              + (1 - r) * (1 + s)]) #dN8_dt

        return dN_dr, dN_ds, dN_dt