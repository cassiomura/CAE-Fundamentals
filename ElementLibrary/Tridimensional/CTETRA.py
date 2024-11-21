# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                             Title: CTETRA.py
# ==============================================================================
import numpy as np
from ElementLibrary.Tridimensional.Element3D import Element3D

class CTETRA(Element3D):

    def __init__(self, row, df_nodes, df_properties, df_materials):
        super().__init__(row, df_nodes, df_properties, df_materials)

        self.type = "CTETRA"

        # Assemble the stiffness matrix:
        self.stiffness_matrix = self.assemble_stiffness_matrix()
    
    def assemble_stiffness_matrix(self):
        # Stiffness matrix initialization:
        K = np.zeros((12, 12))

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
        # 4 Points Gauss Quadrature:
        # https://math.stackexchange.com/questions/1068006/gauss-quadrature-on-tetrahedron
        quadrature_points = np.array([[0.1381966011250105,0.1381966011250105,0.1381966011250105], 
                                      [0.5854101966249685,0.1381966011250105,0.1381966011250105],
                                      [0.1381966011250105,0.5854101966249685,0.1381966011250105],
                                      [0.1381966011250105,0.1381966011250105,0.5854101966249685]])
        
        quadrature_weights = 1/6*np.array([0.25, 0.25, 0.25, 0.25])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float, s: float, t: float) -> np.array:
        # Calculate shape functions for the QUAD4 element
        N1 = 1 - r - s - t
        N2 = r
        N3 = s
        N4 = t
    
        # Return the shape functions as an array
        shape_functions = np.array([N1, N2, N3, N4])

        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float, t: float) -> np.array:
        # Calculate the derivatives of the shape functions with respect to r (dN/dr)
        dN1_dr = -1
        dN2_dr =  1
        dN3_dr =  0
        dN4_dr =  0
    
        # Calculate the derivatives of the shape functions with respect to s (dN/ds)
        dN1_ds = -1
        dN2_ds =  0
        dN3_ds =  1
        dN4_ds =  0

        # Calculate the derivatives of the shape functions with respect to t (dN/dt)
        dN1_dt = -1
        dN2_dt =  0
        dN3_dt =  0
        dN4_dt =  1

        # Return the derivatives as two arrays
        dN_dr = np.array([dN1_dr, dN2_dr, dN3_dr, dN4_dr])
        dN_ds = np.array([dN1_ds, dN2_ds, dN3_ds, dN4_ds])
        dN_dt = np.array([dN1_dt, dN2_dt, dN3_dt, dN4_dt])

        return dN_dr, dN_ds, dN_dt