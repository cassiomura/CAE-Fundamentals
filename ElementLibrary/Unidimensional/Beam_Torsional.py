# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: Beam_Torsional.py
# ==============================================================================
import numpy as np

class Beam_Torsional:
     
    def __init__(self, G_, J_, L_):
        self.G = G_
        self.J = J_
        self.L = L_

    def quadrature(self):
        quadrature_points = np.array([-np.sqrt(3)/3, np.sqrt(3)/3])
        quadrature_weights = np.array([1, 1])

        return quadrature_points, quadrature_weights

    def assemble_axial_B_matrix(self, x: float) -> np.array:
        B_matrix = np.array([- 1/self.L, 1/self.L])

        if B_matrix.ndim == 1:
            B_matrix = B_matrix.reshape(1, -1)  # Reshape to 1xN (or B.T to Nx1 if needed)
        return B_matrix

    def assemble_axial_D_matrix(self) -> float:

        D_matrix = self.G*self.J
        return D_matrix

    def assemble_torsional_stiffness_matrix(self):
        # Stiffness matrix calculation:
        quadrature_points, quadrature_weights = self.quadrature()

        D = self.assemble_axial_D_matrix()

        K = np.zeros((2, 2))
        for quadrature_point, quadrature_weight in zip(quadrature_points, quadrature_weights):
            # Change of variables (x = L/2 + L/2*epsilon):
            quadrature_point_mapped = 0.5*self.L + 0.5*self.L*quadrature_point

            B = self.assemble_axial_B_matrix(quadrature_point_mapped)

            K += self.L/2*quadrature_weight*np.matmul(B.T, D*B)
        return K