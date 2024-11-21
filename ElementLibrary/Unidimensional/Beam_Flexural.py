# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: Beam_Flexural.py
# ==============================================================================
import numpy as np

class Beam_Flexural:
    
    def __init__(self, E_, I_, L_):
        self.L = L_
        self.I = I_
        self.E = E_

    def quadrature(self):

        quadrature_points = np.array([-np.sqrt(3)/3, np.sqrt(3)/3])
        quadrature_weights = np.array([1, 1])

        return quadrature_points, quadrature_weights

    def assemble_A_matrix(self):
        # v(x) = C1 + C2*x + C3*x^2 + C4*x^3
        # v'(x) = C2 + 2x*C3 + 3x^2*C4

        # v1  = v(x = 0)
        # v1' = v'(x = 0)
        # v2  = v(x = L)
        # v2' = v'(x = L)

        # {V} = [A]{C}
        A_matrix = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [1, self.L, np.power(self.L, 2), np.power(self.L, 3)],
                             [0, 1 , 2*self.L, 3*np.power(self.L, 2)]])
        return A_matrix 

    def assemble_flexural_B_matrix(self, x: float) -> np.array:
        A_matrix = self.assemble_A_matrix()
        # Vx = C1 + x*C2 + x^2*C3 + x^3*C4 -> ddVx = 0*C1 + 0*C2 + 2*C3 + 6x*C4
        ddVx = np.array([0, 0, 2, 6*x])

        B_matrix = np.matmul(ddVx, np.linalg.inv(A_matrix))

        if B_matrix.ndim == 1:
            B_matrix = B_matrix.reshape(1, -1)  # Reshape to 1xN (or B.T to Nx1 if needed)
        return B_matrix
    
    def assemble_flexural_D_matrix(self) -> float:

        D_matrix = self.E*self.I

        return D_matrix

    def assemble_flexural_stiffness_matrix(self):
        # Stiffness matrix calculation:
        quadrature_points, quadrature_weights = self.quadrature()

        D = self.assemble_flexural_D_matrix()

        K = np.zeros((4, 4))
        for quadrature_point, quadrature_weight in zip(quadrature_points, quadrature_weights):
            # Change of variables (x = L/2 + L/2*epsilon):
            quadrature_point_mapped = 0.5*self.L + 0.5*self.L*quadrature_point

            B_matrix = self.assemble_flexural_B_matrix(quadrature_point_mapped)

            K += self.L/2*quadrature_weight*np.matmul(B_matrix.T, D*B_matrix)
        return K

def assemble_K_analytical(E: float, I: float, L: float):

    K_matrix = E*I*np.array([[12/np.power(L, 3), 6/np.power(L, 2), -12/np.power(L, 3), 6/np.power(L, 2)],
                     [6/np.power(L, 2), 4/L, -6/np.power(L, 2), 2/L],
                     [-12/np.power(L, 3), -6/np.power(L, 2), 12/np.power(L, 3), -6/np.power(L, 2)],
                     [6/np.power(L, 2), 2/L, -6/np.power(L, 2), 4/L]])

    return K_matrix