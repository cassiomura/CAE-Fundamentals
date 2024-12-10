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

    def assemble_stiffness_matrix(self) -> np.array:
        # Stiffness matrix initialization:
        K = np.zeros((2, 2))

        # [D] - Stress - Strain matrix:
        D = self.assemble_D_matrix()

        # Read Gauss Quadrature data:
        quadrature_points, quadrature_weights = self.compute_quadrature()

        # Looping through Gaussian quadrature points and weights to construct the stiffness matrix:
        for r, w in zip(quadrature_points, quadrature_weights):

            # [J] - Jacobian Matrix:
            _, Jdet = self.assemble_jacobian_matrix(r)

            # [B] - Strain-Displacement matrix:
            B = self.assemble_B_matrix(r)

            # [K] - Stiffness Matrix:
            K += Jdet*w*np.matmul(B.T, D*B)*self.J
        return K
    
    def compute_quadrature(self) -> tuple:
        quadrature_points = np.array([-np.sqrt(3)/3, np.sqrt(3)/3])
        quadrature_weights = np.array([1, 1])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float) -> np.array:
        shape_functions = 0.5 * np.array([(1 - r),  #N1
                                          (1 + r)]) #N2

        return shape_functions
    
    def compute_shape_function_derivatives(self, r: float) -> np.array:
        dN_dr = 0.5 * np.array([- 1,  #dN1_dr
                                + 1]) #dN2_dr

        return dN_dr
    
    def assemble_jacobian_matrix(self, r: float):
        jacobian_matrix = np.array([self.L/2])

        jacobian_determinant = self.L/2

        return jacobian_matrix,  jacobian_determinant
    
    def assemble_D_matrix(self) -> float:
        D_matrix = self.G

        return D_matrix
    
    def assemble_B_matrix(self, r: float) -> np.array:
        # [dN] - Shape Functions Natural Derivatives dr:
        dN_dr = self.compute_shape_function_derivatives(r)

        # [P] - Transformation matrix: Shape Functions Natural Derivatives - Displacement:
        P_matrix = self.assemble_P_matrix(dN_dr)

        # [J] - Jacobian Matrix:
        J_matrix, _ = self.assemble_jacobian_matrix(r)

        # [G] - Transformation matrix: Strain - Shape Functions Natural Derivatives:
        G_matrix = self.assemble_G_matrix(J_matrix)

        # [B] - Strain-Displacement matrix:
        B_matrix = G_matrix*P_matrix
        B_matrix = B_matrix.reshape(1, -1)  # Reshape to 1xN (or B.T to Nx1 if needed)

        return B_matrix
    
    def assemble_G_matrix(self, jacobian_matrix: np.array) -> np.array:
        # Calculate the inverse of the Jacobian matrix inv([J])
        inverse_jacobian_matrix = 1/jacobian_matrix

        # Construct the G_matrix using block matrix construction
        G_matrix = inverse_jacobian_matrix
        
        return G_matrix
    
    def assemble_P_matrix(self, dN_dr: np.array) -> np.array:
        # Block matrix of natural derivatives:
        derivatives_block = dN_dr

        P_matrix = derivatives_block
    
        return P_matrix

#def assemble_K_analytical(G: float, J: float, L: float):

#    K_matrix = np.array([[J*G/L, -J*G/L],
#                         [-J*G/L, J*G/L]])

#    return K_matrix