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

    def compute_quadrature(self) -> tuple:
        quadrature_points = np.array([[-np.sqrt(3)/3, -np.sqrt(3)/3], 
                                      [np.sqrt(3)/3, -np.sqrt(3)/3],
                                      [np.sqrt(3)/3, np.sqrt(3)/3],
                                      [-np.sqrt(3)/3, np.sqrt(3)/3]])
        quadrature_weights = np.array([1, 1, 1, 1])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float, s: float) -> np.array:
        shape_functions = 0.25 * np.array([(1 - r) * (1 - s),  #N1
                                           (1 + r) * (1 - s),  #N2
                                           (1 + r) * (1 + s),  #N3
                                           (1 - r) * (1 + s)]) #N4
        
        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float) -> tuple:
        dN_dr = 0.25 * np.array([- (1 - s),  #dN1_dr
                                 + (1 - s),  #dN2_dr
                                 + (1 + s),  #dN3_dr
                                 - (1 + s)]) #dN4_dr
        
        dN_ds = 0.25 * np.array([- (1 - r),  #dN1_ds
                                 - (1 + r),  #dN2_ds
                                 + (1 + r),  #dN3_ds
                                 + (1 - r)]) #dN4_ds

        return dN_dr, dN_ds