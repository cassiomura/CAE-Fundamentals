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
        
    def compute_quadrature(self) -> tuple:
        quadrature_points = np.array([[0, 1/2], 
                                      [1/2, 0],
                                      [1/2, 1/2]])
        
        quadrature_weights = np.array([1/6, 1/6, 1/6])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float, s: float) -> np.array:
        shape_functions = np.array([1 - r - s,  #N1
                                            r,  #N2
                                            s]) #N3

        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float) -> tuple:
        dN_dr = np.array([- 1,  #dN1_dr
                          + 1,  #dN2_dr
                            0]) #dN3_dr
        
        dN_ds = np.array([- 1,  #dN1_ds
                            0,  #dN2_ds
                          + 1]) #dN3_ds

        return dN_dr, dN_ds