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

    def compute_quadrature(self):
        quadrature_points = np.array([[0.1381966011250105,0.1381966011250105,0.1381966011250105], 
                                      [0.5854101966249685,0.1381966011250105,0.1381966011250105],
                                      [0.1381966011250105,0.5854101966249685,0.1381966011250105],
                                      [0.1381966011250105,0.1381966011250105,0.5854101966249685]])
        
        quadrature_weights = 1/6*np.array([0.25, 0.25, 0.25, 0.25])

        return quadrature_points, quadrature_weights

    def compute_shape_function(self, r: float, s: float, t: float) -> np.array:
        shape_functions = np.array([1 - r - s - t,  #N1
                                                r,  #N2
                                                s,  #N3
                                                t]) #N4

        return shape_functions

    def compute_shape_function_derivatives(self, r: float, s: float, t: float) -> np.array:
        dN_dr = np.array([- 1,  #dN1_dr
                          + 1,  #dN2_dr
                            0,  #dN3_dr
                            0]) #dN4_dr
        
        dN_ds = np.array([- 1,  #dN1_ds
                            0,  #dN2_ds
                          + 1,  #dN3_ds
                            0]) #dN4_ds
        
        dN_dt = np.array([- 1,  #dN1_dt
                            0,  #dN2_dt
                            0,  #dN3_dt
                          + 1]) #dN4_dt
        
        return dN_dr, dN_ds, dN_dt