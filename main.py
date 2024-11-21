# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                              Title: main.py
# ==============================================================================

""" 
                               PROJECT SCOPE
 ==============================================================================
 This script implements a Finite Element Method (FEM) solver with the following
 constraints:

 - Input Format:
     The input data must be provided in the .bdf format, which is the native
     pre-processing file format of Patran.

 - Supported Element Types:
     The solver supports the following elements:
     1. CBAR (Linear Bar Elements)
     2. CTRIA3 (Linear Triangular Elements)
     3. CQUAD4 (Linear Quadrilateral Elements)
     4. CTETRA (Linear Tetrahedral Elements)
     5. CHEXA (Linear Hexahedral Elements)

 - Load Application:
     External loads are restricted to nodal forces only. Distributed forces are
     not supported.

 - Analysis Type:
     This solver is designed for linear static analysis only.

============================================================================== 
"""

import logging
from PreProcessCAE import get_pre_processing_input
from PostProcessCAE import get_post_processing_output
from SolverCAE import solve_system, assemble_global_stiffness_matrix, assemble_global_load_vector, impose_boundary_conditions_penalty_method

# Configure the logging format and level:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Insert the .bdf file name (Ensure that the folder containing this script and the .bdf file is open in your IDE):
INPUT_FILE_NAME = "Cases/CBAR_Benchmark/CBAR_Benchmark.bdf"
OUTPUT_FILE_NAME = "Cases/CBAR_Benchmark/CBAR_displacement_result.csv"

#INPUT_FILE_NAME = "Cases/CTRIA3_Benchmark/CTRIA3_Benchmark.bdf"
#OUTPUT_FILE_NAME = "Cases/CTRIA3_Benchmark/CTRIA3_displacement_result.csv"

#INPUT_FILE_NAME = "Cases/CQUAD4_Benchmark/CQUAD4_Benchmark.bdf"
#OUTPUT_FILE_NAME = "Cases/CQUAD4_Benchmark/CQUAD4_displacement_result.csv"

#INPUT_FILE_NAME = "Cases/CTETRA_Benchmark/CTETRA_Benchmark.bdf"
#OUTPUT_FILE_NAME = "Cases/CTETRA_Benchmark/CTETRA_displacement_result.csv"

#INPUT_FILE_NAME = "Cases/CHEXA_Benchmark/CHEXA_Benchmark.bdf"
#OUTPUT_FILE_NAME = "Cases/CHEXA_Benchmark/CHEXA_displacement_result.csv"

def main(input_file_name: str, output_file_name: str) -> None:
    logging.info(f"(0/6) Reading data ... ")
    df_nodes, df_elements, df_materials, df_properties, df_loads, df_bcs = get_pre_processing_input(input_file_name, print = False)
    
    logging.info(f"(1/6) Assembling the global stiffness matrix ... ")
    K_global = assemble_global_stiffness_matrix(df_elements, df_nodes, df_materials, df_properties)
    
    logging.info(f"(2/6) Assembling the global load vector ... ")
    F_global = assemble_global_load_vector(df_elements.iloc[0, 0], df_nodes, df_loads)

    logging.info(f"(3/6) Setting the boundary conditions ... ")
    K_global_bcs, F_global_bcs = impose_boundary_conditions_penalty_method(df_elements.iloc[0, 0], df_nodes, df_bcs, K_global, F_global)

    logging.info(f"(4/6) Solving the linear system ... ")
    displacement, forces = solve_system(K_global_bcs, F_global_bcs, K_global)

    logging.info(f"(5/6) Computing Post-Process ... ")
    df_displacement, df_forces, df_stress, df_strain = get_post_processing_output(displacement, forces, df_elements, df_nodes, df_materials, df_properties) 
    
    logging.info(f"(6/6) Printing the results ... ")
    df_displacement.to_csv(output_file_name, index=False, float_format='%.2e')

    df_forces.to_csv(output_file_name.replace("displacement", "forces"), index=False, float_format='%.2e')
    df_stress.to_csv(output_file_name.replace("displacement", "stresses"), index=False, float_format='%.2e')
    print("====================================================================================================")
    
if __name__ == '__main__':
    main(INPUT_FILE_NAME, OUTPUT_FILE_NAME)