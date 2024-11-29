# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: execute_tests.py
# ==============================================================================
import pandas as pd
#from main import main
import logging
from PreProcessCAE import get_pre_processing_input
from PostProcessCAE import get_post_processing_output
from SolverCAE import solve_system, assemble_global_stiffness_matrix, assemble_global_load_vector, impose_boundary_conditions_penalty_method

# Configure the logging format and level:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Insert the .bdf file name (Ensure that the folder containing this script and the .bdf file is open in your IDE):
INPUT_FILE_NAME_1 = "Cases/Benchmark/CBAR/CBAR.bdf"
OUTPUT_FILE_NAME_1 = "Cases//Benchmark/CBAR/CBAR_result_displacement.csv"

INPUT_FILE_NAME_2 = "Cases/Benchmark/CTRIA3/CTRIA3.bdf"
OUTPUT_FILE_NAME_2 = "Cases/Benchmark/CTRIA3/CTRIA3_result_displacement.csv"

INPUT_FILE_NAME_3 = "Cases/Benchmark/CQUAD4/CQUAD4.bdf"
OUTPUT_FILE_NAME_3 = "Cases/Benchmark/CQUAD4/CQUAD4_result_displacement.csv"

INPUT_FILE_NAME_4 = "Cases/Benchmark/CTETRA/CTETRA.bdf"
OUTPUT_FILE_NAME_4 = "Cases/Benchmark/CTETRA/CTETRA_result_displacement.csv"

INPUT_FILE_NAME_5 = "Cases/Benchmark/CHEXA/CHEXA.bdf"
OUTPUT_FILE_NAME_5 = "Cases/Benchmark/CHEXA/CHEXA_result_displacement.csv"

def main(input_file_name: str, output_file_name: str) -> None:
    logging.info(f"(0/6) Reading data ... ")
    df_nodes, df_elements, df_materials, df_properties, df_loads, df_bcs = get_pre_processing_input(input_file_name, print_dataframes = False)
    
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
    #df_forces.to_csv(output_file_name.replace("displacement", "forces"), index=False, float_format='%.2e')
    #df_stress.to_csv(output_file_name.replace("displacement", "stresses"), index=False, float_format='%.2e')
    logging.info("==================================================================")

def validate_csv_match(file1: str, file2: str) -> bool:
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        return df1.equals(df2)
    except Exception as e:
        print(f"Error: {e}")
        return False
            
if __name__ == '__main__':
    main(INPUT_FILE_NAME_1, OUTPUT_FILE_NAME_1)
    main(INPUT_FILE_NAME_2, OUTPUT_FILE_NAME_2)
    main(INPUT_FILE_NAME_3, OUTPUT_FILE_NAME_3)
    main(INPUT_FILE_NAME_4, OUTPUT_FILE_NAME_4)
    main(INPUT_FILE_NAME_5, OUTPUT_FILE_NAME_5)

    if validate_csv_match("Cases/Benchmark/CTRIA3/CTRIA3_result_displacement.csv", "Cases/Benchmark/CTRIA3/CTRIA3_displacement_reference.csv"):
        print("[✔] Validation successful: CTRIA3 CSV files are identical.")
    else:
        print("[✖] Validation failed: CTRIA3 CSV files are not identical.")

    if validate_csv_match("Cases/Benchmark/CQUAD4/CQUAD4_result_displacement.csv", "Cases/Benchmark/CQUAD4/CQUAD4_displacement_reference.csv"):
        print("[✔] Validation successful: CQUAD4 CSV files are identical.")
    else:
        print("[✖] Validation failed: CQUAD4 CSV files are not identical.")
    
    if validate_csv_match("Cases/Benchmark/CBAR/CBAR_result_displacement.csv", "Cases/Benchmark/CBAR/CBAR_displacement_reference.csv"):
        print("[✔] Validation successful: CBAR CSV files are identical.")
    else:
        print("[✖] Validation failed: CBAR CSV files are not identical.")

    if validate_csv_match("Cases/Benchmark/CTETRA/CTETRA_result_displacement.csv", "Cases/Benchmark/CTETRA/CTETRA_displacement_reference.csv"):
        print("[✔] Validation successful: CTETRA CSV files are identical.")
    else:
        print("[✖] Validation failed: CTETRA CSV files are not identical.")

    if validate_csv_match("Cases/Benchmark/CHEXA/CHEXA_result_displacement.csv", "Cases/Benchmark/CHEXA/CHEXA_displacement_reference.csv"):
        print("[✔] Validation successful: CHEXA CSV files are identical.")
    else:
        print("[✖] Validation failed: CHEXA CSV files are not identical.")