# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: execute_tests.py
# ==============================================================================
import pandas as pd
import logging
from PreProcessPatran import get_pre_processing_input_patran
from PreProcessExcel import get_pre_processing_input_excel
from PostProcessCAE import get_post_processing_output
from SolverCAE import assemble_global_stiffness_matrix, assemble_global_load_vector, solve_linear_static_analysis
import os

# Configure the logging format and level:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

current_dir = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE_NAME_CBAR_PATRAN = current_dir + "/Cases/Benchmark/CBAR/CBAR.bdf"
INPUT_FILE_NAME_CBAR_EXCEL = current_dir + "/Cases/Benchmark/CBAR/CBAR.xlsx"
OUTPUT_FILE_NAME_CBAR = current_dir + "/Cases/Benchmark/CBAR/CBAR_result_displacement.csv"
REFENCE_FILE_NAME_CBAR = current_dir + "/Cases/Benchmark/CBAR/CBAR_displacement_reference.csv"

INPUT_FILE_NAME_CTRIA3_PATRAN = current_dir + "/Cases/Benchmark/CTRIA3/CTRIA3.bdf"
INPUT_FILE_NAME_CTRIA3_EXCEL = current_dir + "/Cases/Benchmark/CTRIA3/CTRIA3.xlsx"
OUTPUT_FILE_NAME_CTRIA3 = current_dir + "/Cases/Benchmark/CTRIA3/CTRIA3_result_displacement.csv"
REFENCE_FILE_NAME_CTRIA3 = current_dir + "/Cases/Benchmark/CTRIA3/CTRIA3_displacement_reference.csv"

INPUT_FILE_NAME_CQUAD4_PATRAN = current_dir + "/Cases/Benchmark/CQUAD4/CQUAD4.bdf"
INPUT_FILE_NAME_CQUAD4_EXCEL = current_dir + "/Cases/Benchmark/CQUAD4/CQUAD4.xlsx"
OUTPUT_FILE_NAME_CQUAD4 = current_dir + "/Cases/Benchmark/CQUAD4/CQUAD4_result_displacement.csv"
REFENCE_FILE_NAME_CQUAD4 = current_dir + "/Cases/Benchmark/CQUAD4/CQUAD4_displacement_reference.csv"

INPUT_FILE_NAME_CTETRA_PATRAN = current_dir + "/Cases/Benchmark/CTETRA/CTETRA.bdf"
INPUT_FILE_NAME_CTETRA_EXCEL = current_dir + "/Cases/Benchmark/CTETRA/CTETRA.xlsx"
OUTPUT_FILE_NAME_CTETRA = current_dir + "/Cases/Benchmark/CTETRA/CTETRA_result_displacement.csv"
REFENCE_FILE_NAME_CTETRA = current_dir + "/Cases/Benchmark/CTETRA/CTETRA_displacement_reference.csv"

INPUT_FILE_NAME_CHEXA_PATRAN = current_dir + "/Cases/Benchmark/CHEXA/CHEXA.bdf"
INPUT_FILE_NAME_CHEXA_EXCEL = current_dir + "/Cases/Benchmark/CHEXA/CHEXA.xlsx"
OUTPUT_FILE_NAME_CHEXA = current_dir + "/Cases/Benchmark/CHEXA/CHEXA_result_displacement.csv"
REFENCE_FILE_NAME_CHEXA = current_dir + "/Cases/Benchmark/CHEXA/CHEXA_displacement_reference.csv"

def main(input_file_name: str, output_file_name: str) -> None:
    logging.info(f"(0/5) Reading data ... ")
    _, file_extension = os.path.splitext(input_file_name)
    if file_extension == '.bdf':
        df_nodes, df_elements, df_materials, df_properties, df_loads, df_bcs = get_pre_processing_input_patran(input_file_name, print_dataframes = False)
    elif file_extension == '.xlsx':
        df_nodes, df_elements, df_materials, df_properties, df_loads, df_bcs = get_pre_processing_input_excel(input_file_name, print_dataframes = False)

    # Save dataframes:
    #df_nodes.to_excel("df_nodes.xlsx", index=False)
    #df_elements.to_excel("df_elements.xlsx", index=False) 
    #df_materials.to_excel("df_materials.xlsx", index=False) 
    #df_properties.to_excel("df_properties.xlsx", index=False) 
    #df_loads.to_excel("df_loads.xlsx", index=False) 
    #df_bcs.to_excel("df_bcs.xlsx", index=False) 

    logging.info(f"(1/5) Assembling the global stiffness matrix ... ")
    K_global = assemble_global_stiffness_matrix(df_elements, df_nodes, df_materials, df_properties)
    
    logging.info(f"(2/5) Assembling the global load vector ... ")
    F_global = assemble_global_load_vector(df_elements.iloc[0, 0], df_nodes, df_loads)

    logging.info(f"(3/5) Setting the boundary conditions and solving the linear system ... ")
    displacement, forces = solve_linear_static_analysis("Penalty Method", df_elements.iloc[0, 0], df_nodes, df_bcs, K_global, F_global)
    #displacement, forces = solve_linear_static_analysis("Reduction Method", df_elements.iloc[0, 0], df_nodes, df_bcs, K_global, F_global)

    logging.info(f"(4/5) Computing Post-Process ... ")
    df_displacement, df_forces, df_stress, df_strain = get_post_processing_output(displacement, forces, df_elements, df_nodes, df_materials, df_properties) 
    
    logging.info(f"(5/5) Printing the results ... ")
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

def validate_and_print(file1, file2, label):
    if validate_csv_match(file1, file2):
        print(f"[✔] Validation successful: {label} CSV files are identical.")
    else:
        print(f"[✖] Validation failed: {label} CSV files are not identical.")

def validate_displacement_csv_files():
    files = [
        (OUTPUT_FILE_NAME_CBAR, REFENCE_FILE_NAME_CBAR, "CBAR"),
        (OUTPUT_FILE_NAME_CTRIA3, REFENCE_FILE_NAME_CTRIA3, "CTRIA3"),
        (OUTPUT_FILE_NAME_CQUAD4, REFENCE_FILE_NAME_CQUAD4, "CQUAD4"),
        (OUTPUT_FILE_NAME_CTETRA, REFENCE_FILE_NAME_CTETRA, "CTETRA"),
        (OUTPUT_FILE_NAME_CHEXA, REFENCE_FILE_NAME_CHEXA, "CHEXA")
    ]
    
    for file1, file2, label in files:
        validate_and_print(file1, file2, label)

if __name__ == '__main__':
    main(INPUT_FILE_NAME_CBAR_EXCEL, OUTPUT_FILE_NAME_CBAR)
    main(INPUT_FILE_NAME_CTRIA3_EXCEL, OUTPUT_FILE_NAME_CTRIA3)
    main(INPUT_FILE_NAME_CQUAD4_EXCEL, OUTPUT_FILE_NAME_CQUAD4)
    main(INPUT_FILE_NAME_CTETRA_EXCEL, OUTPUT_FILE_NAME_CTETRA)
    main(INPUT_FILE_NAME_CHEXA_EXCEL, OUTPUT_FILE_NAME_CHEXA)

    #main(INPUT_FILE_NAME_CBAR_PATRAN, OUTPUT_FILE_NAME_CBAR)
    #main(INPUT_FILE_NAME_CTRIA3_PATRAN, OUTPUT_FILE_NAME_CTRIA3)
    #main(INPUT_FILE_NAME_CQUAD4_PATRAN, OUTPUT_FILE_NAME_CQUAD4)
    #main(INPUT_FILE_NAME_CTETRA_PATRAN, OUTPUT_FILE_NAME_CTETRA)
    #main(INPUT_FILE_NAME_CHEXA_PATRAN, OUTPUT_FILE_NAME_CHEXA)

    validate_displacement_csv_files()