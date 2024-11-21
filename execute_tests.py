# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: execute_tests.py
# ==============================================================================
import pandas as pd
from main import main

# Insert the .bdf file name (Ensure that the folder containing this script and the .bdf file is open in your IDE):
INPUT_FILE_NAME_1 = "Cases/CBAR_Benchmark/CBAR_Benchmark.bdf"
OUTPUT_FILE_NAME_1 = "Cases/CBAR_Benchmark/CBAR_displacement_result.csv"

INPUT_FILE_NAME_2 = "Cases/CTRIA3_Benchmark/CTRIA3_Benchmark.bdf"
OUTPUT_FILE_NAME_2 = "Cases/CTRIA3_Benchmark/CTRIA3_displacement_result.csv"

INPUT_FILE_NAME_3 = "Cases/CQUAD4_Benchmark/CQUAD4_Benchmark.bdf"
OUTPUT_FILE_NAME_3 = "Cases/CQUAD4_Benchmark/CQUAD4_displacement_result.csv"

INPUT_FILE_NAME_4 = "Cases/CTETRA_Benchmark/CTETRA_Benchmark.bdf"
OUTPUT_FILE_NAME_4 = "Cases/CTETRA_Benchmark/CTETRA_displacement_result.csv"

INPUT_FILE_NAME_5 = "Cases/CHEXA_Benchmark/CHEXA_Benchmark.bdf"
OUTPUT_FILE_NAME_5 = "Cases/CHEXA_Benchmark/CHEXA_displacement_result.csv"

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

    if validate_csv_match("Cases/CTRIA3_Benchmark/CTRIA3_displacement_result.csv", "Cases/CTRIA3_Benchmark/CTRIA3_displacement_reference.csv"):
        print("[✔] Validation successful: CTRIA3 CSV files are identical.")
    else:
        print("[✖] Validation failed: CTRIA3 CSV files are not identical.")

    if validate_csv_match("Cases/CQUAD4_Benchmark/CQUAD4_displacement_result.csv", "Cases/CQUAD4_Benchmark/CQUAD4_displacement_reference.csv"):
        print("[✔] Validation successful: CQUAD4 CSV files are identical.")
    else:
        print("[✖] Validation failed: CQUAD4 CSV files are not identical.")
    
    if validate_csv_match("Cases/CBAR_Benchmark/CBAR_displacement_result.csv", "Cases/CBAR_Benchmark/CBAR_displacement_reference.csv"):
        print("[✔] Validation successful: CBAR CSV files are identical.")
    else:
        print("[✖] Validation failed: CBAR CSV files are not identical.")

    if validate_csv_match("Cases/CTETRA_Benchmark/CTETRA_displacement_result.csv", "Cases/CTETRA_Benchmark/CTETRA_displacement_reference.csv"):
        print("[✔] Validation successful: CTETRA CSV files are identical.")
    else:
        print("[✖] Validation failed: CTETRA CSV files are not identical.")

    if validate_csv_match("Cases/CHEXA_Benchmark/CHEXA_displacement_result.csv", "Cases/CHEXA_Benchmark/CHEXA_displacement_reference.csv"):
        print("[✔] Validation successful: CHEXA CSV files are identical.")
    else:
        print("[✖] Validation failed: CHEXA CSV files are not identical.")