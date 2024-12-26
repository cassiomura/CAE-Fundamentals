# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                          Title: PreProcessExcel.py
# ==============================================================================
import pandas as pd

def get_pre_processing_input_excel(filename: str, print_dataframes: bool) -> tuple: 
    df_nodes = create_nodes_dataframe(filename)
    df_elements = create_elements_dataframe(filename)
    df_materials = create_materials_dataframe(filename)
    df_properties = create_properties_dataframe(filename)
    df_loads = create_loads_dataframe(filename)
    df_boundary_conditions = create_boundary_conditions_dataframe(filename)
    
    return df_nodes, df_elements, df_materials, df_properties, df_loads, df_boundary_conditions

def create_nodes_dataframe(filename):
    df_nodes = pd.read_excel(
                filename,
                usecols= "C:F",
                sheet_name='data_nodes',
                skiprows=2)

    types = {'Node ID': int, 'X': float, 'Y': float, 'Z': float}
    df_nodes = df_nodes.astype(types)

    return df_nodes

def create_elements_dataframe(filename):
    df_elements = pd.read_excel(
                    filename,
                    usecols="C:F",
                    sheet_name="data_elements",
                    skiprows=2)

    types = {'Element Type': str, 'Element ID': int, 'Property': int, 'Connectivity': str}
    df_elements = df_elements.astype(types)

    return df_elements

def create_loads_dataframe(filename):
    df_loads = pd.read_excel(
                    filename,
                    usecols="C:J",
                    sheet_name="data_loads",
                    skiprows=2)

    types = {'Type': str, 'SID': int, 'G': int, 'CID': int, 'F': float, 'N1': float, 'N2': float, 'N3':float}
    df_loads = df_loads.astype(types)

    return df_loads

def create_materials_dataframe(filename):
    df_materials = pd.read_excel(
                filename,
                usecols= "C:E",
                sheet_name='data_materials',
                skiprows=2)
    
    types = {'Material ID': int, 'E': float, 'nu': float}
    df_materials = df_materials.astype(types)

    return df_materials
    
def create_boundary_conditions_dataframe(filename):
    df_boundary_conditions = pd.read_excel(
                filename,
                usecols= "C:F",
                sheet_name='data_bcs',
                skiprows=2)
    
    types = {'Type': str, 'SID': int, 'C': str, 'G': str}
    df_boundary_conditions = df_boundary_conditions.astype(types)

    return df_boundary_conditions

def create_properties_dataframe(filename):
    df_properties = pd.read_excel(
                filename,
                usecols= "C:K",
                sheet_name='data_properties',
                skiprows=2)
    
    types = {'Property name': str, 'Property type': str, 'Property ID': int, 'Material ID': int}
    df_properties = df_properties.astype(types)

    return df_properties