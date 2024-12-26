# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                         Title: PreProcessPatran.py
# ==============================================================================
import pandas as pd

def get_pre_processing_input_patran(filename: str, print_dataframes: bool) -> tuple: 
    # 1. Generate the Node DataFrame:
    df_nodes = create_nodes_dataframe(filename)

    # 2. Generate the Element DataFrame:
    df_elements = create_elements_dataframe(filename)

    # 3. Generate the Material DataFrame:
    df_materials = create_materials_dataframe(filename)

    # 3. Generate the Properties DataFrame:
    df_properties = create_properties_dataframe(filename)

    # 4. Generate the Loads DataFrame:
    df_loads = create_loads_dataframe(filename)

    # 5. Generate the Boundary Conditions DataFrame:
    df_boundary_conditions = create_boundary_conditions_dataframe(filename)
    
    # Print the data if requested:
    if print_dataframes:
        print_all_dataframes(df_nodes, df_elements, df_materials, df_properties, df_loads, df_boundary_conditions)
        
    return df_nodes, df_elements, df_materials, df_properties, df_loads, df_boundary_conditions

def create_nodes_dataframe(filename: str) -> pd.DataFrame:
    ids = []
    x_coords = []
    y_coords = []
    z_coords = []

    startHeader = "Nodes of the Entire Model"
    endHeader = " Loads for Load Case"
    
    read_line = False
    with open(filename, 'r') as file:
        for line in file:
            if startHeader in line.strip():
                read_line = True
            elif endHeader in line.strip():
                break

            splitted_line = line.split() 
            if read_line:
                # Error of 8 digits float: e.g. GRID     205            .09179132.55037  0.
                if len(splitted_line) < 5 and splitted_line[0] == 'GRID':
                    part_1 = splitted_line[2][:8]
                    part_2 = splitted_line[2][8:]
                    splitted_line = splitted_line[:2] + [part_1, part_2, splitted_line[3]]

                if splitted_line[0] == 'GRID':
                    ids.append(splitted_line[1])
                    x_coords.append(splitted_line[2])
                    y_coords.append(splitted_line[3])
                    z_coords.append(splitted_line[4])
                elif splitted_line[0] == 'GRID*':
                    ids.append(splitted_line[1])
                    x_coords.append(splitted_line[2])
                    y_coords.append(splitted_line[3])
                elif splitted_line[0] == "*":
                    z_coords.append(splitted_line[1])

    # Create the data frame:
    df_nodes = pd.DataFrame({
        'Node ID': ids,
        'X': x_coords,
        'Y': y_coords,
        'Z': z_coords
    })

    # Correct the scientific notation when needed (2.32458-8 to 2.32458E-08).
    df_nodes[['X', 'Y', 'Z']] = df_nodes[['X', 'Y', 'Z']].apply(lambda col: col.map(fix_scientific_notation))

    types = {'Node ID': int, 'X': float, 'Y': float, 'Z': float}
    df_nodes = df_nodes.astype(types)

    return df_nodes

def create_elements_dataframe(filename: str) -> pd.DataFrame:
    TYPE = []               # Element Type
    EID = []                # Element identication number. (0 < Integer < 100,000,000)
    PID = []                # Property identification number of a PSHELL, PCOMP, PCOMPG or PLPLANE or PLCOMP entry. (Integer > 0; Default = EID)
    Gi = []                 # Grid points identifivation numbers of connection points.

    nodes_per_element_map = {
    'CTRIA3': 3,
    'CQUAD4': 4,
    'CROD': 2,
    'CBAR': 2,
    'CTETRA': 4,
    'CHEXA': 8,
    }

    startHeader = '$ Direct Text Input for Bulk Data'
    endHeader = 'Referenced Material Records'

    read_line = False
    with open(filename, 'r') as file:
        for line in file:
            if startHeader in line.strip():
                read_line = True
                continue
            elif endHeader in line.strip():
                break
            splitted_line = line.split()

            element_type = splitted_line[0]
            if read_line and element_type in nodes_per_element_map:
                # Determine number of nodes per element:
                nodes_per_element = nodes_per_element_map.get(element_type, None)

                if element_type == 'CHEXA':
                    # Process the current line
                    first_line_nodes = line.split()[3:9]
                    # Read the next line for additional nodes
                    next_line = next(file).strip()  # Advances the iterator
                    second_line_nodes = next_line.split()[:2]

                    # Combine nodes
                    connectivity = ' '.join(first_line_nodes + second_line_nodes)
                else:
                    connectivity = ' '.join(splitted_line[3:3 + nodes_per_element])
    
                # Append values to lists
                TYPE.append(splitted_line[0])
                EID.append(splitted_line[1])
                PID.append(splitted_line[2])
                Gi.append(connectivity)


    # Generate the dataframe:
    df_elements = pd.DataFrame({
        'Element Type': TYPE,
        'Element ID': EID,
        'Property': PID,
        'Connectivity': Gi
        })
    
    types = {'Element Type': str, 'Element ID': int, 'Property': int, 'Connectivity': str}
    df_elements = df_elements.astype(types)

    return df_elements

def create_materials_dataframe(filename: str) -> pd.DataFrame:
    name = []
    ids = []
    E = []
    nu = []

    startHeader = "Referenced Material Records"
    endHeader = " Nodes of the Entire Model"

    read_line = False
    with open(filename, 'r') as file:
        for line in file:
            if startHeader in line.strip():
                read_line = True
            elif endHeader in line.strip():
                break

            splitted_line = line.split() 
            if read_line:
                if splitted_line[0] == 'MAT1':
                    ids.append(splitted_line[1])
                    E.append(splitted_line[2])
                    if len(splitted_line) > 3: # CROD element do not admit a nu value.
                        nu.append(splitted_line[3])
                    else:
                        nu.append(None)

    # Create the data frame:
    df_materials = pd.DataFrame({
        'Material ID': ids,
        'E': E,
        'nu': nu
    })

    # Correct the scientific notation when needed (2.32458-8 to 2.32458E-08).
    df_materials[['E', 'nu']] = df_materials[['E', 'nu']].apply(lambda col: col.map(fix_scientific_notation))

    types = {'Material ID': int, 'E': float, 'nu': float}
    df_materials = df_materials.astype(types)
    #df_materials['Material ID'] = pd.to_numeric(df_materials['Material ID'], downcast='integer', errors='coerce')
    #df_materials['E'] = pd.to_numeric(df_materials['E'], errors='coerce')
    #df_materials['nu'] = pd.to_numeric(df_materials['nu'], errors='coerce')

    return df_materials

def create_properties_dataframe(filename: str) -> pd.DataFrame:
    property_name = []
    property_type = []
    property_id = []
    material_id = []
    # Specific properties
    thickness = []
    area = []
    I11 = []
    I22 = []
    J = []

    properties_map = {
    # (Thickness, Area, I11, I22, J)
    'PSHELL': (3, None, None, None, None),
    'PROD': (None, 3, None, None, None),
    'PBAR': (None, 3, 4, 5, 6),
    'PSOLID': (None, None, None, None, None),
    }
    
    startHeader = '$ Direct Text Input for Bulk Data'
    endHeader = 'Referenced Material Records'

    read_line = False
    with open(filename, 'r') as file:
        for line in file:
            if startHeader in line.strip():
                read_line = True
                continue
            elif endHeader in line.strip():
                break

            splitted_line = line.split()
            if read_line:
                if '$ Elements and Element Properties for region :' in line.strip():
                    property_name.append(splitted_line[-1])
                
                prop_type = splitted_line[0]
                if splitted_line[0] in properties_map:
                    # Read general properties
                    property_type.append(splitted_line[0])
                    property_id.append(splitted_line[1])
                    material_id.append(splitted_line[2])
                    # Read specific properties
                    property = properties_map[prop_type]
                    thickness.append(splitted_line[property[0]] if property[0] is not None else None)
                    area.append(splitted_line[property[1]] if property[1] is not None else None)
                    I11.append(splitted_line[property[2]] if property[2] is not None else None)
                    I22.append(splitted_line[property[3]] if property[3] is not None else None)
                    J.append(splitted_line[property[4]] if property[4] is not None else None)

    # Create a DataFrame from the collected data
    df_properties = pd.DataFrame({
        'Property name': property_name,
        'Property type': property_type,
        'Property ID': property_id,
        'Material ID': material_id,
        'Thickness': thickness,
        'Area': area,
        'I11': I11,
        'I22': I22,
        'J': J
    })

    # Drop columns that contain only None values
    df_properties.dropna(axis=1, how='all', inplace=True)

    types = {'Property name': str, 'Property type': str, 'Property ID': int, 'Material ID': int}
    df_properties = df_properties.astype(types)

    return df_properties

def create_loads_dataframe(filename: str) -> pd.DataFrame:
    Type = []       # FORCE
    SID = []        # Load set identification number. (Integer > 0)
    G = []          # Grid point identification number. (Integer > 0)
    CID = []        # Coordinate system identification number. (Integer > 0)
    F = []          # Scale factor. (Real)          
    N1 = []         # Component of a vector measured in coodinate system defined by CID (Real)
    N2 = []         # Component of a vector measured in coodinate system defined by CID (Real)
    N3 = []         # Component of a vector measured in coodinate system defined by CID (Real)

    startHeader = '$ Nodal Forces of Load Set'
    endHeader = '$ Referenced Coordinate Frames'

    read_line = False
    with open(filename, 'r') as file:
        for line in file:
            if startHeader in line.strip():
                read_line = True
                continue
            elif endHeader in line.strip():
                break

            splitted_line = line.split()
            if read_line and splitted_line[0] == 'FORCE' or splitted_line[0] == 'MOMENT':
                Type.append(splitted_line[0])
                SID.append(splitted_line[1])
                G.append(splitted_line[2])
                CID.append(splitted_line[3])
                F.append(splitted_line[4])
                N1.append(splitted_line[5])
                N2.append(splitted_line[6])
                N3.append(splitted_line[7])

    # Generate the dataframe:
    df_loads = pd.DataFrame({
        'Type': Type,
        'SID': SID,
        'G': G,
        'CID': CID,
        'F': F,
        'N1': N1,
        'N2': N2,
        'N3': N3
        })
    
    types = {'Type': str, 'SID': int, 'G': int, 'CID': int, 'F': float, 'N1': float, 'N2': float, 'N3':float}
    df_loads = df_loads.astype(types)
    
    return df_loads

def create_boundary_conditions_dataframe(filename: str) -> pd.DataFrame:
    Type = []  # SPC1
    SID = []   # Identification number of single-point constraint set. (Integer > 0)
    C = []     # Component numbers (e.g., 123456 for grid points.)
    G = []     # Grid or scalar point identification numbers. (Integer > 0)

    start_header = '$ Displacement Constraints of Load Set'
    end_header = '$'  # Generic end marker for the section

    read_line = False
    current_type = None
    current_sid = None
    current_c = None
    node_list = []

    with open(filename, 'r') as file:
        for line in file:
            stripped_line = line.strip()

            # Detect start of the boundary condition section
            if start_header in stripped_line:
                read_line = True
                continue
            # Detect the end of the section
            elif read_line and stripped_line.startswith(end_header):
                break
            
            if read_line:
                splitted_line = stripped_line.split()

                # Detect a new `SPC1` entry
                if splitted_line[0] == 'SPC1':
                    # Append previous data to lists if a new entry starts
                    if current_type and node_list:
                        Type.append(current_type)
                        SID.append(current_sid)
                        C.append(current_c)
                        G.append(" ".join(map(str, node_list)))

                    # Reset for the new entry
                    current_type = splitted_line[0]
                    current_sid = splitted_line[1]
                    current_c = splitted_line[2]
                    node_list = []  # Reset node list

                    # Add nodes from the current line
                    for idx in range(3, len(splitted_line)):
                        if splitted_line[idx] == 'THRU':
                            start_node = int(splitted_line[idx - 1])
                            end_node = int(splitted_line[idx + 1])
                            node_list.extend(range(start_node, end_node + 1))
                        elif splitted_line[idx - 1] != 'THRU':
                            node_list.append(int(splitted_line[idx]))

                # Handle continuation lines
                else:
                    for idx in range(len(splitted_line)):
                        if splitted_line[idx] == 'THRU':
                            start_node = int(splitted_line[idx - 1])
                            end_node = int(splitted_line[idx + 1])
                            node_list.extend(range(start_node, end_node + 1))
                        elif splitted_line[idx - 1] != 'THRU':
                            node_list.append(int(splitted_line[idx]))
        
        # Append the last set of data if the file ends
        if current_type and node_list:
            Type.append(current_type)
            SID.append(current_sid)
            C.append(current_c)
            G.append(" ".join(map(str, node_list)))

    # Generate the DataFrame
    df_boundary_conditions = pd.DataFrame({
        'Type': Type,
        'SID': SID,
        'C': C,
        'G': G
    })

    types = {'Type': str, 'SID': int, 'C': str, 'G': str}
    df_boundary_conditions = df_boundary_conditions.astype(types)

    return df_boundary_conditions

def print_all_dataframes(df_nodes: pd.DataFrame, df_elements: pd.DataFrame, df_materials: pd.DataFrame, df_properties: pd.DataFrame, df_loads: pd.DataFrame, df_boundary_conditions: pd.DataFrame) -> None:
        # Print all data frames into the console
        print_dataframe_configuration(df_nodes, "Nodes Data Frame")
        print_dataframe_configuration(df_elements, "Elements Data Frame")
        print_dataframe_configuration(df_materials, "Material Data Frame")
        print_dataframe_configuration(df_properties, "Properties Data Frame")
        print_dataframe_configuration(df_loads, "Loads Data Frame")
        print_dataframe_configuration(df_boundary_conditions, "Boundary Conditions Data Frame")
    
def print_dataframe_configuration(df: pd.DataFrame, title: str) -> None:
        # Configuration of the print of the dataframe.
        print(f"\n{'='*20} {title} {'='*20}")
        print(f"Total Rows: {len(df)}\n")
        print(df.head())  
        print(f"\n{'-'*60}\n")

def fix_scientific_notation(value: str) -> str:
    # Correction (Example: -2.123-4 -> -2.123E-4):
    if isinstance(value, str) and '-' in value:
        # Split the string only once at the first '-'
        base, exponent = value.split('-', 1)
        # Check if the exponent part is a valid number or starts with 'E'
        if exponent.isdigit() or (exponent.startswith('E') and exponent[1:].isdigit()):
            return f'{base}E-{exponent}'
        
    # Correction (Example: 2.1+9 -> 2.1E+9):
    elif isinstance(value, str) and '+' in value:
        base, exponent = value.split('+', 1)
        # Strip any spaces in the exponent and check validity
        exponent = exponent.strip()
        if exponent.isdigit() or (exponent.startswith('E') and exponent[1:].isdigit()):
            return f'{base}E+{exponent}'
        
    return value