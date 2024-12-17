# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                           Title: InterfaceCAE.py
# ==============================================================================
import time
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PreProcessCAE import get_pre_processing_input
from PostProcessCAE import get_post_processing_output, save_results_to_csv, plot_mesh, plot_results_displacement, plot_results_stress
from SolverCAE import solve_linear_static_analysis, assemble_global_stiffness_matrix, assemble_global_load_vector

class InterfaceCAE:

    def __init__(self):
        # Set up the main application window
        self.main_window = tk.Tk()
        self.main_window.geometry("800x500")
        self.main_window.title("CAE Fundamentals | Developer: Cássio Murakami")
        self.main_window.configure(bg = "white")
        self.main_window.resizable(True, True)

        # Add the application title
        self.title_label = tk.Label(self.main_window, text='CAE Fundamentals', font=('Montserrat', 32, 'bold'),  bg="white", fg = "black")
        self.title_label.pack(padx = 10, pady = 10)

        # Create frames for different functionalities
        self.create_preprocess_frame()
        self.create_solver_frame()
        self.create_postprocess_frame()

        # Initializators:
        self.filepath = False
        self.system_solved = False

        # Start the main event loop
        self.main_window.mainloop()

    def create_preprocess_frame(self) -> None:
        # Frame for preprocessing section
        self.preprocess_frame = tk.Frame(self.main_window,
                                         highlightbackground="black", 
                                         highlightthickness=1
                                        )
        self.preprocess_frame.pack(side=tk.TOP, anchor="w", padx = 10, pady=10, fill = "both")  # Pack at the top, align to the left

        # Label for instruction to select a file
        self.preprocess_label = tk.Label(self.preprocess_frame, text="1. Select the preprocess file", font=('Helvetica', 16, 'bold'))
        self.preprocess_label.pack(side=tk.LEFT, padx=10)

        # Button to open file dialog for selecting preprocess file
        self.preprocess_button = tk.Button(self.preprocess_frame, text="Select File", command=self.select_file_button)
        self.preprocess_button.pack(side=tk.LEFT, padx=10)
        
        # Text box to display the selected file path (read-only)
        self.preprocess_file_text = tk.Text(self.preprocess_frame, height=1, width=100, font=('Times New Roman', 8), state = 'disabled')
        self.preprocess_file_text.pack(padx=10, pady=10)

    def create_solver_frame(self) -> None:
        # Frame for solver section
        self.solver_frame = tk.Frame(self.main_window,
                                     highlightbackground="black", 
                                     highlightthickness=1,
                                    )         
        self.solver_frame.pack(side = tk.LEFT, anchor = 'w', padx = 10, pady = 10, fill = "both", expand = True)

        # Label for solver section
        self.solver_label = tk.Label(self.solver_frame, text="2. Solve System", font=('Helvetica', 16, 'bold'))
        self.solver_label.pack(side=tk.TOP, padx=10)

        # Create a sub-frame to contain the label and combobox side by side
        combobox_frame = tk.Frame(self.solver_frame)
        combobox_frame.pack(anchor='w', padx=10, pady=10)

        # Label next to the combobox
        self.solver_label_text = tk.Label(combobox_frame, text="Select Solver Method: ", font=('Times New Roman', 12))
        self.solver_label_text.pack(side=tk.LEFT)

        # Combobox to select the solver method
        solver_methods = ["Reduction Method", "Penalty Method"]
        self.solver_selection_box = tk.ttk.Combobox(combobox_frame, values=solver_methods)
        self.solver_selection_box.set("Reduction Method")
        self.solver_selection_box.pack(side=tk.LEFT, padx=5)

        # Button to trigger the solver
        self.solve_button = tk.Button(self.solver_frame, text = "Solve", font = ('Times New Roman', 16), command = self.solve_button, width = 30)
        self.solve_button.pack(padx = 10, pady = 10)

        # Text box to display solver progress and messages
        self.solver_text = tk.Text(self.solver_frame, height=10, width=60, font=('Times New Roman', 8), state = 'disabled')
        self.solver_text.pack(padx=10, pady=10)

    def create_postprocess_frame(self) -> None:
        # Frame for postprocessing section 
        self.postprocess_frame = tk.Frame(self.main_window,
                                          highlightbackground="black", 
                                          highlightthickness=1
                                          )
        self.postprocess_frame.pack(side = tk.RIGHT, anchor='w', padx= 10, pady= 10, fill = 'both', expand =True)

        # Label for the postprocessing section
        self.postprocess_label = tk.Label(self.postprocess_frame, text="3. Display Results", font=('Helvetica', 16, 'bold'))
        self.postprocess_label.pack(side=tk.TOP, padx=10)

        # Button to plot the mesh
        self.plot_mesh_button = tk.Button(self.postprocess_frame, text = "Plot Mesh", font = ('Times New Roman', 16), command = self.plot_mesh_button, width = 30)
        self.plot_mesh_button.pack(padx = 10, pady = 30)

        # Button to plot displacement results
        self.plot_displacement_button = tk.Button(self.postprocess_frame, text = "Plot Displacement", font = ('Times New Roman', 16), command = self.plot_displacement_button, width = 30)
        self.plot_displacement_button.pack(padx = 10, pady = 10)

        # Button to plot stress results
        self.plot_stress_button = tk.Button(self.postprocess_frame, text = "Plot Stress", font = ('Times New Roman', 16), command = self.plot_stress_button, width = 30)
        self.plot_stress_button.pack(padx = 10, pady = 10)

    def select_file_button(self) -> None:
        # Open a file dialog in the script's directory, restricted to .bdf file
        file = filedialog.askopenfile(
            mode='r', 
            filetypes=[('Patran Files', '*.bdf')], 
            initialdir = os.path.dirname(os.path.abspath(__file__)))
        if file:
            # Display the selected file path in the text box (read-only)
            self.filepath = os.path.abspath(file.name)
            self.preprocess_file_text.configure(state="normal")
            self.preprocess_file_text.delete(1.0, tk.END)  # Clear existing text
            self.preprocess_file_text.insert(tk.END, self.filepath)
            self.preprocess_file_text.configure(state="disabled")

    def solve_button(self) -> None:
        """Handles the solve button click event to run the FEM solver."""
        if not self.filepath:
            messagebox.showerror("Error", "Insert a filepath.")
            return
  
        self.solver_text.configure(state="normal")

        self.solver_text.insert(tk.END, "(0/5) Reading data ... ")
        self.solver_text.update_idletasks()
        start_time = time.time()
        self.df_nodes, self.df_elements, self.df_materials, self.df_properties, self.df_loads, self.df_bcs = get_pre_processing_input(self.filepath, print_dataframes = False)
        end_time = time.time()
        time_spent = end_time - start_time
        self.solver_text.insert(tk.END, f"(Duration: {time_spent:.1f} s)\n")
        self.solver_text.update_idletasks()
        
        self.solver_text.insert(tk.END, "(1/5) Assembling the global stiffness matrix ... ")
        self.solver_text.update_idletasks()
        start_time = time.time()
        K_global = assemble_global_stiffness_matrix(self.df_elements, self.df_nodes, self.df_materials, self.df_properties)
        end_time = time.time()
        time_spent = end_time - start_time
        self.solver_text.insert(tk.END, f"(Duration: {time_spent:.1f} s)\n")
        self.solver_text.update_idletasks()

        self.solver_text.insert(tk.END, "(2/5) Assembling the global load vector ... ")
        self.solver_text.update_idletasks()
        start_time = time.time()
        F_global = assemble_global_load_vector(self.df_elements.iloc[0, 0], self.df_nodes, self.df_loads)
        end_time = time.time()
        time_spent = end_time - start_time
        self.solver_text.insert(tk.END, f"(Duration: {time_spent:.1f} s)\n")
        self.solver_text.update_idletasks()

        self.solver_text.insert(tk.END, f"(3/5) Solving the system using the {self.solver_selection_box.get()} ... ")
        self.solver_text.update_idletasks
        start_time = time.time()
        displacement, forces = solve_linear_static_analysis(self.solver_selection_box.get(), self.df_elements.iloc[0, 0], self.df_nodes, self.df_bcs, K_global, F_global)
        end_time = time.time()
        time_spent = end_time - start_time
        self.solver_text.insert(tk.END, f"(Duration: {time_spent:.1f} s)\n")
        self.solver_text.update_idletasks()

        self.solver_text.insert(tk.END, "(4/5) Computing Post-Process ... ")
        self.solver_text.update_idletasks()
        start_time = time.time()
        self.df_displacement, self.df_forces, self.df_stress, self.df_strain = get_post_processing_output(displacement, forces, self.df_elements, self.df_nodes, self.df_materials, self.df_properties)
        end_time = time.time()
        time_spent = end_time - start_time
        self.solver_text.insert(tk.END, f"(Duration: {time_spent:.1f} s)\n")
        self.solver_text.update_idletasks()

        self.solver_text.insert(tk.END, "(5/5) Saving the results as CSV ... ")
        self.solver_text.update_idletasks()
        start_time = time.time()
        save_results_to_csv(self.filepath, self.df_displacement, self.df_forces, self.df_stress)
        end_time = time.time()
        time_spent = end_time - start_time
        self.solver_text.insert(tk.END, f"(Duration: {time_spent:.1f} s)\n")
        self.solver_text.update_idletasks()

        self.solver_text.configure(state="disabled")
        self.system_solved = True
        
    def plot_mesh_button(self) -> None:
        """Handles the "Plot Mesh" button click: validates prerequisites and generates a displacement plot."""

        if not self.filepath:
            messagebox.showerror("Error", "Insert a filepath.")
            return

        if not self.system_solved:
            messagebox.showerror("Error", "Solve the system.")
            return
        
        try:
            plot_mesh(self.df_elements, self.df_nodes)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while plotting mesh: {e}")
        return

    def plot_displacement_button(self) -> None:
        """Handles the "Plot Displacement" button click: validates prerequisites and generates a displacement plot."""

        if not self.filepath:
            messagebox.showerror("Error", "Insert a filepath.")
            return

        if not self.system_solved:
            messagebox.showerror("Error", "Solve the system.")
            return
        
        try:
            plot_results_displacement(self.df_elements, self.df_nodes, self.df_displacement)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while plotting displacement result: {e}")

    def plot_stress_button(self) -> None:
        """Handles the "Plot Stress" button click: validates prerequisites and generates a stress plot."""

        if not self.filepath:
            messagebox.showerror("Error", "Insert a filepath.")
            return

        if not self.system_solved:
            messagebox.showerror("Error", "Solve the system.")
            return

        try:
            plot_results_stress(self.df_elements, self.df_nodes, self.df_stress)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while plotting stree result: {e}")
