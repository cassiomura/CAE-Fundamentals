# ==============================================================================
#                        CASSIO MURAKAMI - FEM SOLVER
#                         Project: CAE Fundamentals
#                              Title: main.py
# ==============================================================================

""" 
                               PROJECT SCOPE
 ==============================================================================
 This script implements a Finite Element Method (FEM) solver with an integrated 
 Graphical User Interface (GUI) for model setup and analysis, coupled with 
 post-processing visualization tools for result interpretation.

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

from InterfaceCAE import InterfaceCAE
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """Main entry point for the FEM solver, initializing and logging execution."""

    try:
        setup_logging()

        logging.info("Launching the FEM Solver user interface and processing data...")
        InterfaceCAE()

        logging.info("FEM Solver execution completed successfully. Exiting...")
        exit(0)

    except Exception as e:
        logging.error(f"Unexpected error during solver execution: {e}")
        exit(1)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    main()