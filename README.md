# 💻 CAE Fundamentals
![GUI_interface](https://github.com/user-attachments/assets/9ad87d7b-9b9a-4c85-96f5-d6e6711b480b)
## 🏁 Objective

Develop a Finite Element Method (FEM)-based structural analysis solver capable of processing input data from native commercial software files (e.g., Patran) or custom, intuitive Excel tables.

## :book: Wiki
For detailed information on the theoretical concepts and development of the Finite Element Method used in this project, please visit the project [Wiki](https://github.com/cassiomura/CAE-Fundamentals/wiki).

## 🔧 Implementation
This **Python-based** project implements a structural analysis solver using the Finite Element Method with an object-oriented approach. Each element type is represented as a class, promoting modularity and algorithm reuse. The project is structured into distinct modules, encompassing pre-processing, the solver, post-processing, and the graphical user interface (GUI).

## 🛠️ Dependencies
- [numpy](https://pypi.org/project/numpy/) - Fundamental package for scientific computing with Python.
- [pyvista](https://pypi.org/project/pyvista/) - Plotting package.
- [pandas](https://pypi.org/project/pandas/) - Powerful data structures for data analysis.
- [openpyxl](https://pypi.org/project/openpyxl/) - Read/write Excel files.
## 🎯 Project Scope
### Input Format
The solver accepts input data in the `.bdf` format, which is the native pre-processing file format used by **Patran**. Alternatively, the solver accepts input in `.xlsx` format via a structured Excel table.
### Supported Element Types
- **CBAR**: Linear Bar Elements.
- **CTRIA3**: Linear Triangular Elements.
- **CQUAD4**: Linear Quadrilateral Elements.
- **CTETRA**: Linear Tetrahedral Elements.
- **CHEXA**: Linear Hexahedral Elements.
### Load Application
  External loads are limited to **nodal forces**. Distributed forces are not supported.
### Analysis Type
  This solver is designed exclusively for **linear static analysis**.

## :thought_balloon: Inspiration
This project was inspired by the ambition to implement the numerical formulation of structural analysis from the ground up, drawing on foundational principles of the Finite Element Method and guided by the reference book _"Elementos Finitos – A base da tecnologia CAE by Alves Filho, A. (2018), Saraiva Educação S.A."_

## 👦 Contributors
- **Cássio Murakami**
## 📑 License
This project is licensed under the MIT License.
