------------
Introduction
------------
This repository contains the material submitted for Module 3 of the Python course. The project demonstrates the application of Python and the SUND (Simulation and UNcertainty in Dynamical systems) toolbox to a larger systems biology modelling problem.

The original project focused on developing, calibrating, validating, and simulating a mechanistic mouse model of metabolic regulation and pharmacology. To comply with copyright agreements and confidentiality requirements, portions of the original repository have been removed or simplified. This includes selected datasets, manuscript material, figures, and other project-specific resources that cannot be redistributed.

The remaining material is intended to demonstrate the overall project structure, modelling workflow, and Python programming techniques developed during the course.

-----------------
Required software
-----------------

The following software packages are required:

- Python
- SUND toolbox
- NumPy
- Pandas
- Matplotlib
- SciPy
- Joblib
- pyswarm
- pyswarms

Additional packages may be required depending on the specific functionality being tested.

-----------------
Running the project
-----------------

The main entry point is:

    Main.py

Run the script from your preferred Python environment:

    python Main.py

All analyses should be started through Main.py. The main script loads the required data, models, parameter files, and functions in the correct order and serves as the central entry point for the project.

Individual functions and scripts are generally not intended to be executed directly.

-----------------
Project content
-----------------

The project demonstrates a complete modelling workflow, including:

- Loading and preprocessing experimental data
- Building mathematical models using the SUND toolbox
- Running model simulations
- Parameter estimation and optimisation
- Uncertainty analysis
- Model validation
- Generation of figures and simulation outputs

The supplied version is a reduced course submission and therefore does not necessarily reproduce all analyses or figures from the original research project.

-----------------
Repository structure
--------------------

Main.py
    Main entry point used to run the project.

Functions/
    Collection of utility functions used for simulations, optimisation, plotting, data handling, and model evaluation.

Models/
    Model definitions used by the SUND toolbox.

Parameters/
    Example parameter sets and optimisation results.

Data/
    Example datasets included for educational purposes.

-----------------
Notes
-----------------

This project was submitted as part of Module 3 of the Python course and is intended to demonstrate the practical application of Python programming to a scientific modelling problem.

Some original project files, datasets, figures, and supporting material have been removed due to copyright restrictions, publication requirements, or confidentiality considerations. The remaining files are sufficient to illustrate the project structure, workflow, and programming concepts developed during the course.