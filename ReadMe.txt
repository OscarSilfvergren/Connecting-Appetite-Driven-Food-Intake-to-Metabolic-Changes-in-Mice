A Mathematical Model Connecting Appetite-Driven Food Intake to Metabolic Changes across Diet and GLP-1RA Treatments in Mice

Oscar Silfvergren1,2, Christian Simonsson3, Monika Sundqvist3, Pär Nordell3, Peter Gennemark1,3, Gunnar Cedersund1,2,4,5,*

1. Department of Biomedical Engineering, Linköping University, Linköping, Sweden.  
2. SUND sound medical decisions, Linköping, Sweden. 
3. Drug Metabolism and Pharmacokinetics, Research and Early Development, Cardiovascular, Renal and Metabolism (CVRM), BioPharmaceuticals R&D, AstraZeneca, Gothenburg, Sweden.
4. Centre for Medical Image Science and Visualisation (CMIV), Linköping University, Linköping, Sweden.
5. School of Medical Sciences and Inflammatory Response and Infection Susceptibility Centre (iRiSC), Faculty of Medicine and Health, Örebro University, Örebro, Sweden.



---------------------------------------------------------------------------------------------------------------------------------------------------------
This repository contains the code used to develop, calibrate, validate, and simulate a mechanistic mouse model describing:

- Appetite regulation
- Food intake and meal patterns
- Body-weight dynamics
- Glucose-insulin regulation
- Exenatide pharmacology
- Semaglutide pharmacology

The model was implemented using the SUND (Simulation and UNcertainty in Dynamical systems) toolbox.

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

The model files (*.txt), parameter files (*.csv), and experimental datasets (*.xlsx) must remain in their provided locations within the repository.

-----------------
Running the model
-----------------
The main entry point is:

    Main.py

Run the script from your preferred Python environment:

    python Main.py

When launched, a graphical user interface (GUI) will appear.

The GUI allows you to select which manuscript figure to generate and whether to create:
- Rapid plot
- More complete plot

Available figures include:
- Guo & Hall 2011
- Mack et al. 2006 (short-term)
- Mack et al. 2006 (long-term)
- He et al. 2020
- Semaglutide studies
- Gabery et al. 2020
- Digital mouse population simulations
- Supplementary figures

After selecting a figure and plot mode, click "Plot" to generate the selected simulation and figure.

-----------------
Parameter estimation
--------------------
Parameter estimation routines are implemented in:

    optimise.py

Most optimisation calls are disabled by default in Main.py to avoid unintentionally launching computationally expensive optimisation procedures.

To perform optimisation, uncomment the relevant optimisation function call in Main.py and run the script.

-----------------
Repository structure
--------------------
Main.py
    Entry point and plotting GUI.

Functions/functions.py
    Data loading and utility functions.

Functions/simulate.py
    Simulation routines.

Functions/cost.py
    Cost-function calculations.

Functions/optimise.py
    Parameter estimation and profile-likelihood routines.

Functions/plots.py
    Plotting functions used to generate manuscript figures.

Models/
    Model definitions used by the SUND toolbox.

Parameters/
    Optimised parameter sets and profile-likelihood results.

MiceBW_data.xlsx
    Experimental data used for model calibration and validation.


Most users only need to run Main.py and use the graphical interface to reproduce the figures presented in the manuscript.
