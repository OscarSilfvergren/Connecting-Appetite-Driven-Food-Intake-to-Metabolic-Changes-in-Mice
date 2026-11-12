# Standard library
import os
import tkinter as tk

# Third-party packages
import sund
import numpy as np
import pandas as pd

# Import functions stored in other .py files
from Functions.functions import Data_formating, load_locked_scan_results
from Functions.plots import plot_ExenatidePK_uncertainty, plot_locked_scan, plot_PD_uncertainty, plot_locked_scan_PD, plot_AppetiteBW_uncertainty, plot_MackAppetite_uncertainty, plot_MackWeight_uncertainty, plot_He_uncertainty, plot_SemaglutidePK_uncertainty, plot_locked_scan_SemaglutidePK, plot_Gabery_uncertainty, plot_AZstudy_uncertainty, plot_DigitalPopulations, plot_PhyslicalActivityDependency
from Functions.optimise import Optimise_ExenatidePKstudy_PSO, Optimise_ExenatidePKstudy_PSO_locked_scan, Optimise_PDstudy_PSO, Optimise_PDstudy_PSO_locked_scan, Optimise_AppetiteBW__Parallel, Optimise_SemaglutidePKstudy_PSO, Optimise_SemaglutidePKstudy_PSO_locked_scan

# Needed to run parallel optimisations
def AppetiteBW_optimisation():
    data = Data_formating()[0]
    AcceptedAppetiteParams, AcceptedHallBWParams, AcceptedHeParams, best_cost = Optimise_AppetiteBW__Parallel(data)
    return AcceptedAppetiteParams, AcceptedHallBWParams, AcceptedHeParams, best_cost
def ProfileLikelihoodParallell_PD():
    data = Data_formating()[0]
    Optimise_PDstudy_PSO_locked_scan(Model_PK, model_GI, data, X0,OptimisedExenatidePKparams)
    
# Needed for the UI to select which figure to plot
def plot_selection_ui():
    root = tk.Tk()
    root.title("Select figure to plot")
    root.geometry("800x800")

    selection = tk.StringVar(value="__NONE__")
    fastplot = tk.BooleanVar(value=True)

    options = [
        "Plot Guo & Hall 2011 (Figs. 2B, 3B, and 4)",
        "Plot short-term study of Mack et al. 2006 (Figs. 2C and 5B)",
        "Plot long-term study of Mack et al. 2006 (Fig. 5A, S3A)",
        "Plot He et al. 2020 (Fig. 3C)",
        "Plot new semaglutide study (Fig. 6A, S3B)",
        "Plot Gabery et al. 2020 (Fig. 6B, S3C)",
        "Plot mice interventions (Fig. 7)",
        "Plot supplementary figures"
    ]

    tk.Label(root,text="Select figure to plot",font=("Arial",12,"bold")).pack(pady=(10,5))

    for option in options:
        tk.Radiobutton(root,text=option,variable=selection,value=option,anchor="w",justify="left").pack(fill="x",padx=10,pady=2)

    tk.Label(root,text="Plot mode",font=("Arial",12,"bold")).pack(pady=(15,0))
    tk.Radiobutton(root,text="Rapid plot",variable=fastplot,value=True).pack(anchor="w",padx=20)
    tk.Radiobutton(root,text="A more complete plot",variable=fastplot,value=False).pack(anchor="w",padx=20)

    def run_selected():

        choice = selection.get()
        FastPlot = fastplot.get()

        if choice == "__NONE__":
            return
        root.destroy()

        if choice == "Plot Guo & Hall 2011 (Figs. 2B, 3B, and 4)":
            plot_AppetiteBW_uncertainty(Model_Appetite, model_BW, Model_PK, model_GI, data, OptimisedAppetiteparams, OptimisedBWparams, OptimisedExenatidePKparams, OptimisedHeparams, FastPlot)

        elif choice == "Plot short-term study of Mack et al. 2006 (Figs. 2C and 5B)":
            plot_MackAppetite_uncertainty(Model_Appetite, Model_PK, data, OptimisedAppetiteparams, OptimisedExenatidePKparams, FastPlot)

        elif choice == "Plot long-term study of Mack et al. 2006 (Fig. 5A, S3A)":
            plot_MackWeight_uncertainty(Model_Appetite, Model_PK, model_GI, model_BW, data, OptimisedAppetiteparams, OptimisedBWparams, OptimisedPDparams, OptimisedExenatidePKparams, FastPlot)

        elif choice == "Plot He et al. 2020 (Fig. 3C)":
            plot_He_uncertainty(Model_Appetite, model_GI, model_BW, data, OptimisedAppetiteparams, OptimisedHeparams, OptimisedBWparams, FastPlot)

        elif choice == "Plot new semaglutide study (Fig. 6A, S3B)":
            plot_AZstudy_uncertainty(Model_Appetite, model_BW, Model_PK, model_GI, data, OptimisedAppetiteparams, OptimisedBWparams, OptimisedSemaglutidePKparams, OptimisedHeparams, FastPlot)

        elif choice == "Plot Gabery et al. 2020 (Fig. 6B, S3C)":
            plot_Gabery_uncertainty(Model_Appetite, model_BW, Model_PK, data, OptimisedAppetiteparams, OptimisedBWparams, OptimisedSemaglutidePKparams, FastPlot)

        elif choice == "Plot mice interventions (Fig. 7)":
            plot_DigitalPopulations(Model_Appetite, model_BW, Model_PK, model_GI, data, OptimisedAppetiteparams, OptimisedBWparams, OptimisedSemaglutidePKparams, OptimisedHeparams, FastPlot)

        elif choice == "Plot supplementary figures":
            supplementary_window()

    tk.Button(root,text="Plot",command=run_selected,font=("Arial",10,"bold")).pack(pady=15)

    root.mainloop()
def supplementary_window():

    root = tk.Tk()
    root.title("Supplementary figures")
    root.geometry("800x800")

    selection = tk.StringVar(value="__NONE__")

    options = [
        "Plot Gao and Jusko pharmacokinetics (Fig. S1B)",
        "Plot semaglutide pharmacokinetics (Fig. S1C)",
        "Plot Gao and Jusko pharmacodynamics (Fig. S2)",
        "Plot physical activity cohort analysis (Fig. S4)",
    ]

    for option in options:
        tk.Radiobutton(root,text=option,variable=selection,value=option,anchor="w",justify="left").pack(fill="x",padx=10,pady=3)

    def run_selected():

        choice = selection.get()
        if choice == "__NONE__":
            return
        root.destroy()

        if choice == "Plot Gao and Jusko pharmacokinetics (Fig. S1B)":
            plot_ExenatidePK_uncertainty(Model_PK, data, OptimisedExenatidePKparams)

        elif choice == "Plot semaglutide pharmacokinetics (Fig. S1C)":
            plot_SemaglutidePK_uncertainty(Model_PK, data, OptimisedSemaglutidePKparams)

        elif choice == "Plot Gao and Jusko pharmacodynamics (Fig. S2)":
            plot_PD_uncertainty(Model_PK, model_GI, data, OptimisedPDparams, OptimisedExenatidePKparams)

        elif choice == "Plot physical activity cohort analysis (Fig. S4)":
            plot_PhyslicalActivityDependency(OptimisedAppetiteparams, OptimisedBWparams, OptimisedSemaglutidePKparams, OptimisedHeparams)

    tk.Button(root,text="Plot",command=run_selected,font=("Arial",10,"bold")).pack(pady=15)

    root.mainloop()

# Load parameter vectors stored in /Parameters
def load_parameter_file(filename):
    path = os.path.join(os.path.dirname(__file__), "Parameters", filename)
    return pd.read_csv(path, header=None)

if __name__ == "__main__":

    # --------------------- Load data and models --------------------- ##
    [data] = Data_formating()

    for model_name in ["Model_GI","Model_PK","Model_BW","Model_Appetite"]:
        sund.install_model(f"{model_name}.txt")
    model_GI = sund.load_model("Model_GI")
    Model_PK = sund.load_model("Model_PK")
    model_BW = sund.load_model("Model_BW")
    Model_Appetite = sund.load_model("Model_Appetite")

    OptimisedSemaglutidePKparams = load_parameter_file("FullParams_SemaglutidePK_PSO.csv")
    OptimisedExenatidePKparams   = load_parameter_file("FullParams_ExenatidePK_PSO.csv")
    OptimisedPDparams            = load_parameter_file("Fullparams_PD.csv")
    OptimisedAppetiteparams      = load_parameter_file("Fullparams_Appetite.csv")
    OptimisedBWparams            = load_parameter_file("Fullparams_BW.csv")
    OptimisedHeparams            = load_parameter_file("Fullparams_He.csv")

    ## --------------  Extra plots and functions not part of the UI --------------------- ##
    
    ## ---  Pharmacokinetics  --- ##
    X0 = OptimisedExenatidePKparams.to_numpy(dtype=float)[0, :-1]
    #Optimise_ExenatidePKstudy_PSO(Model_PK, data, X0) # -- Optimise
    #Optimise_SemaglutidePKstudy_PSO(Model_PK, data)   # -- Optimise
    #Optimise_ExenatidePKstudy_PSO_locked_scan(Model_PK, data)   # -- Optimise 
    #Optimise_SemaglutidePKstudy_PSO_locked_scan(Model_PK, data) # -- Optimise 
    #results_dict = load_locked_scan_results("LockedScan_ExenatidePK.json") ; plot_locked_scan(results_dict,data)
    #results_dict = load_locked_scan_results("LockedScan_SemaglutidePK.json"); plot_locked_scan_SemaglutidePK(results_dict, data)

    ## ---  Glucose-Insulin-Exenatide --- ##
    X0 = OptimisedPDparams.to_numpy(dtype=float)[0, :-1]
    #Optimise_PDstudy_PSO(Model_PK, model_GI, data,X0,OptimisedExenatidePKparams)      # -- Optimise
    # ProfileLikelihoodParallell_PD()                                                  # -- Optimise
    #results_dict = load_locked_scan_results("LockedScan_PD.json"); plot_locked_scan_PD(results_dict,data)

    ## --- Appetite and BW modules --- ##
    #AppetiteBW_optimisation()                             # -- Optimise

    plot_selection_ui()