# Python built-in modules import
import json
import os
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

# Python package imports (these needs to be installed)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
from joblib import Parallel, delayed

from Functions.simulate import Simulate_PKstudy, Simulate_PDstudy, Simulate_HallAppetiteStudy, Simulate_MackStudy1, Simulate_MackStudy2, Uncertainty_MackStudy1, Simulate_HEStudy, Simulate_SemaglutidePKstudy, Simulate_GaberyStudy, Simulate_AZstudy, Simulate_DigitalPopulations
from Functions.cost import Costfunction_AppetiteBW, Costfunction_SemaglutidePKstudy
from Functions.functions import average_meals_per_day_and_meal_size, daily_energy_intake_from_ei_ac


# -------------------- Plot PK -------------------- #

# Exenatide
def plot_ExenatidePK_uncertainty(model, data, OptimisedExenatidePKparams):

    OptimisedExenatidePKparams = OptimisedExenatidePKparams.to_numpy(dtype=float)

    n_select = 1000
    idx = np.linspace(0, OptimisedExenatidePKparams.shape[0] - 1, n_select, dtype=int)
    OptimisedExenatidePKparams = OptimisedExenatidePKparams[idx]

    first = True

    for param_vector in OptimisedExenatidePKparams:
        param_vector = np.asarray(param_vector[:-1], dtype=float).flatten()
        try:
            sims = Simulate_PKstudy(model, param_vector)
        except Exception:
            continue
        IV_low_new       = sims['IV_low'].feature_values[:, sims['IV_low'].feature_names.index('Plasma_Exenatide')]
        IV_mid_new       = sims['IV_mid'].feature_values[:, sims['IV_mid'].feature_names.index('Plasma_Exenatide')]
        IV_high_new      = sims['IV_high'].feature_values[:, sims['IV_high'].feature_names.index('Plasma_Exenatide')]
        IV_inf_low_new   = sims['IV_infusion_low'].feature_values[:, sims['IV_infusion_low'].feature_names.index('Plasma_Exenatide')]
        IV_inf_mid_new   = sims['IV_infusion_mid'].feature_values[:, sims['IV_infusion_mid'].feature_names.index('Plasma_Exenatide')]
        IV_inf_high_new  = sims['IV_infusion_high'].feature_values[:, sims['IV_infusion_high'].feature_names.index('Plasma_Exenatide')]
        SC_low_new       = sims['SC_low'].feature_values[:, sims['SC_low'].feature_names.index('Plasma_Exenatide')]
        SC_mid_new       = sims['SC_mid'].feature_values[:, sims['SC_mid'].feature_names.index('Plasma_Exenatide')]
        SC_high_new      = sims['SC_high'].feature_values[:, sims['SC_high'].feature_names.index('Plasma_Exenatide')]

        if first:
            best_sims = sims
            # IV bolus
            IV_low_min, IV_low_max = IV_low_new.copy(), IV_low_new.copy()
            IV_mid_min, IV_mid_max = IV_mid_new.copy(), IV_mid_new.copy()
            IV_high_min, IV_high_max = IV_high_new.copy(), IV_high_new.copy()
            # IV infusion
            IV_inf_low_min, IV_inf_low_max = IV_inf_low_new.copy(), IV_inf_low_new.copy()
            IV_inf_mid_min, IV_inf_mid_max = IV_inf_mid_new.copy(), IV_inf_mid_new.copy()
            IV_inf_high_min, IV_inf_high_max = IV_inf_high_new.copy(), IV_inf_high_new.copy()
            # SC
            SC_low_min, SC_low_max = SC_low_new.copy(), SC_low_new.copy()
            SC_mid_min, SC_mid_max = SC_mid_new.copy(), SC_mid_new.copy()
            SC_high_min, SC_high_max = SC_high_new.copy(), SC_high_new.copy()
            # time
            time_IV     = sims['IV_low'].time_vector * 24 # unit convertion from days to hours
            time_IV_inf = sims['IV_infusion_low'].time_vector * 24
            time_SC     = sims['SC_low'].time_vector * 24
            first = False
        else:
            # IV bolus
            IV_low_min = np.minimum(IV_low_min, IV_low_new)
            IV_low_max = np.maximum(IV_low_max, IV_low_new)
            IV_mid_min = np.minimum(IV_mid_min, IV_mid_new)
            IV_mid_max = np.maximum(IV_mid_max, IV_mid_new)
            IV_high_min = np.minimum(IV_high_min, IV_high_new)
            IV_high_max = np.maximum(IV_high_max, IV_high_new)
            # IV infusion
            IV_inf_low_min = np.minimum(IV_inf_low_min, IV_inf_low_new)
            IV_inf_low_max = np.maximum(IV_inf_low_max, IV_inf_low_new)
            IV_inf_mid_min = np.minimum(IV_inf_mid_min, IV_inf_mid_new)
            IV_inf_mid_max = np.maximum(IV_inf_mid_max, IV_inf_mid_new)
            IV_inf_high_min = np.minimum(IV_inf_high_min, IV_inf_high_new)
            IV_inf_high_max = np.maximum(IV_inf_high_max, IV_inf_high_new)
            # SC
            SC_low_min = np.minimum(SC_low_min, SC_low_new)
            SC_low_max = np.maximum(SC_low_max, SC_low_new)
            SC_mid_min = np.minimum(SC_mid_min, SC_mid_new)
            SC_mid_max = np.maximum(SC_mid_max, SC_mid_new)
            SC_high_min = np.minimum(SC_high_min, SC_high_new)
            SC_high_max = np.maximum(SC_high_max, SC_high_new)


    y_axis_marker = 1e-5  # bottom for log scale
    colors = {
        "low": "teal",
        "mid": "darkorange",
        "high": "firebrick"
    }
    # ======================
    # Plot 1
    # ======================
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    # IV_BOLUS
    axes[0,0].fill_between(time_IV, IV_low_min, IV_low_max, color=colors["low"], alpha=0.2, edgecolor="none")
    axes[0,0].plot(time_IV, best_sims['IV_low'].feature_values[:, best_sims['IV_low'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["low"], label='0.5 nmol')
    axes[0,0].fill_between(time_IV, IV_mid_min, IV_mid_max, color=colors["mid"], alpha=0.2, edgecolor="none")
    axes[0,0].plot(time_IV, best_sims['IV_mid'].feature_values[:, best_sims['IV_mid'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["mid"], label='5 nmol')
    axes[0,0].fill_between(time_IV, IV_high_min, IV_high_max, color=colors["high"], alpha=0.2, edgecolor="none")
    axes[0,0].plot(time_IV, best_sims['IV_high'].feature_values[:, best_sims['IV_high'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["high"], label='50 nmol')
    axes[0,0].errorbar(np.array(data['RatPK']['timeExenatideIV'], dtype=float)/60, np.array(data['RatPK']['ExenatideIV_50nM'], dtype=float)/1000, yerr=np.array(data['RatPK']['ExenatideIV_50nMSEM'], dtype=float)/1000, fmt='none', color=colors["high"], capsize=8, elinewidth=2, capthick=2)
    axes[0,0].errorbar(np.array(data['RatPK']['timeExenatideIV'], dtype=float)/60, np.array(data['RatPK']['ExenatideIV_5nM'], dtype=float)/1000, yerr=np.array(data['RatPK']['ExenatideIV_5nMSEM'], dtype=float)/1000, fmt='none', color=colors["mid"], capsize=8, elinewidth=2, capthick=2)
    axes[0,0].errorbar(np.array(data['RatPK']['timeExenatideIV'], dtype=float)/60, np.array(data['RatPK']['ExenatideIV_0p5nM'], dtype=float)/1000, yerr=np.array(data['RatPK']['ExenatideIV_0p5nMSEM'], dtype=float)/1000, fmt='none', color=colors["low"], capsize=8, elinewidth=2, capthick=2)
    axes[0,0].plot([0,0.25],[y_axis_marker,y_axis_marker], color='black', lw=10, solid_capstyle='butt')
    # IV_INFUSION
    axes[0,1].fill_between(time_IV_inf, IV_inf_low_min, IV_inf_low_max, color=colors["low"], alpha=0.2, edgecolor="none")
    axes[0,1].plot(time_IV_inf, best_sims['IV_infusion_low'].feature_values[:, best_sims['IV_infusion_low'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["low"], label='0.5 nmol/h')
    axes[0,1].fill_between(time_IV_inf, IV_inf_mid_min, IV_inf_mid_max, color=colors["mid"], alpha=0.2, edgecolor="none")
    axes[0,1].plot(time_IV_inf, best_sims['IV_infusion_mid'].feature_values[:, best_sims['IV_infusion_mid'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["mid"], label='5 nmol/h')
    axes[0,1].fill_between(time_IV_inf, IV_inf_high_min, IV_inf_high_max, color=colors["high"], alpha=0.2, edgecolor="none")
    axes[0,1].plot(time_IV_inf, best_sims['IV_infusion_high'].feature_values[:, best_sims['IV_infusion_high'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["high"], label='50 nmol/h')
    axes[0,1].errorbar(np.array(data['RatPK']['timeExenatideIVtransfusion'], dtype=float) / 60,
                    np.array(data['RatPK']['ExenatideIVinfusion_50nM'], dtype=float) / 1000,
                    yerr=np.array(data['RatPK']['ExenatideIVinfusion_50nMSEM'], dtype=float) / 1000,
                    fmt='none', color=colors["high"], capsize=8, elinewidth=2, capthick=2)
    axes[0,1].errorbar(np.array(data['RatPK']['timeExenatideIVtransfusion'][0:14], dtype=float) / 60,
                np.array(data['RatPK']['ExenatideIVinfusion_5nM'][0:14], dtype=float) / 1000,
                yerr=np.array(data['RatPK']['ExenatideIVinfusion_5nMSEM'][0:14], dtype=float) / 1000,
                fmt='none', color=colors["mid"], capsize=8, elinewidth=2, capthick=2)
    axes[0,1].errorbar(np.array(data['RatPK']['timeExenatideIVtransfusion'][14], dtype=float) / 60,
            np.array(data['RatPK']['ExenatideIVinfusion_5nM'][14], dtype=float)*1.1 / 1000,
            yerr=np.array(data['RatPK']['ExenatideIVinfusion_5nMSEM'][14]*0.7, dtype=float) / 1000,
            fmt='none', color=colors["mid"], capsize=8, elinewidth=2, capthick=2)
    axes[0,1].errorbar(np.array(data['RatPK']['timeExenatideIVtransfusion'][15:17], dtype=float) / 60,
            np.array(data['RatPK']['ExenatideIVinfusion_5nM'][15:17], dtype=float) / 1000,
            yerr=np.array(data['RatPK']['ExenatideIVinfusion_5nMSEM'][15:17], dtype=float) / 1000,
            fmt='none', color=colors["mid"], capsize=8, elinewidth=2, capthick=2)
    axes[0,1].errorbar(np.array(data['RatPK']['timeExenatideIVtransfusion'][0:13], dtype=float) / 60,
                    np.array(data['RatPK']['ExenatideIVinfusion_0p5nM'][0:13], dtype=float) / 1000,
                    yerr=np.array(data['RatPK']['ExenatideIVinfusion_0p5nMSEM'][0:13], dtype=float) / 1000,
                    fmt='none', color=colors["low"], capsize=8, elinewidth=2, capthick=2)
    axes[0,1].errorbar(np.array(data['RatPK']['timeExenatideIVtransfusion'][13], dtype=float) / 60,
                np.array(data['RatPK']['ExenatideIVinfusion_0p5nM'][13]*1.1, dtype=float) / 1000,
                yerr=np.array(data['RatPK']['ExenatideIVinfusion_0p5nMSEM'][13]*0.75, dtype=float) / 1000,
                fmt='none', color=colors["low"], capsize=8, elinewidth=2, capthick=2)
    axes[0,1].plot([0,3],[y_axis_marker,y_axis_marker], color='black', lw=10, solid_capstyle='butt')
    # SC
    axes[0,2].fill_between(time_SC, SC_low_min, SC_low_max, color=colors["low"], alpha=0.2, edgecolor="none")
    axes[0,2].plot(time_SC, best_sims['SC_low'].feature_values[:, best_sims['SC_low'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["low"], label='0.5 nmol')
    axes[0,2].fill_between(time_SC, SC_mid_min, SC_mid_max, color=colors["mid"], alpha=0.2, edgecolor="none")
    axes[0,2].plot(time_SC, best_sims['SC_mid'].feature_values[:, best_sims['SC_mid'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["mid"], label='5 nmol')
    axes[0,2].fill_between(time_SC, SC_high_min, SC_high_max, color=colors["high"], alpha=0.2, edgecolor="none")
    axes[0,2].plot(time_SC, best_sims['SC_high'].feature_values[:, best_sims['SC_high'].feature_names.index('Plasma_Exenatide')], lw=2, color=colors["high"], label='50 nmol')
    axes[0,2].errorbar(np.array(data['RatPK']['Exenatidestimesd'], dtype=float)/60, np.array(data['RatPK']['Exenatidesd_50nM'], dtype=float)/1000, yerr=np.array(data['RatPK']['Exenatidesd_50nMSEM'], dtype=float)/1000, fmt='none', color=colors["high"], capsize=8, elinewidth=2, capthick=2)
    axes[0,2].errorbar(np.array(data['RatPK']['Exenatidestimesd'], dtype=float)/60, np.array(data['RatPK']['Exenatidesd_5nM'], dtype=float)/1000, yerr=np.array(data['RatPK']['Exenatidesd_5nMSEM'], dtype=float)/1000, fmt='none', color=colors["mid"], capsize=8, elinewidth=2, capthick=2)
    axes[0,2].errorbar(np.array(data['RatPK']['Exenatidestimesd'], dtype=float)/60, np.array(data['RatPK']['Exenatidesd_0p5nM'], dtype=float)/1000, yerr=np.array(data['RatPK']['Exenatidesd_0p5nMSEM'], dtype=float)/1000, fmt='none', color=colors["low"], capsize=8, elinewidth=2, capthick=2)
    axes[0,2].plot([0,0.25],[y_axis_marker,y_axis_marker], color='black', lw=10, solid_capstyle='butt')
    # FORMATTING
    axes_list = [axes[0,0], axes[0,1], axes[0,2]]
    titles = ['IV bolus', 'IV infusion', 'SC']
    legend_locs = ['lower left', 'lower left', 'lower right']
    for i in range(3):
        axes_list[i].set_yscale('log')
        axes_list[i].set_yticks([1e-5,1e-3,1e-1,1e1,1e3])
        axes_list[i].set_xlim(0,7)
        axes_list[i].set_ylim(1e-5,1e3)
        axes_list[i].set_xlabel('Time (h)', fontsize=18)
        axes_list[i].set_ylabel('Plasma exenatide\n(nmol/L)', fontsize=18)
        #axes_list[i].set_title(titles[i], fontsize=18)
        axes_list[i].tick_params(labelsize=18)
        axes_list[i].legend(loc=legend_locs[i], framealpha=1, fontsize=14, ncol=1)
        axes_list[i].grid(True)
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    plt.show()
def plot_locked_scan(results_dict, data):

    # Degrees of freedom and chi2 limit
    dgf = len(data['RatPK']['all_time'])
    chi2_limit = chi2.ppf(0.95, dgf)

    runs = results_dict["runs"]

    # Unique parameter indices
    param_indices = sorted(list(set([r["locked_parameter_index"] for r in runs])))
    n_params = len(param_indices)

    # Grid
    ncols = 4
    nrows = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.8 * nrows))
    axes = axes.flatten()

    # Parameter descriptions (PK)
    param_titles = [
        "SC absorption (Kamax)",
        "SC absorption (Ka50)",
        "Inter-compartmental clearance",
        "Central volume",
        "Peripheral volume",
        "Clearance (CLmax)",
        "Clearance (CL50)",
    ]

    # Units after conversion to days
    param_units = [
        "1/day",      # Kamax
        "nmol",       # Ka50
        "1/day",      # Q
        "mL",          # Vc
        "mL",          # Vp
        "L/day",      # CLmax
        "nmol/L"      # CL50
    ]

    # Convert all rate parameters from minutes to days
    parameter_conversion = [
        60 * 24,  # Kamax: 1/min -> 1/day
        1,        # Ka50
        60 * 24,  # Q: 1/min -> 1/day
        1000,     # Vc
        1000,     # Vp
        60 * 24,  # CLmax: L/min -> L/day
        1         # CL50
    ]

    # Find max cost
    all_costs = np.array([r["best_cost"] for r in runs], dtype=float)
    all_costs = all_costs[np.isfinite(all_costs) & (all_costs > 0)]
    global_ymax = max(np.max(all_costs), chi2_limit)

    for i, param_idx in enumerate(param_indices):

        ax = axes[i]
        param_runs = [r for r in runs if r["locked_parameter_index"] == param_idx]
        locked_values = np.array([r["locked_parameter_value"] for r in param_runs], dtype=float)
        costs = np.array([r["best_cost"] for r in param_runs], dtype=float)
        valid_mask = np.isfinite(locked_values) & np.isfinite(costs) & (locked_values > 0) & (costs > 0)
        locked_values = locked_values[valid_mask]
        costs = costs[valid_mask]
        sort_idx = np.argsort(locked_values)
        locked_values = locked_values[sort_idx]
        costs = costs[sort_idx]
        locked_values_plot = locked_values * parameter_conversion[param_idx]

        # Plot
        ax.plot(locked_values_plot, costs, lw=2, color="blue")
        ax.scatter(locked_values_plot, costs, color="black", s=20)
        ax.axhline(y=chi2_limit, linestyle="--", color="black", lw=2)
        # Scales
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Title
        ax.set_title(param_titles[param_idx], fontsize=16)
        # X label with units
        ax.set_xlabel(f"Parameter value ({param_units[param_idx]})", fontsize=16)
        ax.set_ylabel("Cost\n(Disagreement with data)", fontsize=16)
        ax.tick_params(axis='both', which='both', labelsize=16)
        plt.setp(ax.get_xticklabels(), fontsize=16)
        plt.setp(ax.get_yticklabels(), fontsize=16)
        ax.set_yticks([100, 1000, 10000,100000])
        ax.set_ylim(89.9, 100000)
        ax.grid(True)
    # Hide unused axes
    for j in range(n_params, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Profile Likelihood of PK-model", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.45, hspace=0.65)

    plt.show()

# Semaglutide
def plot_SemaglutidePK_uncertainty(model, data, OptimisedSemaglutidePKparams):

    OptimisedSemaglutidePKparams = OptimisedSemaglutidePKparams.to_numpy(dtype=float)

    n_select = 1000
    idx = np.linspace(0, OptimisedSemaglutidePKparams.shape[0] - 1, n_select, dtype=int)
    OptimisedSemaglutidePKparams = OptimisedSemaglutidePKparams[idx]

    first = True
    for param_vector in OptimisedSemaglutidePKparams:
        param_vector = np.asarray(param_vector[:-1], dtype=float)
        try:
            sims = Simulate_SemaglutidePKstudy(model, data, param_vector, TimeRes_prestudy=60, TimeRes_exposure=2)
        except Exception:
            continue
        idx_C1 = sims["Sema2"].feature_names.index("Plasma_Semaglutide")
        Sema0p3_new = sims["Sema0p3"].feature_values[:, idx_C1]
        Sema2_new = sims["Sema2"].feature_values[:, idx_C1]
        Sema4_new = sims["Sema4_Q2D"].feature_values[:, idx_C1]
        Sema10_new = sims["Sema10"].feature_values[:, idx_C1]
        Prestudy_new = sims["Sema10_prestudy"].feature_values[:, idx_C1]
        if first:
            best_sims = sims
            Sema0p3_min, Sema0p3_max = Sema0p3_new.copy(), Sema0p3_new.copy()
            Sema2_min, Sema2_max = Sema2_new.copy(), Sema2_new.copy()
            Sema4_min, Sema4_max = Sema4_new.copy(), Sema4_new.copy()
            Sema10_min, Sema10_max = Sema10_new.copy(), Sema10_new.copy()
            Prestudy_min, Prestudy_max = Prestudy_new.copy(), Prestudy_new.copy()
            first = False
        else:
            Sema0p3_min = np.minimum(Sema0p3_min, Sema0p3_new)
            Sema0p3_max = np.maximum(Sema0p3_max, Sema0p3_new)
            Sema2_min = np.minimum(Sema2_min, Sema2_new)
            Sema2_max = np.maximum(Sema2_max, Sema2_new)
            Sema4_min = np.minimum(Sema4_min, Sema4_new)
            Sema4_max = np.maximum(Sema4_max, Sema4_new)
            Sema10_min = np.minimum(Sema10_min, Sema10_new)
            Sema10_max = np.maximum(Sema10_max, Sema10_new)
            Prestudy_min = np.minimum(Prestudy_min, Prestudy_new)
            Prestudy_max = np.maximum(Prestudy_max, Prestudy_new)

    colors = {
        "Vehicle": "blue",
        "Sema0p3": "teal",
        "Sema2": "darkorange",
        "Sema4_Q2D": "goldenrod",
        "Sema10": "firebrick"
    }

    # ======================
    # Plots
    # ======================
    fig = plt.figure(figsize=(20, 5))
    gs = fig.add_gridspec(1, 3)
    ax_prestudy = fig.add_subplot(gs[0, 0])
    ax_longterm = fig.add_subplot(gs[0, 1:3])
    time_Sema0p3 = best_sims["Sema0p3"].time_vector
    time_Sema2 = best_sims["Sema2"].time_vector
    time_Sema4 = best_sims["Sema4_Q2D"].time_vector
    time_Sema10 = best_sims["Sema10"].time_vector
    time_Prestudy = best_sims["Sema10_prestudy"].time_vector
    idx_C1 = best_sims["Sema10_prestudy"].feature_names.index("Plasma_Semaglutide")
    # COLUMN 1: PRESTUDY PK
    ax_prestudy.fill_between(time_Prestudy * 24, np.maximum(Prestudy_min, 1e-6), np.maximum(Prestudy_max, 1e-6), color=colors["Sema10"], alpha=0.2, edgecolor="none")
    ax_prestudy.plot(time_Prestudy * 24, np.maximum(best_sims["Sema10_prestudy"].feature_values[:, idx_C1], 1e-6), color=colors["Sema10"], lw=2, label="Semaglutide 10 nmol/kg")
    ax_prestudy.plot(np.array(data["AZstudy"]["time_prestudy"][0:16], dtype=float) * 24, np.array(data["AZstudy"]["Sema10PK_prestudy"][0:16], dtype=float), "ko", markersize=5, label="Data")
    ax_prestudy.plot([0,1],[1e-2,1e-2], color='black', lw=18, solid_capstyle='butt')
    ax_prestudy.set_xlabel("Time (h)", fontsize=18)
    ax_prestudy.set_ylabel("Plasma semaglutide\n(nmol/L)", fontsize=18)
    ax_prestudy.set_title("Prestudy", fontsize=18)
    ax_prestudy.set_yscale("log")
    ax_prestudy.set_xlim(0, 30)
    ax_prestudy.set_ylim(1e-2, 1e2)
    ax_prestudy.set_xticks([0, 6, 12, 18, 24, 30])
    ax_prestudy.set_yticks([1e-2, 1, 100])
    ax_prestudy.tick_params(axis="both", which="major", labelsize=18)
    ax_prestudy.legend(loc="lower left", framealpha=1, fontsize=13)
    ax_prestudy.grid(True)
    # COLUMNS 2-3: LONG-TERM STUDY
    ax_longterm.fill_between(time_Sema0p3/7, np.maximum(Sema0p3_min, 1e-6), np.maximum(Sema0p3_max, 1e-6), color=colors["Sema0p3"], alpha=0.2, edgecolor="none")
    ax_longterm.plot(time_Sema0p3/7, np.maximum(best_sims["Sema0p3"].feature_values[:, idx_C1], 1e-6), color=colors["Sema0p3"], lw=2, label="Semaglutide 0.3 nmol/kg QD")
    ax_longterm.fill_between(time_Sema2/7, np.maximum(Sema2_min, 1e-6), np.maximum(Sema2_max, 1e-6), color=colors["Sema2"], alpha=0.2, edgecolor="none")
    ax_longterm.plot(time_Sema2/7, np.maximum(best_sims["Sema2"].feature_values[:, idx_C1], 1e-6), color=colors["Sema2"], lw=2, label="Semaglutide 2 nmol/kg QD")
    ax_longterm.fill_between(time_Sema4/7, np.maximum(Sema4_min, 1e-6), np.maximum(Sema4_max, 1e-6), color=colors["Sema4_Q2D"], alpha=0.2, edgecolor="none")
    ax_longterm.plot(time_Sema4/7, np.maximum(best_sims["Sema4_Q2D"].feature_values[:, idx_C1], 1e-6), color=colors["Sema4_Q2D"], lw=2, label="Semaglutide 4 nmol/kg Q2D")
    ax_longterm.fill_between(time_Sema10/7, np.maximum(Sema10_min, 1e-6), np.maximum(Sema10_max, 1e-6), color=colors["Sema10"], alpha=0.2, edgecolor="none")
    ax_longterm.plot(time_Sema10/7, np.maximum(best_sims["Sema10"].feature_values[:, idx_C1], 1e-6), color=colors["Sema10"], lw=2, label="Semaglutide 10 nmol/kg QD")
    ax_longterm.errorbar(np.array(data["AZstudy"]["timeExposure"][0:1], dtype=float)/7, np.array(data["AZstudy"]["Sema2Exposure"][0:1], dtype=float), yerr=np.array(data["AZstudy"]["Sema2Exposure_stdev"][0:1], dtype=float), fmt="none",color=colors["Sema2"], capsize=6, elinewidth=2, capthick=2)
    ax_longterm.errorbar(np.array(data["AZstudy"]["timeExposure"][0:1], dtype=float)/7, np.array(data["AZstudy"]["Sema4_Q2DExposure"][0:1], dtype=float), yerr=np.array(data["AZstudy"]["Sema4_Q2DExposure_stdev"][0:1], dtype=float), fmt="none", color=colors["Sema4_Q2D"], capsize=6, elinewidth=2, capthick=2)
    ax_longterm.errorbar(np.array(data["AZstudy"]["timeExposureLateEnd"][0:1], dtype=float)/7, np.array(data["AZstudy"]["Sema10Exposure"][0:1], dtype=float), yerr=np.array(data["AZstudy"]["Sema10Exposure_stdev"][0:1], dtype=float), fmt="none", color=colors["Sema10"], capsize=6, elinewidth=2, capthick=2)
    for day in range(22):
        ax_longterm.plot([day/7 - 0.1/7, day/7 + 0.1/7], [1e-2, 1e-2], color='black', lw=18, solid_capstyle='butt')
    ax_longterm.set_xlabel("Time (weeks)", fontsize=18)
    ax_longterm.set_ylabel("Plasma semaglutide\n(nmol/L)", fontsize=18)
    ax_longterm.set_title("Longterm study", fontsize=18)
    ax_longterm.set_yscale("log")
    ax_longterm.set_xlim(-0.1, 4.1)
    ax_longterm.set_ylim(1e-2, 1e2)
    ax_longterm.set_xticks([0, 1, 2, 3, 4])
    ax_longterm.set_yticks([1e-2, 1, 100])
    ax_longterm.tick_params(axis="both", which="major", labelsize=18)
    ax_longterm.legend(loc="lower right", framealpha=1, fontsize=12)
    ax_longterm.grid(True)
    fig.subplots_adjust(left=0.07,right=0.993,top=0.92,bottom=0.13,wspace=0.3,hspace=0.3)
    plt.show()
def plot_locked_scan_SemaglutidePK(results_dict, data):

    chi2_limit = float(results_dict["chi2_limit"])
    runs = results_dict["runs"]
    param_indices = sorted(list(set([r["locked_parameter_index"] for r in runs])))
    n_params = len(param_indices)
    ncols = 3
    nrows = 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8*nrows))
    axes = np.asarray(axes).flatten()

    param_titles = [
        "SC absorption",
        "Central volume",
        "Clearance"
    ]

    param_units = [
        "1/day",
        "mL",
        "L/day"
    ]

    parameter_conversion = [
        60*24,
        1000,
        60*24
    ]

    # Find max cost
    all_costs = np.array([r["best_cost"] for r in runs], dtype=float)
    all_costs = all_costs[np.isfinite(all_costs) & (all_costs > 0)]
    global_ymax = max(np.max(all_costs), chi2_limit)

    for i, param_idx in enumerate(param_indices):

        ax = axes[i]

        param_runs = [r for r in runs if r["locked_parameter_index"] == param_idx]
        locked_values = np.array([r["locked_parameter_value"] for r in param_runs], dtype=float)
        costs = np.array([r["best_cost"] for r in param_runs], dtype=float)
        valid_mask = np.isfinite(locked_values) & np.isfinite(costs) & (locked_values > 0) & (costs > 0)
        locked_values = locked_values[valid_mask]
        costs = costs[valid_mask]
        sort_idx = np.argsort(locked_values)
        locked_values = locked_values[sort_idx]
        costs = costs[sort_idx]
        locked_values_plot = locked_values*parameter_conversion[param_idx]

        ax.plot(locked_values_plot, costs, lw=2, color="blue")
        ax.scatter(locked_values_plot, costs, color="black", s=20)
        ax.axhline(y=chi2_limit, linestyle="--", color="black", lw=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(param_titles[param_idx], fontsize=16)
        ax.set_xlabel(f"Parameter value ({param_units[param_idx]})", fontsize=16)
        ax.set_ylabel("Cost\n(Disagreement with data)", fontsize=16)
        ax.tick_params(axis="both", which="both", labelsize=16)
        ax.grid(True)
        ax.set_yticks([1, 10, 100, 1000, 10000])
        ax.set_ylim(9.9, max(global_ymax * 1.2, 100))
    for j in range(n_params, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Profile likelihood of semaglutide PK model", fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.45, hspace=0.55)
    plt.show()

# -------------------- Plot PD -------------------- #
def plot_PD_uncertainty(model_PK, model_GI, data, OptimisedPDparams,OptimisedExenatidePKparams):

    OptimisedPDparams = OptimisedPDparams.to_numpy(dtype=float)
    OptimisedExenatidePKparams = OptimisedExenatidePKparams.to_numpy(dtype=float)

    n_select = 1000
    idx = np.linspace(0, OptimisedPDparams.shape[0] - 1, n_select, dtype=int)
    OptimisedPDparams = OptimisedPDparams[idx]
    idx = np.linspace(0, OptimisedExenatidePKparams.shape[0] - 1, n_select, dtype=int)
    OptimisedExenatidePKparams = OptimisedExenatidePKparams[idx]

    Best_PKParams               = np.asarray(OptimisedExenatidePKparams[0][:-1], dtype=float)
    Best_PDParams               = np.asarray(OptimisedPDparams[0][:-1], dtype=float)

    first = True
    LogSpacedExposure = np.logspace(-5, 3, 100)
    for param_vector in OptimisedPDparams:
        full_param = np.asarray(param_vector[:-1], dtype=float)
        try:
            sims = Simulate_PDstudy(model_PK, model_GI, full_param, Best_PKParams)
        except Exception:
            continue
        idx_G = sims["simulationIV_infusion_0"].feature_names.index("Plasma_Glucose")
        idx_I = sims["simulationIV_infusion_0"].feature_names.index("Plasma_Insulin")
        idx_C1 = sims["simulationIV_infusion_0"].feature_names.index("Plasma_Exenatide")
        time = sims["simulationIV_infusion_0"].time_vector * 1440 - 30 # change unit and move the glucose injection to t=0
        G_0_new = sims["simulationIV_infusion_0"].feature_values[:, idx_G]
        G_3_new = sims["simulationIV_infusion_3"].feature_values[:, idx_G]
        G_30_new = sims["simulationIV_infusion_30"].feature_values[:, idx_G]
        G_300_new = sims["simulationIV_infusion_300"].feature_values[:, idx_G]
        G_3000_new = sims["simulationIV_infusion_3000"].feature_values[:, idx_G]

        I_0_new = sims["simulationIV_infusion_0"].feature_values[:, idx_I]
        I_3_new = sims["simulationIV_infusion_3"].feature_values[:, idx_I]
        I_30_new = sims["simulationIV_infusion_30"].feature_values[:, idx_I]
        I_300_new = sims["simulationIV_infusion_300"].feature_values[:, idx_I]
        I_3000_new = sims["simulationIV_infusion_3000"].feature_values[:, idx_I]

        ExenatideInsulinMax = sims["simulationIV_infusion_300"].parameter_values[sims["simulationIV_infusion_300"].parameter_names.index("ExenatideInsulinMax")]
        ExenatideInsulin50 = sims["simulationIV_infusion_300"].parameter_values[sims["simulationIV_infusion_300"].parameter_names.index("ExenatideInsulin50")]
        ExenatideInsulin_hill = sims["simulationIV_infusion_300"].parameter_values[sims["simulationIV_infusion_300"].parameter_names.index("ExenatideInsulin_hill")]
        ExenatideInsulinDecayMax = sims["simulationIV_infusion_300"].parameter_values[sims["simulationIV_infusion_300"].parameter_names.index("ExenatideInsulinDecayMax")]
        ExenatideInsulinDecay50 = sims["simulationIV_infusion_300"].parameter_values[sims["simulationIV_infusion_300"].parameter_names.index("ExenatideInsulinDecay50")]
        Exenatide_hill = LogSpacedExposure**ExenatideInsulin_hill
        Exenatide_PD_new = 1 + ((ExenatideInsulinMax * Exenatide_hill / ((ExenatideInsulin50**ExenatideInsulin_hill) + Exenatide_hill)) - (Exenatide_hill * ExenatideInsulinDecayMax / (Exenatide_hill + (ExenatideInsulinDecay50**ExenatideInsulin_hill))))

        if first:
            best_sims = sims
            G_0_min = G_0_new.copy()
            G_0_max = G_0_new.copy()
            G_3_min = G_3_new.copy()
            G_3_max = G_3_new.copy()
            G_30_min = G_30_new.copy()
            G_30_max = G_30_new.copy()
            G_300_min = G_300_new.copy()
            G_300_max = G_300_new.copy()
            G_3000_min = G_3000_new.copy()
            G_3000_max = G_3000_new.copy()

            I_0_min = I_0_new.copy()
            I_0_max = I_0_new.copy()
            I_3_min = I_3_new.copy()
            I_3_max = I_3_new.copy()
            I_30_min = I_30_new.copy()
            I_30_max = I_30_new.copy()
            I_300_min = I_300_new.copy()
            I_300_max = I_300_new.copy()
            I_3000_min = I_3000_new.copy()
            I_3000_max = I_3000_new.copy()

            Exenatide_PD_min = Exenatide_PD_new.copy()
            Exenatide_PD_max = Exenatide_PD_new.copy()
            Exenatide_PD_best = Exenatide_PD_new.copy()

            first = False
        else:
            G_0_min = np.minimum(G_0_min, G_0_new)
            G_0_max = np.maximum(G_0_max, G_0_new)
            G_3_min = np.minimum(G_3_min, G_3_new)
            G_3_max = np.maximum(G_3_max, G_3_new)
            G_30_min = np.minimum(G_30_min, G_30_new)
            G_30_max = np.maximum(G_30_max, G_30_new)
            G_300_min = np.minimum(G_300_min, G_300_new)
            G_300_max = np.maximum(G_300_max, G_300_new)
            G_3000_min = np.minimum(G_3000_min, G_3000_new)
            G_3000_max = np.maximum(G_3000_max, G_3000_new)

            I_0_min = np.minimum(I_0_min, I_0_new)
            I_0_max = np.maximum(I_0_max, I_0_new)
            I_3_min = np.minimum(I_3_min, I_3_new)
            I_3_max = np.maximum(I_3_max, I_3_new)
            I_30_min = np.minimum(I_30_min, I_30_new)
            I_30_max = np.maximum(I_30_max, I_30_new)
            I_300_min = np.minimum(I_300_min, I_300_new)
            I_300_max = np.maximum(I_300_max, I_300_new)
            I_3000_min = np.minimum(I_3000_min, I_3000_new)
            I_3000_max = np.maximum(I_3000_max, I_3000_new)

            Exenatide_PD_min = np.minimum(Exenatide_PD_min, Exenatide_PD_new)
            Exenatide_PD_max = np.maximum(Exenatide_PD_max, Exenatide_PD_new)

    # PK-sim
    first = True
    for param_vector in OptimisedExenatidePKparams:
        full_param = np.asarray(param_vector[:-1], dtype=float)

        try:
            sims = Simulate_PDstudy(model_PK, model_GI, Best_PDParams, full_param)
        except Exception:
            continue
        time = sims["simulationIV_infusion_0"].time_vector * 1440 - 30 # change unit and move the glucose injection to t=0

        C1_0_new = sims["simulationIV_infusion_0"].feature_values[:, idx_C1]
        C1_3_new = sims["simulationIV_infusion_3"].feature_values[:, idx_C1]
        C1_30_new = sims["simulationIV_infusion_30"].feature_values[:, idx_C1]
        C1_300_new = sims["simulationIV_infusion_300"].feature_values[:, idx_C1]
        C1_3000_new = sims["simulationIV_infusion_3000"].feature_values[:, idx_C1]

        if first:
            C1_0_min = C1_0_new.copy()
            C1_0_max = C1_0_new.copy()
            C1_3_min = C1_3_new.copy()
            C1_3_max = C1_3_new.copy()
            C1_30_min = C1_30_new.copy()
            C1_30_max = C1_30_new.copy()
            C1_300_min = C1_300_new.copy()
            C1_300_max = C1_300_new.copy()
            C1_3000_min = C1_3000_new.copy()
            C1_3000_max = C1_3000_new.copy()

            first = False

        else:
            C1_0_min = np.minimum(C1_0_min, C1_0_new)
            C1_0_max = np.maximum(C1_0_max, C1_0_new)
            C1_3_min = np.minimum(C1_3_min, C1_3_new)
            C1_3_max = np.maximum(C1_3_max, C1_3_new)
            C1_30_min = np.minimum(C1_30_min, C1_30_new)
            C1_30_max = np.maximum(C1_30_max, C1_30_new)
            C1_300_min = np.minimum(C1_300_min, C1_300_new)
            C1_300_max = np.maximum(C1_300_max, C1_300_new)
            C1_3000_min = np.minimum(C1_3000_min, C1_3000_new)
            C1_3000_max = np.maximum(C1_3000_max, C1_3000_new)


    colors = {
        "Vehicle": "blue",
        "E3": "teal",
        "E30": "darkorange",
        "E300": "goldenrod",
        "E3000": "firebrick"
        }

    t_data = np.array(data["RatPD"]["Time"], dtype=float)

    #  ===== Plot =====
    fig, axes = plt.subplots(2, 5, figsize=(18, 6), sharex=True, sharey='row')
    # Glucose  
    axes[0,0].fill_between(time, G_0_min, G_0_max, color=colors["Vehicle"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sims["simulationIV_infusion_0"].feature_values[:, idx_G], color=colors["Vehicle"], lw=2)
    axes[0,0].errorbar(t_data, np.array(data["RatPD"]["glucoseMean_0"], dtype=float), yerr=np.array(data["RatPD"]["glucoseSEM_0"], dtype=float), color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[0,1].fill_between(time, G_3_min, G_3_max, color=colors["E3"], alpha=0.4, linewidth=0)
    axes[0,1].plot(time, best_sims["simulationIV_infusion_3"].feature_values[:, idx_G], color=colors["E3"], lw=2)
    axes[0,1].errorbar(t_data, np.array(data["RatPD"]["glucoseMean_3"], dtype=float), yerr=np.array(data["RatPD"]["glucoseSEM_3"], dtype=float), color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[0,2].fill_between(time, G_30_min, G_30_max, color=colors["E30"], alpha=0.4, linewidth=0)
    axes[0,2].plot(time, best_sims["simulationIV_infusion_30"].feature_values[:, idx_G], color=colors["E30"], lw=2)
    axes[0,2].errorbar(t_data, np.array(data["RatPD"]["glucoseMean_30"], dtype=float), yerr=np.array(data["RatPD"]["glucoseSEM_30"], dtype=float), color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[0,3].fill_between(time, G_300_min, G_300_max, color=colors["E300"], alpha=0.4, linewidth=0)
    axes[0,3].plot(time, best_sims["simulationIV_infusion_300"].feature_values[:, idx_G], color=colors["E300"], lw=2)
    axes[0,3].errorbar(t_data, np.array(data["RatPD"]["glucoseMean_300"], dtype=float), yerr=np.array(data["RatPD"]["glucoseSEM_300"], dtype=float), color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[0,4].fill_between(time, G_3000_min, G_3000_max, color=colors["E3000"], alpha=0.4, linewidth=0)
    axes[0,4].plot(time, best_sims["simulationIV_infusion_3000"].feature_values[:, idx_G], color=colors["E3000"], lw=2)
    axes[0,4].errorbar(t_data, np.array(data["RatPD"]["glucoseMean_3000"], dtype=float), yerr=np.array(data["RatPD"]["glucoseSEM_3000"], dtype=float), color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    #  Insulin 
    axes[1,0].fill_between(time, I_0_min/1000, I_0_max/1000, color=colors["Vehicle"], alpha=0.4, linewidth=0)
    axes[1,0].plot(time, best_sims["simulationIV_infusion_0"].feature_values[:, idx_I]/1000, color=colors["Vehicle"], lw=2)
    axes[1,0].errorbar(t_data, np.array(data["RatPD"]["insulinMean_0"], dtype=float)/1000, yerr=np.array(data["RatPD"]["insulinSEM_0"], dtype=float)/1000, color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[1,1].fill_between(time, I_3_min/1000, I_3_max/1000, color=colors["E3"], alpha=0.4, linewidth=0)
    axes[1,1].plot(time, best_sims["simulationIV_infusion_3"].feature_values[:, idx_I]/1000, color=colors["E3"], lw=2)
    axes[1,1].errorbar(t_data, np.array(data["RatPD"]["insulinMean_3"], dtype=float)/1000, yerr=np.array(data["RatPD"]["insulinSEM_3"], dtype=float)/1000, color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[1,2].fill_between(time, I_30_min/1000, I_30_max/1000, color=colors["E30"], alpha=0.4, linewidth=0)
    axes[1,2].plot(time, best_sims["simulationIV_infusion_30"].feature_values[:, idx_I]/1000, color=colors["E30"], lw=2)
    axes[1,2].errorbar(t_data, np.array(data["RatPD"]["insulinMean_30"], dtype=float)/1000, yerr=np.array(data["RatPD"]["insulinSEM_30"], dtype=float)/1000, color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[1,3].fill_between(time, I_300_min/1000, I_300_max/1000, color=colors["E300"], alpha=0.4, linewidth=0)
    axes[1,3].plot(time, best_sims["simulationIV_infusion_300"].feature_values[:, idx_I]/1000, color=colors["E300"], lw=2)
    axes[1,3].errorbar(t_data, np.array(data["RatPD"]["insulinMean_300"], dtype=float)/1000, yerr=np.array(data["RatPD"]["insulinSEM_300"], dtype=float)/1000, color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    axes[1,4].fill_between(time, I_3000_min/1000, I_3000_max/1000, color=colors["E3000"], alpha=0.4, linewidth=0)
    axes[1,4].plot(time, best_sims["simulationIV_infusion_3000"].feature_values[:, idx_I]/1000, color=colors["E3000"], lw=2)
    axes[1,4].errorbar(t_data, np.array(data["RatPD"]["insulinMean_3000"], dtype=float)/1000, yerr=np.array(data["RatPD"]["insulinSEM_3000"], dtype=float)/1000, color="black", fmt='none', capsize=8, elinewidth=2, capthick=2)
    #  Formatting 
    titles = ["Vehicle", "Exenatide infusion\n3 pmol/kg/min", "Exenatide infusion\n30 pmol/kg/min", "Exenatide infusion\n300 pmol/kg/min", "Exenatide infusion\n3000 pmol/kg/min"]
    for i in range(5):
        axes[0,i].set_title(titles[i], fontsize=18)
        for j in range(2):
            axes[j,i].set_xlim(-35, 95)
            axes[j,i].set_xticks(np.arange(-30, 91, 30))
            axes[j,i].grid(True)
            axes[j,i].tick_params(axis='both', which='major', labelsize=16)
            axes[j,i].tick_params(axis='both', which='minor', labelsize=16)
            plt.setp(axes[j,i].get_xticklabels(), fontsize=18)
            plt.setp(axes[j,i].get_yticklabels(), fontsize=18)
        axes[0,i].set_ylim(5, 40)
        axes[1,i].set_ylim(0, 4)
        axes[0,i].set_yticks(np.arange(5, 40, 10))
        axes[1,i].set_yticks(np.arange(0, 4, 1))
        axes[0,i].plot([-2, 2], [5, 5], color="black", linewidth=10)
        axes[1,i].plot([-2, 2], [0, 0], color="black", linewidth=10)
        axes[1,i].set_xlabel("Time (min)", fontsize=18)
    axes[0,0].set_ylabel("Plasma glucose\n(mmol/L)", fontsize=18)
    axes[1,0].set_ylabel("Plasma insulin\n(nmol/L)", fontsize=18)
    fig.subplots_adjust(left=0.07,right=0.993,top=0.925,bottom=0.065,wspace=0.06,hspace=0.06)
    plt.show()


    # ===== Fig. 2 =====
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Exposure
    axes[0,0].fill_between(time, C1_0_min, C1_0_max, color=colors["Vehicle"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sims["simulationIV_infusion_0"].feature_values[:, idx_C1], color=colors["Vehicle"], lw=2, label="Vehicle")
    axes[0,0].fill_between(time, C1_3_min, C1_3_max, color=colors["E3"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sims["simulationIV_infusion_3"].feature_values[:, idx_C1], color=colors["E3"], lw=2, label="3 (pmol/kg/min)")
    axes[0,0].fill_between(time, C1_30_min, C1_30_max, color=colors["E30"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sims["simulationIV_infusion_30"].feature_values[:, idx_C1], color=colors["E30"], lw=2, label="30 (pmol/kg/min)")
    axes[0,0].fill_between(time, C1_300_min, C1_300_max, color=colors["E300"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sims["simulationIV_infusion_300"].feature_values[:, idx_C1], color=colors["E300"], lw=2, label="300 (pmol/kg/min)")
    axes[0,0].fill_between(time, C1_3000_min, C1_3000_max, color=colors["E3000"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sims["simulationIV_infusion_3000"].feature_values[:, idx_C1], color=colors["E3000"], lw=2, label="3000 (pmol/kg/min)")
    axes[0,0].plot([-2, 2], [1e-3, 1e-3], color="black", linewidth=10)
    # PK-PD curve
    axes[0,1].fill_between(LogSpacedExposure, Exenatide_PD_min, Exenatide_PD_max, color="black", alpha=0.2, linewidth=0)
    axes[0,1].plot(LogSpacedExposure, Exenatide_PD_best, color="black", lw=2)
    # Formatting
    axes[0,0].set_yscale("log")
    axes[0,1].set_xscale("log")
    axes[0,0].set_xlim(-35, 95)
    axes[0,0].set_xticks(np.arange(-30, 91, 30))
    axes[0,0].set_yticks([1e-3, 1e-1, 1e1, 1e3])
    axes[0,0].set_ylim(1e-3, 1e3)
    axes[0,0].set_xlabel("Time (min)", fontsize=18)
    axes[0,0].set_ylabel("Plasma exenatide\n(nmol/L)", fontsize=18)
    axes[0,0].grid(True)
    axes[0,0].legend(fontsize=12, framealpha=1)
    axes[0,1].set_xlim(1e-3, 1e3)
    axes[0,1].set_xticks([1e-3, 1e-1, 1e1, 1e3])
    axes[0,1].set_yticks([1, 3, 6, 9])
    axes[0,1].set_ylim(0, np.max(Exenatide_PD_max)*1.2)
    axes[0,1].set_xlabel("Plasma exenatide (nmol/L)", fontsize=18)
    axes[0,1].set_ylabel("Stimulated insulin secretion\n(fold-increase)", fontsize=18)
    axes[0,1].grid(True)
    for i in range(2):
        for j in range(2):
            axes[i,j].tick_params(axis='both', which='major', labelsize=18)
            axes[i,j].tick_params(axis='both', which='minor', labelsize=18)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=18)
            plt.setp(axes[i,j].get_yticklabels(), fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(wspace=0.25, hspace=0.25)
    plt.show()
def plot_locked_scan_PD(results_dict, data):

    # Degrees of freedom and chi2 limit
    dgf = len(data['RatPD']['all_mean'])
    chi2_limit = chi2.ppf(0.95, dgf)

    runs = results_dict["runs"]

    # Unique parameter indices
    param_indices = sorted(list(set([r["locked_parameter_index"] for r in runs])))
    n_params = len(param_indices)

    # Grid split
    ncols = 4
    nrows_per_fig = 3
    params_per_fig = ncols * nrows_per_fig

    param_titles = [
        "Glucose exchange",        
        "Insulin production delay",           
        "EGP reduction delay",               
        "Brain glucose utilisation",    
        "SC glucose absorption",   
        "EGP strength to baseline",         
        "Glucose utilisation E50",         
        "Glucose utilisation insulin dep.",      
        "Basal glucose utilisation",            
        "Glucose basal set-point",                
        "Basal insulin production",           
        "Glucose-stimulated insulin prod.",   
        "Glucose-stimulated insulin threshold",       
        "Insulin clearance",       
        "Exenatide effect (Emax)",    
        "Exenatide effect (EC50)",          
        "Exenatide hill",          
        "Exenatide effect loss (Imax)",     
        "Exenatide effect loss (IC50)"     
    ]

    param_units = [
        "L/h",                                  # Q
        "1/h",                                  # k_IE
        "1/h",                                  # k_GE2
        "L/h",                                  # CL_G
        "1/h",                                  # k_ip
        "dimensionless",                        # GPRG
        "mmol/L",                               # U_50
        "fold-increase * (1/insulin [pmol/L])", # U_max_id
        "fold-increase",                        # U_max0
        "mmol/L",                               # G_SS
        "pmol/L",                               # I_SS
        "pM insulin * (1/glucose [mmol/L])",    # I_Gdep
        "mmol/L",                               # I_G0
        "L/h",                                  # CL_I
        "fold-increase",                        # ExenatideInsulinMax
        "nmol/L",                               # ExenatideInsulin50
        "dimensionless",                        # Hill
        "fold-decrease",                        # Decay max
        "nmol/L"                                # Decay50
    ]
    for fig_idx, param_subset in enumerate([param_indices[:params_per_fig], param_indices[params_per_fig:]]):

        fig, axes = plt.subplots(nrows_per_fig, ncols, figsize=(18, 4.5 * nrows_per_fig))
        axes = np.asarray(axes).flatten()

        for i, param_idx in enumerate(param_subset):

            ax = axes[i]
            param_runs = [r for r in runs if r["locked_parameter_index"] == param_idx]
            locked_values = np.array([r["locked_parameter_value"] for r in param_runs])
            costs = np.array([r["best_cost"] for r in param_runs])
            sort_idx = np.argsort(locked_values)
            locked_values = locked_values[sort_idx]
            costs = costs[sort_idx]
            costs_plot = np.minimum(costs, 1e5)
            ax.plot(locked_values, costs_plot, lw=2, color="blue")
            ax.scatter(locked_values, costs_plot, color="black", s=20)
            ax.axhline(y=chi2_limit, linestyle="--", color="black", lw=2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(param_titles[param_idx], fontsize=16)
            ax.set_xlabel(f"Parameter value ({param_units[param_idx]})", fontsize=16)
            ax.set_ylabel("Cost\n(Disagreement with data)", fontsize=16)
            ax.tick_params(axis="both", which="both", labelsize=16)
            plt.setp(ax.get_xticklabels(), fontsize=16)
            plt.setp(ax.get_yticklabels(), fontsize=16)
            ax.set_ylim(90, 1e4)
            ax.set_yticks([1e2, 1e3, 1e4])
            ax.grid(True)

        for j in range(len(param_subset), len(axes)):
            axes[j].axis("off")

        if fig_idx == 0:
            fig.suptitle("Profile Likelihood of PD model - parameters 1-12", fontsize=18)
        else:
            fig.suptitle("Profile Likelihood of PD model - parameters 13-19", fontsize=18)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.subplots_adjust(wspace=0.45, hspace=0.65)

        plt.show()

# ------------ Appetite moddeling ----------------
def plot_AppetiteBW_uncertainty(Model_Appetite,model_BW, model_PK, Model_GI, data, OptimisedAppetiteParams,OptimisedBWparams, OptimisedExenatidePKparams,OptimisedHeparams,FastPlot):

    OptimisedAppetiteParams     = OptimisedAppetiteParams.to_numpy(dtype=float)
    OptimisedBWparams           = OptimisedBWparams.to_numpy(dtype=float)
    OptimisedExenatidePKparams  = OptimisedExenatidePKparams.to_numpy(dtype=float)
    OptimisedHeparams           = OptimisedHeparams.to_numpy(dtype=float)
    Best_PKParams               = np.asarray(OptimisedExenatidePKparams[0][:-1], dtype=float)

    TimeRes = 96


    # Run this if you want to re-calculate cost for Hall et al., Mack et al., or He et al..  
    RecalculateCost = False # use True or False
    def _validate_param(VectorNumber):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        HeParams = np.asarray(OptimisedHeparams[VectorNumber][:-1], dtype=float)
        TypeOfData = "Validation" # use "Validation" or "Estimation"
        Study = "Mack"            # use "Hall", "Mack", "He"
        try:
            CostValidation, CostComponents = Costfunction_AppetiteBW(Model_Appetite, Model_GI, model_BW, model_PK, AppetiteParams, BWParams, HeParams, Best_PKParams, 0.02, data, TimeRes, TypeOfData, Study)
            return CostValidation
        except Exception:
            return np.inf
    if RecalculateCost:
        ValidationCosts = Parallel(n_jobs=16, prefer="threads", verbose=10)(delayed(_validate_param)(VectorNumber) for VectorNumber in range(len(OptimisedAppetiteParams)))
        BestVectorNumber = np.argmin(ValidationCosts)
        BestValidationCost = ValidationCosts[BestVectorNumber]
        print(f"Best validation cost: {BestValidationCost}")
        print(f"Best parameter vector index: {BestVectorNumber}")


    n_select = 200
    if FastPlot:
        n_select = 3
    idx = np.linspace(0, OptimisedAppetiteParams.shape[0] - 1, n_select, dtype=int)
    OptimisedAppetiteParams = OptimisedAppetiteParams[idx]
    idx = np.linspace(0, OptimisedBWparams.shape[0] - 1, n_select, dtype=int)
    OptimisedBWparams = OptimisedBWparams[idx]
    idx = np.linspace(0, OptimisedHeparams.shape[0] - 1, n_select, dtype=int)
    OptimisedHeparams = OptimisedHeparams[idx]


    # Appetite simulations  
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):

        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_HallAppetiteStudy(Model_Appetite,model_BW, data, AppetiteParams,BWParams, TimeRes, True)
        except Exception:
            continue

        idx_ei_ac        = sims["control"].feature_names.index("EI_AC")
        idx_appetite     = sims["control"].feature_names.index("Appetite")
        idx_eating       = sims["control"].feature_names.index("Eating")
        idx_bw           = sims["control"].feature_names.index("Body_weight")
        idx_fm           = sims["control"].feature_names.index("FM")
        idx_lambda       = sims["control"].feature_names.index("lambda")
        idx_lambda_delay = sims["control"].feature_names.index("lambda_delay")

        meals_control, meal_size_control, daily_ei_control, night_meals_control = average_meals_per_day_and_meal_size(sims["control"].feature_values[:, idx_ei_ac], sims["control"].feature_values[:, idx_eating], sims["control"].time_vector, TimeRes)
        meals_4HF_C, meal_size_4HF_C, daily_ei_4HF_C, night_meals_4HF_C = average_meals_per_day_and_meal_size(sims["4HF_C"].feature_values[:, idx_ei_ac], sims["4HF_C"].feature_values[:, idx_eating], sims["4HF_C"].time_vector, TimeRes)
        meals_7HF_C, meal_size_7HF_C, daily_ei_7HF_C, night_meals_7HF_C = average_meals_per_day_and_meal_size(sims["7HF_C"].feature_values[:, idx_ei_ac], sims["7HF_C"].feature_values[:, idx_eating], sims["7HF_C"].time_vector, TimeRes)
        meals_20HF_C, meal_size_20HF_C, daily_ei_20HF_C, night_meals_20HF_C = average_meals_per_day_and_meal_size(sims["20HF_C"].feature_values[:, idx_ei_ac], sims["20HF_C"].feature_values[:, idx_eating], sims["20HF_C"].time_vector, TimeRes)
        meals_HF_C_HF_C, meal_size_HF_C_HF_C, daily_ei_HF_C_HF_C, night_meals_HF_C_HF_C = average_meals_per_day_and_meal_size(sims["HF_C_HF_C"].feature_values[:, idx_ei_ac], sims["HF_C_HF_C"].feature_values[:, idx_eating], sims["HF_C_HF_C"].time_vector, TimeRes)
        if first:
            best_sim = sims
            # ---------- daily_ei ----------
            daily_ei_control_min, daily_ei_control_max = daily_ei_control.copy(), daily_ei_control.copy()
            daily_ei_7HF_C_min, daily_ei_7HF_C_max = daily_ei_7HF_C.copy(), daily_ei_7HF_C.copy()
            daily_ei_HF_C_HF_C_min, daily_ei_HF_C_HF_C_max = daily_ei_HF_C_HF_C.copy(), daily_ei_HF_C_HF_C.copy()
            daily_ei_20HF_C_min, daily_ei_20HF_C_max = daily_ei_20HF_C.copy(), daily_ei_20HF_C.copy()
            daily_ei_4HF_C_min, daily_ei_4HF_C_max = daily_ei_4HF_C.copy(), daily_ei_4HF_C.copy()
            daily_ei_control_best = daily_ei_control.copy()
            daily_ei_7HF_C_best = daily_ei_7HF_C.copy()
            daily_ei_HF_C_HF_C_best = daily_ei_HF_C_HF_C.copy()
            daily_ei_20HF_C_best = daily_ei_20HF_C.copy()
            daily_ei_4HF_C_best = daily_ei_4HF_C.copy()
            # ---------- Appetite ----------
            App_control_min, App_control_max = sims["control"].feature_values[:, idx_appetite].copy(), sims["control"].feature_values[:, idx_appetite].copy()
            App_7HF_C_min, App_7HF_C_max = sims["7HF_C"].feature_values[:, idx_appetite].copy(), sims["7HF_C"].feature_values[:, idx_appetite].copy()
            App_20HF_C_min, App_20HF_C_max = sims["20HF_C"].feature_values[:, idx_appetite].copy(), sims["20HF_C"].feature_values[:, idx_appetite].copy()
            App_4HF_C_min, App_4HF_C_max = sims["4HF_C"].feature_values[:, idx_appetite].copy(), sims["4HF_C"].feature_values[:, idx_appetite].copy()
            # ----- BW -----
            BW_control_min, BW_control_max = sims["control"].feature_values[:, idx_bw].copy(), sims["control"].feature_values[:, idx_bw].copy()
            BW_7HF_C_min, BW_7HF_C_max = sims["7HF_C"].feature_values[:, idx_bw].copy(), sims["7HF_C"].feature_values[:, idx_bw].copy()
            BW_HF_C_HF_C_min, BW_HF_C_HF_C_max = sims["HF_C_HF_C"].feature_values[:, idx_bw].copy(), sims["HF_C_HF_C"].feature_values[:, idx_bw].copy()
            BW_20HF_C_min, BW_20HF_C_max = sims["20HF_C"].feature_values[:, idx_bw].copy(), sims["20HF_C"].feature_values[:, idx_bw].copy()
            BW_4HF_C_min, BW_4HF_C_max = sims["4HF_C"].feature_values[:, idx_bw].copy(), sims["4HF_C"].feature_values[:, idx_bw].copy()
            # ----- FM -----
            FM_control_min, FM_control_max = sims["control"].feature_values[:, idx_fm].copy(), sims["control"].feature_values[:, idx_fm].copy()
            FM_7HF_C_min, FM_7HF_C_max = sims["7HF_C"].feature_values[:, idx_fm].copy(), sims["7HF_C"].feature_values[:, idx_fm].copy()
            FM_HF_C_HF_C_min, FM_HF_C_HF_C_max = sims["HF_C_HF_C"].feature_values[:, idx_fm].copy(), sims["HF_C_HF_C"].feature_values[:, idx_fm].copy()
            FM_20HF_C_min, FM_20HF_C_max = sims["20HF_C"].feature_values[:, idx_fm].copy(), sims["20HF_C"].feature_values[:, idx_fm].copy()
            FM_4HF_C_min, FM_4HF_C_max = sims["4HF_C"].feature_values[:, idx_fm].copy(), sims["4HF_C"].feature_values[:, idx_fm].copy()
            # ----- lambda_delay -----
            lambda_delay_HF_C_HF_C_min, lambda_delay_HF_C_HF_C_max = sims["HF_C_HF_C"].feature_values[:, idx_lambda_delay].copy(), sims["HF_C_HF_C"].feature_values[:, idx_lambda_delay].copy()

            first = False
        else:
            # ---------- daily_ei ----------
            daily_ei_control_max = np.maximum(daily_ei_control_max, daily_ei_control)
            daily_ei_control_min = np.minimum(daily_ei_control_min, daily_ei_control)
            daily_ei_7HF_C_max = np.maximum(daily_ei_7HF_C_max, daily_ei_7HF_C)
            daily_ei_7HF_C_min = np.minimum(daily_ei_7HF_C_min, daily_ei_7HF_C)
            daily_ei_HF_C_HF_C_max = np.maximum(daily_ei_HF_C_HF_C_max, daily_ei_HF_C_HF_C)
            daily_ei_HF_C_HF_C_min = np.minimum(daily_ei_HF_C_HF_C_min, daily_ei_HF_C_HF_C)
            daily_ei_20HF_C_max = np.maximum(daily_ei_20HF_C_max, daily_ei_20HF_C)
            daily_ei_20HF_C_min = np.minimum(daily_ei_20HF_C_min, daily_ei_20HF_C)
            daily_ei_4HF_C_max = np.maximum(daily_ei_4HF_C_max, daily_ei_4HF_C)
            daily_ei_4HF_C_min = np.minimum(daily_ei_4HF_C_min, daily_ei_4HF_C)
            # ---------- Appetite ----------
            new_vals = sims["control"].feature_values[:, idx_appetite]
            App_control_min = np.minimum(App_control_min, new_vals)
            App_control_max = np.maximum(App_control_max, new_vals)
            new_vals = sims["7HF_C"].feature_values[:, idx_appetite]
            App_7HF_C_min = np.minimum(App_7HF_C_min, new_vals)
            App_7HF_C_max = np.maximum(App_7HF_C_max, new_vals)
            new_vals = sims["20HF_C"].feature_values[:, idx_appetite]
            App_20HF_C_min = np.minimum(App_20HF_C_min, new_vals)
            App_20HF_C_max = np.maximum(App_20HF_C_max, new_vals)
            new_vals = sims["4HF_C"].feature_values[:, idx_appetite]
            App_4HF_C_min = np.minimum(App_4HF_C_min, new_vals)
            App_4HF_C_max = np.maximum(App_4HF_C_max, new_vals)
            # ----- BW -----
            BW_control = sims["control"].feature_values[:, idx_bw]
            BW_control_max = np.maximum(BW_control_max, BW_control)
            BW_control_min = np.minimum(BW_control_min, BW_control)
            BW_7HF_C = sims["7HF_C"].feature_values[:, idx_bw]
            BW_7HF_C_max = np.maximum(BW_7HF_C_max, BW_7HF_C)
            BW_7HF_C_min = np.minimum(BW_7HF_C_min, BW_7HF_C)
            BW_HF_C_HF_C = sims["HF_C_HF_C"].feature_values[:, idx_bw]
            BW_HF_C_HF_C_max = np.maximum(BW_HF_C_HF_C_max, BW_HF_C_HF_C)
            BW_HF_C_HF_C_min = np.minimum(BW_HF_C_HF_C_min, BW_HF_C_HF_C)
            BW_20HF_C = sims["20HF_C"].feature_values[:, idx_bw]
            BW_20HF_C_max = np.maximum(BW_20HF_C_max, BW_20HF_C)
            BW_20HF_C_min = np.minimum(BW_20HF_C_min, BW_20HF_C)
            BW_4HF_C = sims["4HF_C"].feature_values[:, idx_bw]
            BW_4HF_C_max = np.maximum(BW_4HF_C_max, BW_4HF_C)
            BW_4HF_C_min = np.minimum(BW_4HF_C_min, BW_4HF_C)
            # ----- FM -----
            FM_control = sims["control"].feature_values[:, idx_fm]
            FM_control_max = np.maximum(FM_control_max, FM_control)
            FM_control_min = np.minimum(FM_control_min, FM_control)
            FM_7HF_C = sims["7HF_C"].feature_values[:, idx_fm]
            FM_7HF_C_max = np.maximum(FM_7HF_C_max, FM_7HF_C)
            FM_7HF_C_min = np.minimum(FM_7HF_C_min, FM_7HF_C)
            FM_HF_C_HF_C = sims["HF_C_HF_C"].feature_values[:, idx_fm]
            FM_HF_C_HF_C_max = np.maximum(FM_HF_C_HF_C_max, FM_HF_C_HF_C)
            FM_HF_C_HF_C_min = np.minimum(FM_HF_C_HF_C_min, FM_HF_C_HF_C)
            FM_20HF_C = sims["20HF_C"].feature_values[:, idx_fm]
            FM_20HF_C_max = np.maximum(FM_20HF_C_max, FM_20HF_C)
            FM_20HF_C_min = np.minimum(FM_20HF_C_min, FM_20HF_C)
            FM_4HF_C = sims["4HF_C"].feature_values[:, idx_fm]
            FM_4HF_C_max = np.maximum(FM_4HF_C_max, FM_4HF_C)
            FM_4HF_C_min = np.minimum(FM_4HF_C_min, FM_4HF_C)
            # ----- lambda_delay -----
            lambda_delay_HF_C_HF_C = sims["HF_C_HF_C"].feature_values[:, idx_lambda_delay]
            lambda_delay_HF_C_HF_C_max = np.maximum(lambda_delay_HF_C_HF_C_max, lambda_delay_HF_C_HF_C)
            lambda_delay_HF_C_HF_C_min = np.minimum(lambda_delay_HF_C_HF_C_min, lambda_delay_HF_C_HF_C)
    def compute_is_night(time, phase_shift_hours=16):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        IsNight = cycle > 0
        return IsNight
    time = sims['control'].time_vector
    IsNight = compute_is_night(time)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time)-1]))


    colors = {
        "control": "blue",
        "4HF-C": "teal",
        "7HF-C": "darkorange",
        "20HF-C": "firebrick",
        "HF_C_HF_C": "firebrick"
        }
    
   # ---------- Plot 1: Hall EI + eating ----------
    time = sims['control'].time_vector
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3, width_ratios=[2.3,1,1], wspace=0.1, hspace=0.3)
    axes = np.empty((4,3), dtype=object)
    for i in range(4):
        axes[i,0] = fig.add_subplot(gs[i,0])
        axes[i,1] = fig.add_subplot(gs[i,1])
        axes[i,2] = fig.add_subplot(gs[i,2])
    for i in range(4):
        pos0 = axes[i,0].get_position()
        pos1 = axes[i,1].get_position()
        pos2 = axes[i,2].get_position()
        axes[i,0].set_position([pos0.x0 - 0.03, pos0.y0, pos0.width, pos0.height])
        axes[i,1].set_position([pos1.x0 + 0.05, pos1.y0, pos1.width, pos1.height])
        axes[i,2].set_position([pos2.x0 + 0.05, pos2.y0, pos2.width, pos2.height])
    time_daily = np.arange(len(daily_ei_control_best))
    hall_time_weeks = np.asarray(data["Hall_EI"]["Time"], dtype=float) / 7
    time_weeks = time / 7
    #  CONTROL 
    axes[0,0].fill_between(time_daily/7, daily_ei_control_min, daily_ei_control_max, color=colors["control"], alpha=0.2)
    axes[0,0].plot(time_daily/7, daily_ei_control_best, color=colors["control"], lw=2)
    axes[0,0].scatter(hall_time_weeks, data["Hall_EI"]["Control"], color="black", s=35)
    axes[0,1].fill_between(time_weeks, App_control_min, App_control_max, color=colors["control"], alpha=0.2)
    axes[0,1].plot(time_weeks, best_sim["control"].feature_values[:, idx_appetite], color=colors["control"], lw=2)
    axes[0,1].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    axes[0,2].fill_between(time, App_control_min, App_control_max, color=colors["control"], alpha=0.2)
    axes[0,2].plot(time, best_sim["control"].feature_values[:, idx_appetite], color=colors["control"], lw=2)
    axes[0,2].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    #  4HF_C 
    axes[1,0].fill_between(time_daily/7, daily_ei_4HF_C_min, daily_ei_4HF_C_max, color=colors["4HF-C"], alpha=0.2)
    axes[1,0].plot(time_daily/7, daily_ei_4HF_C_best, color=colors["4HF-C"], lw=2)
    axes[1,0].scatter(hall_time_weeks, data["Hall_EI"]["4HF-C"], color="black", s=35)
    axes[1,1].fill_between(time_weeks, App_4HF_C_min, App_4HF_C_max, color=colors["4HF-C"], alpha=0.2)
    axes[1,1].plot(time_weeks, best_sim["4HF_C"].feature_values[:, idx_appetite], color=colors["4HF-C"], lw=2)
    axes[1,1].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    axes[1,2].fill_between(time, App_4HF_C_min, App_4HF_C_max, color=colors["4HF-C"], alpha=0.2)
    axes[1,2].plot(time, best_sim["4HF_C"].feature_values[:, idx_appetite], color=colors["4HF-C"], lw=2)
    axes[1,2].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    #  7HF_C 
    axes[2,0].fill_between(time_daily/7, daily_ei_7HF_C_min, daily_ei_7HF_C_max, color=colors["7HF-C"], alpha=0.2)
    axes[2,0].plot(time_daily/7, daily_ei_7HF_C_best, color=colors["7HF-C"], lw=2)
    axes[2,0].scatter(hall_time_weeks, data["Hall_EI"]["7HF-C"], color="black", s=35)
    axes[2,1].fill_between(time_weeks, App_7HF_C_min, App_7HF_C_max, color=colors["7HF-C"], alpha=0.2)
    axes[2,1].plot(time_weeks, best_sim["7HF_C"].feature_values[:, idx_appetite], color=colors["7HF-C"], lw=2)
    axes[2,1].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    axes[2,2].fill_between(time, App_7HF_C_min, App_7HF_C_max, color=colors["7HF-C"], alpha=0.2)
    axes[2,2].plot(time, best_sim["7HF_C"].feature_values[:, idx_appetite], color=colors["7HF-C"], lw=2)
    axes[2,2].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    #  20HF_C 
    axes[3,0].fill_between(time_daily/7, daily_ei_20HF_C_min, daily_ei_20HF_C_max, color=colors["20HF-C"], alpha=0.2)
    axes[3,0].plot(time_daily/7, daily_ei_20HF_C_best, color=colors["20HF-C"], lw=2)
    axes[3,0].scatter(hall_time_weeks, data["Hall_EI"]["20HF-C"], color="black", s=35)
    axes[3,1].fill_between(time_weeks, App_20HF_C_min, App_20HF_C_max, color=colors["20HF-C"], alpha=0.2)
    axes[3,1].plot(time_weeks, best_sim["20HF_C"].feature_values[:, idx_appetite], color=colors["20HF-C"], lw=2)
    axes[3,1].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    axes[3,2].fill_between(time, App_20HF_C_min, App_20HF_C_max, color=colors["20HF-C"], alpha=0.2)
    axes[3,2].plot(time, best_sim["20HF_C"].feature_values[:, idx_appetite], color=colors["20HF-C"], lw=2)
    axes[3,2].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    #  formatting 
    diet_intervals = [(None,None),(0,4),(0,7),(0,20)]
    y_linediet_intervals = [0,0,0]
    for i in range(4):
        t0, t1 = diet_intervals[i]
        axes[i,0].set_ylim(0,28)
        axes[i,0].set_yticks([0,10,20])
        axes[i,1].set_ylim(0,1.5)
        axes[i,1].set_yticks([1])
        axes[i,1].set_yticklabels(["Hunger\nthreshold"])
        axes[i,2].set_ylim(0,1.5)
        axes[i,2].set_yticks([])
        axes[i,2].set_yticks([], minor=True)
        axes[i,2].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        axes[i,2].set_ylabel("")
        axes[i,0].set_ylabel("(kcal/day)", fontsize=18)
        for j in range(3):
            axes[i,j].grid(True)
            axes[i,j].tick_params(axis='both', which='major', labelsize=18)
            axes[i,j].tick_params(axis='both', which='minor', labelsize=18)
            plt.setp(axes[i,j].get_xticklabels(), fontsize=18)
            plt.setp(axes[i,j].get_yticklabels(), fontsize=18)
            if t0 is not None:
                axes[i,j].plot([t0,t1],[y_linediet_intervals[j],y_linediet_intervals[j]], color="black", lw=8, solid_capstyle="butt")
            if IsNight is not None:
                if j < 2:
                    x_shade = time_weeks
                else:
                    x_shade = time
                for k in range(len(edges)-1):
                    if IsNight[edges[k]]:
                        axes[i,j].axvspan(x_shade[edges[k]], x_shade[edges[k+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
        axes[i,0].set_xlim(-0.2,33)
        axes[i,1].set_xlim(-0.2,33)
        axes[i,2].set_xlim(0,1)
        axes[i,2].set_xticks([0,0.5,1.0])
        axes[i,2].set_xticklabels(["12:00","24:00","12:00"])
    axes[3,0].set_xlabel("Time (weeks)", fontsize=18)
    axes[3,1].set_xlabel("Time (weeks)", fontsize=18)
    axes[3,2].set_xlabel("Time of day", fontsize=18)
    axes[0,0].set_title("Daily caloric intake", fontsize=18)
    axes[0,1].set_title("Appetite (full study)", fontsize=18)
    axes[0,2].set_title("Appetite (first day)", fontsize=18)

    fig.tight_layout(rect=[0,0,1,0.96])
    plt.show()
   

    # ---------- Plot 2: Hall BW and FM ----------
    fig, axes = plt.subplots(4, 2, figsize=(11, 13), sharex=True)
    t_data = np.array(data["Hall_BW"]["Time"], dtype=float) / 7.0
    time = sims["control"].time_vector / 7
    #  Row 1: Control 
    axes[0,0].fill_between(time, BW_control_min, BW_control_max, color=colors["control"], alpha=0.4, linewidth=0)
    axes[0,0].plot(time, best_sim["control"].feature_values[:,idx_bw], lw=2, color=colors["control"], label="Chow diet")
    axes[0,0].errorbar(t_data, data["Hall_BW"]["ControlBW_mean"], yerr=data["Hall_BW"]["ControlBW_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[0,1].fill_between(time, FM_control_min, FM_control_max, color=colors["control"], alpha=0.4, linewidth=0)
    axes[0,1].plot(time, best_sim["control"].feature_values[:,idx_fm], lw=2, color=colors["control"], label="Chow diet")
    axes[0,1].errorbar(t_data, data["Hall_BW"]["ControlFM_mean"], yerr=data["Hall_BW"]["ControlFM_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[0,0].set_ylabel("(g)", fontsize=18)
    axes[0,1].set_ylabel("(g)", fontsize=18)
    #  Row 2: 4HF-C 
    axes[1,0].fill_between(time, BW_4HF_C_min, BW_4HF_C_max, color=colors["4HF-C"], alpha=0.4, linewidth=0)
    axes[1,0].plot(time, best_sim["4HF_C"].feature_values[:,idx_bw], lw=2, color=colors["4HF-C"], label="High fat diet for 4 weeks")
    axes[1,0].errorbar(t_data, data["Hall_BW"]["HF4_CBW_mean"], yerr=data["Hall_BW"]["HF4_CBW_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[1,1].fill_between(time, FM_4HF_C_min, FM_4HF_C_max, color=colors["4HF-C"], alpha=0.4, linewidth=0)
    axes[1,1].plot(time, best_sim["4HF_C"].feature_values[:,idx_fm], lw=2, color=colors["4HF-C"], label="High fat diet for 4 weeks")
    axes[1,1].errorbar(t_data, data["Hall_BW"]["HF4_CFM_mean"], yerr=data["Hall_BW"]["HF4_CFM_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[1,0].set_ylabel("(g)", fontsize=18)
    axes[1,1].set_ylabel("(g)", fontsize=18)
    axes[1,0].plot([0,4],[20,20], color='black', lw=8, solid_capstyle='butt')
    axes[1,1].plot([0,4],[0,0], color='black', lw=8, solid_capstyle='butt')
    #  Row 3: 7HF-C 
    axes[2,0].fill_between(time, BW_7HF_C_min, BW_7HF_C_max, color=colors["7HF-C"], alpha=0.4, linewidth=0)
    axes[2,0].plot(time, best_sim["7HF_C"].feature_values[:,idx_bw], lw=2, color=colors["7HF-C"], label="High fat diet for 7 weeks")
    axes[2,0].errorbar(t_data, data["Hall_BW"]["HF7_CBW_mean"], yerr=data["Hall_BW"]["HF7_CBW_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[2,1].fill_between(time, FM_7HF_C_min, FM_7HF_C_max, color=colors["7HF-C"], alpha=0.4, linewidth=0)
    axes[2,1].plot(time, best_sim["7HF_C"].feature_values[:,idx_fm], lw=2, color=colors["7HF-C"], label="High fat diet for 7 weeks")
    axes[2,1].errorbar(t_data, data["Hall_BW"]["HF7_CFM_mean"], yerr=data["Hall_BW"]["HF7_CFM_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[2,0].set_ylabel("(g)", fontsize=18)
    axes[2,1].set_ylabel("(g)", fontsize=18)
    axes[2,0].plot([0,7],[20,20], color='black', lw=8, solid_capstyle='butt')
    axes[2,1].plot([0,7],[0,0], color='black', lw=8, solid_capstyle='butt')
    #  Row 4: 20HF-C 
    axes[3,0].fill_between(time, BW_20HF_C_min, BW_20HF_C_max, color=colors["20HF-C"], alpha=0.4, linewidth=0)
    axes[3,0].plot(time, best_sim["20HF_C"].feature_values[:,idx_bw], lw=2, color=colors["20HF-C"], label="High fat diet for 20 weeks")
    axes[3,0].errorbar(t_data, data["Hall_BW"]["HF20_CBW_mean"], yerr=data["Hall_BW"]["HF20_CBW_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[3,1].fill_between(time, FM_20HF_C_min, FM_20HF_C_max, color=colors["20HF-C"], alpha=0.4, linewidth=0)
    axes[3,1].plot(time, best_sim["20HF_C"].feature_values[:,idx_fm], lw=2, color=colors["20HF-C"], label="High fat diet for 20 weeks")
    axes[3,1].errorbar(t_data, data["Hall_BW"]["HF20_CFM_mean"], yerr=data["Hall_BW"]["HF20_CFM_sd"],fmt='none',color="black",capsize=8, elinewidth=2, capthick=2,markersize=8)
    axes[3,0].set_ylabel("(g)", fontsize=18)
    axes[3,1].set_ylabel("(g)", fontsize=18)
    axes[3,0].plot([0,20],[20,20], color='black', lw=8, solid_capstyle='butt')
    axes[3,1].plot([0,20],[0,0], color='black', lw=8, solid_capstyle='butt')
    #  Formatting 
    for i in range(4):
        axes[i,0].set_xlim(-0.5, 33)
        axes[i,1].set_xlim(-0.5, 33)
        axes[i,0].set_ylim(20, 60)
        axes[i,1].set_ylim(0, 30)
        axes[i,0].set_yticks([20,40,60])
        axes[i,1].set_yticks([0,15,30])
        axes[i,0].grid(True)
        axes[i,1].grid(True)
        axes[i,0].legend(loc="lower right", framealpha=1, fontsize=12)
        axes[i,1].legend(loc="upper left", framealpha=1, fontsize=12)
        axes[i,0].tick_params(axis='both', which='major', labelsize=18)
        axes[i,1].tick_params(axis='both', which='major', labelsize=18)
        plt.setp(axes[i,0].get_xticklabels(), fontsize=18)
        plt.setp(axes[i,0].get_yticklabels(), fontsize=18)
        plt.setp(axes[i,1].get_xticklabels(), fontsize=18)
        plt.setp(axes[i,1].get_yticklabels(), fontsize=18)
    axes[-1,0].set_xlabel("Time (weeks)", fontsize=18)
    axes[-1,1].set_xlabel("Time (weeks)", fontsize=18)
    axes[0,0].set_title("Total body mass", fontsize=18)
    axes[0,1].set_title("Fat mass", fontsize=18)
    for i in range(4):
        for j in range(2):
            for k in range(len(edges)-1):
                if IsNight[edges[k]]:
                    axes[i,j].axvspan( time[edges[k]], time[edges[k+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    fig.tight_layout(rect=[0,0,1,0.96])
    plt.show()


    # ---------- Plot 3: Hall validation ----------
    fig_val, axes_val = plt.subplots(2, 2, figsize=(12, 8))
    t_data = np.array(data["Hall_BW"]["Time"], dtype=float) / 7.0
    time = sims["control"].time_vector / 7
    ax_val_BW     = axes_val[0,0]
    ax_val_FM     = axes_val[0,1]
    ax_val_EI     = axes_val[1,0]
    ax_val_lambda = axes_val[1,1]
    axes_all = [ax_val_BW, ax_val_FM, ax_val_EI, ax_val_lambda]
    #  ROW 1: BW 
    ax_val_BW.fill_between(time, BW_HF_C_HF_C_min, BW_HF_C_HF_C_max, color=colors["HF_C_HF_C"], alpha=0.4, linewidth=0)
    ax_val_BW.plot(time, best_sim["HF_C_HF_C"].feature_values[:,idx_bw], lw=2, color=colors["HF_C_HF_C"], label="High fat diet 0-7 and 10-20 weeks")
    ax_val_BW.errorbar(t_data, data["Hall_BW"]["HF_C_HF_CBW_mean"], yerr=data["Hall_BW"]["HF_C_HF_CBW_sd"], fmt='none', color="black", capsize=8, elinewidth=2, capthick=2)
    ax_val_BW.set_ylabel("Total body mass\n(g)", fontsize=18)
    ax_val_BW.set_xlim(-0.2,33)
    ax_val_BW.set_ylim(20,60)
    ax_val_BW.grid(True)
    ax_val_BW.plot([0,7],[20,20], color='black', lw=8, solid_capstyle='butt')
    ax_val_BW.plot([10,20],[20,20], color='black', lw=8, solid_capstyle='butt')
    #  ROW 1: FM 
    ax_val_FM.fill_between(time, FM_HF_C_HF_C_min, FM_HF_C_HF_C_max, color=colors["HF_C_HF_C"], alpha=0.4, linewidth=0)
    ax_val_FM.plot(time, best_sim["HF_C_HF_C"].feature_values[:,idx_fm], lw=2, color=colors["HF_C_HF_C"], label="High fat diet 0-7 and 10-20 weeks")
    ax_val_FM.errorbar(t_data, data["Hall_BW"]["HF_C_HF_CFM_mean"], yerr=data["Hall_BW"]["HF_C_HF_CFM_sd"], fmt='none', color="black", capsize=8, elinewidth=2, capthick=2)
    ax_val_FM.set_ylabel("Fat mass\n(g)", fontsize=18)
    ax_val_FM.set_xlim(-0.2,33)
    ax_val_FM.set_ylim(0,30)
    ax_val_FM.grid(True)
    ax_val_FM.plot([0,7],[0,0], color='black', lw=8, solid_capstyle='butt')
    ax_val_FM.plot([10,20],[0,0], color='black', lw=8, solid_capstyle='butt')
    #  ROW 2: EI 
    t_data_EI = np.array(data["Hall_EI"]["Time"], dtype=float) / 7
    ax_val_EI.fill_between(time_daily/7, daily_ei_HF_C_HF_C_min, daily_ei_HF_C_HF_C_max, color=colors["HF_C_HF_C"], alpha=0.2)
    ax_val_EI.plot(time_daily/7, daily_ei_HF_C_HF_C_best, color=colors["HF_C_HF_C"], lw=2, label="Energy intake (predicted)")
    ax_val_EI.scatter(t_data_EI, data["Hall_EI"]["HF-C-HF-C"], color="black", s=35, label="Energy intake (data)")
    ax_val_EI.set_ylabel("Energy intake\n(kcal/day)", fontsize=18)
    ax_val_EI.set_xlim(-0.2,33)
    ax_val_EI.set_ylim(-5,35)
    ax_val_EI.set_yticks([0,15,30])
    ax_val_EI.grid(True)
    ax_val_EI.plot([0.1,7],[-5,-5], color='black', lw=8, solid_capstyle='butt')
    ax_val_EI.plot([10,20],[-5,-5], color='black', lw=8, solid_capstyle='butt')
    #  ROW 2: lambda + lambda_delay 
    ax_val_lambda.plot(time, best_sim["HF_C_HF_C"].feature_values[:,idx_lambda], lw=2, color=colors["HF_C_HF_C"])
    ax_val_lambda.fill_between(time, lambda_delay_HF_C_HF_C_min, lambda_delay_HF_C_HF_C_max, color=colors["HF_C_HF_C"], alpha=0.15, linewidth=0, label="Diet induced change")
    ax_val_lambda.plot(time, best_sim["HF_C_HF_C"].feature_values[:,idx_lambda_delay], lw=2, color=colors["HF_C_HF_C"], ls="--", label="Delayed dynamics")
    ax_val_lambda.set_ylabel("Physical activity\n(kcal/g/day)", fontsize=18)
    ax_val_lambda.set_xlim(-0.2,33)
    ax_val_lambda.set_ylim(0,0.35)
    ax_val_lambda.set_yticks([0,0.15,0.3])
    ax_val_lambda.grid(True)
    ax_val_lambda.plot([0,7],[0,0], color='black', lw=8, solid_capstyle='butt')
    ax_val_lambda.plot([10,20],[0,0], color='black', lw=8, solid_capstyle='butt')
    #  FORMATTING 
    for ax in axes_all:
        ax.set_xlabel("Time (weeks)", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.setp(ax.get_yticklabels(), fontsize=18)
    for ax in axes_all:
        for k in range(len(edges)-1):
            if IsNight[edges[k]]:
                ax.axvspan(
                    time[edges[k]],
                    time[edges[k+1]],
                    color="black",
                    alpha=0.08,
                    edgecolor="none",
                    linewidth=0,
                    zorder=0
                )
    fig_val.tight_layout(rect=[0,0,1,0.96])
    fig_val.subplots_adjust(wspace=0.35, hspace=0.30)
    plt.show()

    
    # ----- Plot 4: Weight change zoomed in -----
    fig2 = plt.figure(figsize=(16,12))
    gs = gridspec.GridSpec(4,3, width_ratios=[1.2,1,1], hspace=0.35, wspace=0.35)
    # Column 2 and 3
    axes2 = np.empty((4,2), dtype=object)
    for i in range(4):
        axes2[i,0] = fig2.add_subplot(gs[i,1])
        axes2[i,1] = fig2.add_subplot(gs[i,2])
    # SINGLE BW zoom axis (0–7 days)
    ax_cum = fig2.add_subplot(gs[:,0])
    # use DAYS (not weeks)
    time_days = sims["control"].time_vector
    #  BW min/max for all diets 
    ax_cum.fill_between(time_days, BW_control_min, BW_control_max, color=colors["control"], alpha=0.2, edgecolor="none")
    ax_cum.plot(time_days, best_sim["control"].feature_values[:,idx_bw], lw=2, color=colors["control"], label="Control")
    ax_cum.errorbar(data["Hall_BW"]["Time"], data["Hall_BW"]["ControlBW_mean"], yerr=data["Hall_BW"]["ControlBW_sd"], fmt='none', color=colors["control"], capsize=8, capthick = 3,elinewidth=3)
    ax_cum.fill_between(time_days, BW_20HF_C_min, BW_20HF_C_max, color=colors["20HF-C"], alpha=0.2, edgecolor="none")
    ax_cum.plot(time_days, best_sim["20HF_C"].feature_values[:,idx_bw], lw=2, color=colors["20HF-C"], label="High fat diet 20 weeks")
    ax_cum.errorbar(data["Hall_BW"]["Time"], data["Hall_BW"]["HF20_CBW_mean"], yerr=data["Hall_BW"]["HF20_CBW_sd"], fmt='none', color=colors["20HF-C"], capsize=8, capthick = 3, elinewidth=3)
    #  formatting 
    ax_cum.set_ylabel("(g)", fontsize=18)
    ax_cum.set_title("Total body mass (first week)", fontsize=18)
    ax_cum.set_xlabel("Time (days)", fontsize=18)
    ax_cum.set_xlim(-0.1, 7.1)
    ax_cum.set_ylim(25, 36)
    ax_cum.set_yticks([26,28,30,32,34,36])
    ax_cum.set_xticks([0,1,2,3,4,5,6,7])
    ax_cum.tick_params(axis='both', which='major', labelsize=18)
    ax_cum.grid(True)
    ax_cum.legend(loc="upper left", fontsize=13, framealpha=1)
    for k in range(len(edges)-1):
        if IsNight[edges[k]]:
            ax_cum.axvspan(time_days[edges[k]],  time_days[edges[k+1]], color="black",  alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    plt.show()


# --- Mack studies ---
# Study 1
def plot_MackAppetite_uncertainty(Model_Appetite, model_PK, data, AppetiteParams, PKParams, FastPlot):

    AppetiteParams = AppetiteParams.to_numpy(dtype=float)
    PKParams       = PKParams.to_numpy(dtype=float)

    Best_AppetiteParams = np.asarray(AppetiteParams[0][:-1], dtype=float)
    Best_PKParams       = np.asarray(PKParams[0][:-1], dtype=float)


    n_select = 300
    if FastPlot:
        n_select = 3
    idx = np.linspace(0, AppetiteParams.shape[0] - 1, n_select, dtype=int)
    AppetiteParams = AppetiteParams[idx]
    n_select = 500
    if FastPlot:
        n_select = 100
    idx = np.linspace(0, PKParams.shape[0] - 1, n_select, dtype=int)
    PKParams = PKParams[idx]

    MackSimStudy1_IC50_1   = Uncertainty_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, Best_PKParams, 0.00205, True) # 0.00205 nmol/L (PMID: 28698254)
    MackSimStudy1_IC50_2   = Uncertainty_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, Best_PKParams, 0.093, True)   # 0.093 nmol/L (PMID: 8405712)
    MackSimStudy1_IC50_3   = Uncertainty_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, Best_PKParams, 0.65, True)    # 0.65 nmol/L (PMID: 17138746)
    MackSimStudy1_IC50_4   = Uncertainty_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, Best_PKParams, 0.02, True)    # Our calibrated value

    # PK uncertainty
    first = True
    for param_vector in PKParams:
        full_param = np.asarray(param_vector[:-1], dtype=float)
        try:
            MackSimStudy1 = Simulate_MackStudy1(Model_Appetite, model_PK, data, Best_AppetiteParams, full_param, 0.02, True)
        except Exception:
            continue
        idx_ei           = MackSimStudy1["vehicle"].feature_names.index("EI_AC")
        idx_Appetite     = MackSimStudy1["vehicle"].feature_names.index("Appetite")
        idx_EI_appetite  = MackSimStudy1["vehicle"].feature_names.index("EI_appetite")
        idx_c1           = MackSimStudy1["exenatide"].feature_names.index("Plasma_Exenatide")
        C1_exenatide_new = MackSimStudy1["exenatide"].feature_values[:, idx_c1]
        if first:
            C1_exenatide_min = C1_exenatide_new.copy()
            C1_exenatide_max = C1_exenatide_new.copy()
            first = False
        else:
            C1_exenatide_min = np.minimum(C1_exenatide_min, C1_exenatide_new)
            C1_exenatide_max = np.maximum(C1_exenatide_max, C1_exenatide_new)
    def compute_is_night(time, phase_shift_hours=12):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        IsNight = cycle > 0
        return IsNight
    
    time   = MackSimStudy1["vehicle"].time_vector
    IsNight = compute_is_night(time,20)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time)-1]))
    t_data = np.array(data["MackStudy1"]["Time"][0:7], dtype=float) *24 + 16
    vehicle_data = np.array(data["MackStudy1"]["VehicleEI"][0:7], dtype=float)/3     # 3 mice/cage
    exenatide_data = np.array(data["MackStudy1"]["ExenatideEI"][0:7], dtype=float)/3 # 3 mice/cage

    colors = {
        "control": "blue",
        "exenatide": "firebrick"
        }

    # ---------- Plot 1 (estimation data) ----------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    time_hours = time * 24
    time_days = MackSimStudy1_IC50_1["best_sim"]["vehicle"].time_vector
    time_hours_dn = time_days * 24
    #  COLUMN 1: ENERGY INTAKE 
    axes[0].fill_between(time_hours, MackSimStudy1_IC50_1["EI_vehicle_min"]/3.4, MackSimStudy1_IC50_1["EI_vehicle_max"]/3.4, color=colors["control"], alpha=0.2, edgecolor="none")
    axes[0].plot(time_hours, MackSimStudy1_IC50_1["best_sim"]["vehicle"].feature_values[:, idx_ei]/3.4, color=colors["control"], lw=2)
    axes[0].errorbar(t_data, vehicle_data, yerr=np.array(data["MackStudy1"]["VehicleEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["control"], capsize=8, elinewidth=2, capthick=2)
    axes[0].set_title("Energy intake", fontsize=18)
    axes[0].set_ylabel("Food consumed\n(g/mice)", fontsize=18)
    axes[0].grid(True)
    #  COLUMN 2: APPETITE 
    axes[1].plot(time_hours, MackSimStudy1_IC50_1["best_sim"]["vehicle"].feature_values[:, idx_Appetite], color=colors["control"], lw=2)
    axes[1].fill_between(time_hours, MackSimStudy1_IC50_1["Appetite_vehicle_min"], MackSimStudy1_IC50_1["Appetite_vehicle_max"], color=colors["control"], alpha=0.3)
    axes[1].set_yscale("log")
    axes[1].set_yticklabels(["1x Hunger\nthreshold", "10x Hunger\nthreshold" , "100x Hunger\nthreshold"])
    axes[1].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Appetite", fontsize=18)
    axes[1].grid(True)
    #  COLUMN 3: FED STATE DEPENDENCY 
    axes[2].fill_between(time_hours, MackSimStudy1_IC50_1["EI_appetite_vehicle_min"], MackSimStudy1_IC50_1["EI_appetite_vehicle_max"], color=colors["control"], alpha=0.3)
    axes[2].plot(time_hours,MackSimStudy1_IC50_1["best_sim"]["vehicle"].feature_values[:, idx_EI_appetite] / MackSimStudy1_IC50_1["best_sim"]["vehicle"].feature_values[0, idx_EI_appetite], color=colors["control"],lw=2)
    axes[2].axhline(y=1, color="black", linestyle="--", linewidth=1.5)
    axes[2].set_title("Fed state impact on appetite", fontsize=18)
    axes[2].set_ylabel("(fold increase)", fontsize=18)
    axes[2].grid(True)
    #  FORMATTING 
    axes[0].set_xlim(0,23)
    axes[1].set_xlim(0,23)
    axes[2].set_xlim(0,23)
    axes[0].set_xticks([0,4,8,12,16,20])
    axes[1].set_xticks([0,4,8,12,16,20])
    axes[2].set_xticks([0,4,8,12,16,20])
    axes[0].set_yticks([0,1,2,3])
    axes[1].set_yticks([1,10,100])
    axes[2].set_yticks([0.6,1,1.4,1.8])
    axes[0].set_xlabel("Time (h)", fontsize=18)
    axes[1].set_xlabel("Time (h)", fontsize=18)
    axes[2].set_xlabel("Time (h)", fontsize=18)
    axes[0].set_ylim(-0.2,3.2)
    app_ylim = np.max(MackSimStudy1_IC50_1["Appetite_vehicle_max"]) * 1.05
    axes[1].set_ylim(0.2, app_ylim)
    axes[2].set_ylim(0.6, 1.8)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[2].tick_params(axis='both', which='major', labelsize=18)
    for j in range(3):
        for k in range(len(edges)-1):
           if IsNight[edges[k]]:
                axes[j].axvspan(time_hours_dn[edges[k]], time_hours_dn[edges[k+1]], color="black", alpha=0.08, linewidth=0)
    axes[0].plot([0,16],[-0.2,-0.2], color='black', lw=8, solid_capstyle='butt')
    axes[1].plot([0,16],[0.2,0.2], color='black', lw=8, solid_capstyle='butt')
    axes[2].plot([0,16],[0.6,0.6], color='black', lw=8, solid_capstyle='butt')
    fig.subplots_adjust(wspace=0.15)
    fig.tight_layout(rect=[0,0,1,0.95])
    plt.show()


    # ---------- Plot 2 (exenatide) ----------
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    time_hours = time * 24
    time_days = MackSimStudy1_IC50_1["best_sim"]["vehicle"].time_vector
    time_hours_dn = time_days * 24
    #  COLUMN 1 
    axes[0].fill_between(time_hours, MackSimStudy1_IC50_1["EI_vehicle_min"]/3.4, MackSimStudy1_IC50_1["EI_vehicle_max"]/3.4, color=colors["control"], alpha=0.2, edgecolor="none")
    axes[0].plot(time_hours, MackSimStudy1_IC50_1["best_sim"]["vehicle"].feature_values[:, idx_ei]/3.4, color=colors["control"], lw=2)
    axes[0].errorbar(t_data, vehicle_data, yerr=np.array(data["MackStudy1"]["VehicleEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["control"], capsize=8, elinewidth=2, capthick=2)
    axes[0].fill_between(time_hours, MackSimStudy1_IC50_1["EI_exenatide_min"]/3.4, MackSimStudy1_IC50_1["EI_exenatide_max"]/3.4, color=colors["exenatide"], alpha=0.2, edgecolor="none")
    axes[0].plot(time_hours, MackSimStudy1_IC50_1["best_sim"]["exenatide"].feature_values[:, idx_ei]/3.4, color=colors["exenatide"], lw=2)
    axes[0].errorbar(t_data, exenatide_data, yerr=np.array(data["MackStudy1"]["ExenatideEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["exenatide"], capsize=8, elinewidth=2, capthick=2)
    axes[0].set_title("Iwasaki et al.\n(IC50 = 0.00205 nmol/L)", fontsize=18)
    axes[0].set_ylabel("Food consumed\n(g/mice)", fontsize=18)
    axes[0].grid(True)
    #  COLUMN 2 
    axes[1].fill_between(time_hours, MackSimStudy1_IC50_4["EI_vehicle_min"]/3.4, MackSimStudy1_IC50_4["EI_vehicle_max"]/3.4, color=colors["control"], alpha=0.2, edgecolor="none")
    axes[1].plot(time_hours, MackSimStudy1_IC50_4["best_sim"]["vehicle"].feature_values[:, idx_ei]/3.4, color=colors["control"], lw=2)
    axes[1].errorbar(t_data, vehicle_data, yerr=np.array(data["MackStudy1"]["VehicleEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["control"], capsize=8, elinewidth=2, capthick=2)
    axes[1].fill_between(time_hours, MackSimStudy1_IC50_4["EI_exenatide_min"]/3.4, MackSimStudy1_IC50_4["EI_exenatide_max"]/3.4, color=colors["exenatide"], alpha=0.2, edgecolor="none")
    axes[1].plot(time_hours, MackSimStudy1_IC50_4["best_sim"]["exenatide"].feature_values[:, idx_ei]/3.4, color=colors["exenatide"], lw=2)
    axes[1].errorbar(t_data, exenatide_data, yerr=np.array(data["MackStudy1"]["ExenatideEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["exenatide"], capsize=8, elinewidth=2, capthick=2)
    axes[1].set_title("Our best estimate from Fig. 5A\n(IC50 = 0.02 nmol/L)", fontsize=18)
    axes[1].grid(True)
    #  COLUMN 3 
    axes[2].fill_between(time_hours, MackSimStudy1_IC50_2["EI_vehicle_min"]/3.4, MackSimStudy1_IC50_2["EI_vehicle_max"]/3.4, color=colors["control"], alpha=0.2, edgecolor="none")
    axes[2].plot(time_hours, MackSimStudy1_IC50_2["best_sim"]["vehicle"].feature_values[:, idx_ei]/3.4, color=colors["control"], lw=2)
    axes[2].errorbar(t_data, vehicle_data, yerr=np.array(data["MackStudy1"]["VehicleEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["control"], capsize=8, elinewidth=2, capthick=2)
    axes[2].fill_between(time_hours, MackSimStudy1_IC50_2["EI_exenatide_min"]/3.4, MackSimStudy1_IC50_2["EI_exenatide_max"]/3.4, color=colors["exenatide"], alpha=0.2, edgecolor="none")
    axes[2].plot(time_hours, MackSimStudy1_IC50_2["best_sim"]["exenatide"].feature_values[:, idx_ei]/3.4, color=colors["exenatide"], lw=2)
    axes[2].errorbar(t_data, exenatide_data, yerr=np.array(data["MackStudy1"]["ExenatideEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["exenatide"], capsize=8, elinewidth=2, capthick=2)
    axes[2].set_title("Thorens et al.\n(IC50 = 0.093 nmol/L)", fontsize=18)
    axes[2].grid(True)
    #  COLUMN 4
    axes[3].fill_between(time_hours, MackSimStudy1_IC50_3["EI_vehicle_min"]/3.4, MackSimStudy1_IC50_3["EI_vehicle_max"]/3.4, color=colors["control"], alpha=0.2, edgecolor="none")
    axes[3].plot(time_hours, MackSimStudy1_IC50_3["best_sim"]["vehicle"].feature_values[:, idx_ei]/3.4, color=colors["control"], lw=2)
    axes[3].errorbar(t_data, vehicle_data, yerr=np.array(data["MackStudy1"]["VehicleEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["control"], capsize=8, elinewidth=2, capthick=2)
    axes[3].fill_between(time_hours, MackSimStudy1_IC50_3["EI_exenatide_min"]/3.4, MackSimStudy1_IC50_3["EI_exenatide_max"]/3.4, color=colors["exenatide"], alpha=0.2, edgecolor="none")
    axes[3].plot(time_hours, MackSimStudy1_IC50_3["best_sim"]["exenatide"].feature_values[:, idx_ei]/3.4, color=colors["exenatide"], lw=2)
    axes[3].errorbar(t_data, exenatide_data, yerr=np.array(data["MackStudy1"]["ExenatideEI_stdev"][0:7], dtype=float)/3, fmt='none', color=colors["exenatide"], capsize=8, elinewidth=2, capthick=2)
    axes[3].set_title("Wild et al.\n(IC50 = 0.65 nmol/L)", fontsize=18)
    axes[3].grid(True)
    #  LAYOUT
    for j in range(4):
        axes[j].set_xlim(0,23)
        axes[j].set_xticks([0,4,8,12,16,20])
        axes[j].set_ylim(-0.2,3.2)
        axes[j].set_xlabel("Time (hours)", fontsize=18)
        axes[j].plot([0,16],[-0.2,-0.2], color='black', lw=8, solid_capstyle='butt')
        axes[j].tick_params(axis='both', which='major', labelsize=18)
        for k in range(len(edges)-1):
            if IsNight[edges[k]]:
                axes[j].axvspan(time_hours_dn[edges[k]], time_hours_dn[edges[k+1]], color="black", alpha=0.08, linewidth=0)
    axes[0].set_yticks([0,1,2,3])
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    axes[3].tick_params(labelleft=False)
    fig.suptitle("Starved female NIH swiss", fontsize=18)
    fig.subplots_adjust(wspace=0.02)
    fig.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    # ---------- Fig. 2B: Exposure / IC50 ----------
    fig, axes = plt.subplots(3, 1, figsize=(12,8))
    axes[0].axis('off')
    axes[1].axis('off')
    ax_c1 = axes[2]
    ax_c1.fill_between(time_hours, np.maximum(C1_exenatide_min, 1e-6), np.maximum(C1_exenatide_max, 1e-6), color=colors["exenatide"], alpha=0.2, edgecolor="none")
    ax_c1.plot(time_hours, np.maximum(MackSimStudy1_IC50_1["best_sim"]["exenatide"].feature_values[:, idx_c1], 1e-6), color=colors["exenatide"], lw=2)
    ax_c1.axhline(y=0.00205, color="black", linestyle="--", linewidth=1)
    ax_c1.axhline(y=0.02, color="black", linestyle="--", linewidth=1.5)
    ax_c1.axhline(y=0.093, color="black", linestyle="--", linewidth=1)
    ax_c1.axhline(y=0.65, color="black", linestyle="--", linewidth=1)
    ax_c1.set_xlabel("Time (hours)", fontsize=18)
    ax_c1.set_ylabel("Plasma exenatide\n(nmol/L)", fontsize=18)
    ax_c1.set_title("Exposure and tested IC50 values", fontsize=18)
    ax_c1.set_yscale("log")
    ax_c1.set_xlim(0,23)
    ax_c1.set_ylim(0.0001,1)
    ax_c1.set_xticks([0,2,4,6,8,10,12,14,16,18,20,22])
    ax_c1.set_yticks([0.0001,0.01,1])
    ax_c1.plot([0,16],[0.0001,0.0001], color="black", lw=8, solid_capstyle="butt")
    ax_c1.tick_params(axis='both', which='major', labelsize=18)
    ax_c1.text(24.4,0.00205,"Iwasaki et al. IC50",va="center",fontsize=16)
    ax_c1.text(24.4,0.02,"Our best estimate",va="center",fontsize=16,fontweight="bold")
    ax_c1.text(24.4,0.093,"Thorens et al. IC50",va="center",fontsize=16)
    ax_c1.text(24.4,0.65,"Wild et al. IC50",va="center",fontsize=16)
    ax_c1.grid(True)
    fig.subplots_adjust(left=0.07,right=0.993,top=0.98,bottom=0.065,wspace=0.3,hspace=0.3)
    plt.show()


# Study 2
def plot_MackWeight_uncertainty(Model_Appetite, model_PK, model_GI, model_BW, data, OptimisedAppetiteParams,OptimisedBWparams, OptimisedGIparams,OptimisedPKParams, FastPlot):

    OptimisedAppetiteParams = OptimisedAppetiteParams.to_numpy(dtype=float)
    OptimisedBWparams       = OptimisedBWparams.to_numpy(dtype=float)
    OptimisedGIparams       = OptimisedGIparams.to_numpy(dtype=float)
    OptimisedPKParams       = OptimisedPKParams.to_numpy(dtype=float)

    Best_AppetiteParams = np.asarray(OptimisedAppetiteParams[0][:-1], dtype=float)
    Best_BWparams       = np.asarray(OptimisedBWparams[0][:-1], dtype=float)
    Best_GIparams       = np.asarray(OptimisedGIparams[0][:-1], dtype=float)
    Best_PKParams       = np.asarray(OptimisedPKParams[0][:-1], dtype=float)

    def weekly_cumulative_energy_intake_from_ei_ac(ei_ac, time, week_length=7):
        weekly_ei = np.zeros_like(ei_ac)
        week_idx = np.floor(time / week_length).astype(int)
        for w in np.unique(week_idx):
            idx = np.where(week_idx == w)[0]
            weekly_ei[idx] = ei_ac[idx] - ei_ac[idx[0]]
        return weekly_ei

    CalibratedActivity = 0.195; CalibratedStress = 3.1; CalibratedPotency = 0.02
    
    RecalculateCost = False # True or False, use to recalculate the validation cost for each parameter vector
    def _validate_param(VectorNumber):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, Best_PKParams, Best_GIparams, Best_BWparams, CalibratedActivity, CalibratedStress, CalibratedPotency)
            idx_BW = MackSimStudy2["vehicle"].feature_names.index("Body_weight")
            idx_EI_AC = MackSimStudy2["vehicle"].feature_names.index("EI_AC")
            sim_t = np.linspace(0, 28, (28 * 48*2) + 1)
            t_BW = np.asarray(data["MackStudy2"]["BW_Time"][0:5], dtype=float)
            t_EI = np.asarray(data["MackStudy2"]["EI_Time"][0:4], dtype=float)
            BW_vehicle = np.interp(t_BW, sim_t, MackSimStudy2["vehicle"].feature_values[:, idx_BW])
            BW_E3 = np.interp(t_BW, sim_t, MackSimStudy2["E3"].feature_values[:, idx_BW])
            BW_E10 = np.interp(t_BW, sim_t, MackSimStudy2["E10"].feature_values[:, idx_BW])
            BW_E30 = np.interp(t_BW, sim_t, MackSimStudy2["E30"].feature_values[:, idx_BW])
            EI_vehicle = weekly_cumulative_energy_intake_from_ei_ac(MackSimStudy2["vehicle"].feature_values[:, idx_EI_AC], sim_t)
            EI_E3 = weekly_cumulative_energy_intake_from_ei_ac(MackSimStudy2["E3"].feature_values[:, idx_EI_AC], sim_t)
            EI_E10 = weekly_cumulative_energy_intake_from_ei_ac(MackSimStudy2["E10"].feature_values[:, idx_EI_AC], sim_t)
            EI_E30 = weekly_cumulative_energy_intake_from_ei_ac(MackSimStudy2["E30"].feature_values[:, idx_EI_AC], sim_t)
            t_EI_sim = t_EI - (sim_t[1] - sim_t[0])
            EI_vehicle_atData = np.interp(t_EI_sim, sim_t, EI_vehicle)
            EI_E3_atData = np.interp(t_EI_sim, sim_t, EI_E3)
            EI_E10_atData = np.interp(t_EI_sim, sim_t, EI_E10)
            EI_E30_atData = np.interp(t_EI_sim, sim_t, EI_E30)
            SimulatedValues = np.concatenate([BW_vehicle, BW_E3, BW_E10, BW_E30, EI_vehicle_atData, EI_E3_atData, EI_E10_atData, EI_E30_atData])
            DataMean = np.asarray(data["MackStudy2"]["all_mean"], dtype=float)
            DataWeightedSD = np.asarray(data["MackStudy2"]["all_weighted_sd"], dtype=float)
            if len(SimulatedValues) != len(DataMean) or len(DataMean) != len(DataWeightedSD):
                raise ValueError(f"Length mismatch: simulation={len(SimulatedValues)}, mean={len(DataMean)}, SD={len(DataWeightedSD)}")
            CostValidation = np.sum(((DataMean - SimulatedValues) / DataWeightedSD) ** 2)
            return CostValidation
        except Exception as error:
            print(f"Vector {VectorNumber} failed: {error}")
            return np.inf
    if RecalculateCost:
        ValidationCosts = Parallel(n_jobs=1, verbose=10)(delayed(_validate_param)(VectorNumber) for VectorNumber in range(len(OptimisedAppetiteParams)))
        BestVectorNumber = int(np.argmin(ValidationCosts))
        BestValidationCost = float(ValidationCosts[BestVectorNumber])
        print(f"Best validation cost: {BestValidationCost}")
        print(f"Best parameter vector index: {BestVectorNumber}")


    n_select = 300
    if FastPlot:
        n_select = 10
    idx = np.linspace(0, OptimisedAppetiteParams.shape[0] - 1, n_select, dtype=int)
    OptimisedAppetiteParams = OptimisedAppetiteParams[idx]
    OptimisedBWparams       = OptimisedBWparams[idx]

    n_select = 500
    if FastPlot:
        n_select = 100
    idx = np.linspace(0, OptimisedPKParams.shape[0] - 1, n_select, dtype=int)
    OptimisedPKParams = OptimisedPKParams[idx]

    # No calibration
    first = True
    n_fail = 0
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, Best_PKParams, Best_GIparams, Best_BWparams,Best_BWparams[-1],1,0.00205) # IC50 appetite: 0.00205 nmol/L (PMID: 28698254); 0.093 nmol/L (PMID: 8405712); 0.65 nmol/L (PMID: 17138746); 0.24 lamda is the value used in the hall et al. 2011 paper (PMID: 21406690)
        except Exception as e:
            n_fail = n_fail + 1
            continue
        if first:
            idx_BW       = MackSimStudy2["vehicle"].feature_names.index("Body_weight")
            idx_EI_AC    = MackSimStudy2["vehicle"].feature_names.index("EI_AC")
            idx_E        = MackSimStudy2["vehicle"].feature_names.index("Plasma_Exenatide")
            idx_Appetite = MackSimStudy2["vehicle"].feature_names.index("Appetite")
            idx_Eating   = MackSimStudy2["vehicle"].feature_names.index("Eating")
            idx_lambda       = MackSimStudy2["vehicle"].feature_names.index("lambda")
            idx_lambda_delay = MackSimStudy2["vehicle"].feature_names.index("lambda_delay")
            time             = MackSimStudy2["vehicle"].time_vector
            # ===== Extract new curves =====
            BW_vehicle_new_NoCal = MackSimStudy2["vehicle"].feature_values[:, idx_BW]
            BW_E3_new_NoCal      = MackSimStudy2["E3"].feature_values[:, idx_BW]
            BW_E10_new_NoCal     = MackSimStudy2["E10"].feature_values[:, idx_BW]
            BW_E30_new_NoCal     = MackSimStudy2["E30"].feature_values[:, idx_BW]
            best_simAppetite_NoCal = MackSimStudy2
            # ===== BW =====
            BW_vehicle_min_NoCal,BW_vehicle_max_NoCal  = BW_vehicle_new_NoCal.copy(),BW_vehicle_new_NoCal.copy()
            BW_E3_min_NoCal,BW_E3_max_NoCal = BW_E3_new_NoCal.copy(),BW_E3_new_NoCal.copy()
            BW_E10_min_NoCal,BW_E10_max_NoCal = BW_E10_new_NoCal.copy(),BW_E10_new_NoCal.copy()
            BW_E30_min_NoCal,BW_E30_max_NoCal = BW_E30_new_NoCal.copy(),BW_E30_new_NoCal.copy()
            first = False
        else:
            # ===== Extract new curves =====
            BW_vehicle_new_NoCal = MackSimStudy2["vehicle"].feature_values[:, idx_BW]
            BW_E3_new_NoCal = MackSimStudy2["E3"].feature_values[:, idx_BW]
            BW_E10_new_NoCal = MackSimStudy2["E10"].feature_values[:, idx_BW]
            BW_E30_new_NoCal = MackSimStudy2["E30"].feature_values[:, idx_BW]
            # ===== BW =====
            BW_vehicle_min_NoCal = np.minimum(BW_vehicle_min_NoCal, BW_vehicle_new_NoCal)
            BW_vehicle_max_NoCal = np.maximum(BW_vehicle_max_NoCal, BW_vehicle_new_NoCal)
            BW_E3_min_NoCal = np.minimum(BW_E3_min_NoCal, BW_E3_new_NoCal)
            BW_E3_max_NoCal = np.maximum(BW_E3_max_NoCal, BW_E3_new_NoCal)
            BW_E10_min_NoCal = np.minimum(BW_E10_min_NoCal, BW_E10_new_NoCal)
            BW_E10_max_NoCal = np.maximum(BW_E10_max_NoCal, BW_E10_new_NoCal)
            BW_E30_min_NoCal = np.minimum(BW_E30_min_NoCal, BW_E30_new_NoCal)
            BW_E30_max_NoCal = np.maximum(BW_E30_max_NoCal, BW_E30_new_NoCal)
    print("1/6 done")
    print("Simulation failed:", n_fail)
    n_fail = 0

    # 1) IC50 = 0.00205 & Initial activity
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, Best_PKParams, Best_GIparams, Best_BWparams,CalibratedActivity,CalibratedStress,0.00205) # IC50 appetite: 0.00205 nmol/L (PMID: 28698254); 0.093 nmol/L (PMID: 8405712); 0.65 nmol/L (PMID: 17138746)
        except Exception as e:
            n_fail = n_fail + 1
            continue

        BW_vehicle_new_IC50_1 = MackSimStudy2["vehicle"].feature_values[:, idx_BW]
        BW_E3_new_IC50_1      = MackSimStudy2["E3"].feature_values[:, idx_BW]
        BW_E10_new_IC50_1     = MackSimStudy2["E10"].feature_values[:, idx_BW]
        BW_E30_new_IC50_1     = MackSimStudy2["E30"].feature_values[:, idx_BW]

        if first:
            # ===== BW =====
            BW_vehicle_min_IC50_1,BW_vehicle_max_IC50_1 = BW_vehicle_new_IC50_1.copy(),BW_vehicle_new_IC50_1.copy()
            BW_E3_min_IC50_1,BW_E3_max_IC50_1 = BW_E3_new_IC50_1.copy(),BW_E3_new_IC50_1.copy()
            BW_E10_min_IC50_1,BW_E10_max_IC50_1 = BW_E10_new_IC50_1.copy(),BW_E10_new_IC50_1.copy()
            BW_E30_min_IC50_1,BW_E30_max_IC50_1 = BW_E30_new_IC50_1.copy(),BW_E30_new_IC50_1.copy()
            best_simAppetite_IC50_1 = MackSimStudy2
            first = False
        else:
            # ===== BW =====
            BW_vehicle_min_IC50_1 = np.minimum(BW_vehicle_min_IC50_1, BW_vehicle_new_IC50_1)
            BW_vehicle_max_IC50_1 = np.maximum(BW_vehicle_max_IC50_1, BW_vehicle_new_IC50_1)
            BW_E3_min_IC50_1 = np.minimum(BW_E3_min_IC50_1, BW_E3_new_IC50_1)
            BW_E3_max_IC50_1 = np.maximum(BW_E3_max_IC50_1, BW_E3_new_IC50_1)
            BW_E10_min_IC50_1 = np.minimum(BW_E10_min_IC50_1, BW_E10_new_IC50_1)
            BW_E10_max_IC50_1 = np.maximum(BW_E10_max_IC50_1, BW_E10_new_IC50_1)
            BW_E30_min_IC50_1 = np.minimum(BW_E30_min_IC50_1, BW_E30_new_IC50_1)
            BW_E30_max_IC50_1 = np.maximum(BW_E30_max_IC50_1, BW_E30_new_IC50_1)
    print("2/6 done")
    print("Simulation failed:", n_fail)
    n_fail = 0
    
    # 2) IC50 = 0.093
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, Best_PKParams, Best_GIparams, Best_BWparams,CalibratedActivity,CalibratedStress,0.093) # IC50 appetite: 0.00205 nmol/L (PMID: 28698254); 0.093 nmol/L (PMID: 8405712); 0.65 nmol/L (PMID: 17138746)
        except Exception as e:
            n_fail = n_fail + 1
            continue
        BW_vehicle_new_IC50_2 = MackSimStudy2["vehicle"].feature_values[:, idx_BW]
        BW_E3_new_IC50_2      = MackSimStudy2["E3"].feature_values[:, idx_BW]
        BW_E10_new_IC50_2     = MackSimStudy2["E10"].feature_values[:, idx_BW]
        BW_E30_new_IC50_2     = MackSimStudy2["E30"].feature_values[:, idx_BW]
        if first:
            # ===== BW =====
            BW_vehicle_min_IC50_2,BW_vehicle_max_IC50_2 = BW_vehicle_new_IC50_2.copy(),BW_vehicle_new_IC50_2.copy()
            BW_E3_min_IC50_2,BW_E3_max_IC50_2 = BW_E3_new_IC50_2.copy(),BW_E3_new_IC50_2.copy()
            BW_E10_min_IC50_2,BW_E10_max_IC50_2 = BW_E10_new_IC50_2.copy(),BW_E10_new_IC50_2.copy()
            BW_E30_min_IC50_2,BW_E30_max_IC50_2 = BW_E30_new_IC50_2.copy(),BW_E30_new_IC50_2.copy()
            best_simAppetite_IC50_2 = MackSimStudy2
            first = False
        else:
            # ===== BW =====
            BW_vehicle_min_IC50_2 = np.minimum(BW_vehicle_min_IC50_2, BW_vehicle_new_IC50_2)
            BW_vehicle_max_IC50_2 = np.maximum(BW_vehicle_max_IC50_2, BW_vehicle_new_IC50_2)
            BW_E3_min_IC50_2 = np.minimum(BW_E3_min_IC50_2, BW_E3_new_IC50_2)
            BW_E3_max_IC50_2 = np.maximum(BW_E3_max_IC50_2, BW_E3_new_IC50_2)
            BW_E10_min_IC50_2 = np.minimum(BW_E10_min_IC50_2, BW_E10_new_IC50_2)
            BW_E10_max_IC50_2 = np.maximum(BW_E10_max_IC50_2, BW_E10_new_IC50_2)
            BW_E30_min_IC50_2 = np.minimum(BW_E30_min_IC50_2, BW_E30_new_IC50_2)
            BW_E30_max_IC50_2 = np.maximum(BW_E30_max_IC50_2, BW_E30_new_IC50_2)
    print("3/6 done")
    print("Simulation failed:", n_fail)
    n_fail = 0

    # 3) IC50 = 0.65
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, Best_PKParams, Best_GIparams, Best_BWparams,CalibratedActivity,CalibratedStress,0.65) # IC50 appetite: 0.00205 nmol/L (PMID: 28698254); 0.093 nmol/L (PMID: 8405712); 0.65 nmol/L (PMID: 17138746)
        except Exception as e:
            n_fail = n_fail + 1
            continue
        BW_vehicle_new_IC50_3 = MackSimStudy2["vehicle"].feature_values[:, idx_BW]
        BW_E3_new_IC50_3      = MackSimStudy2["E3"].feature_values[:, idx_BW]
        BW_E10_new_IC50_3     = MackSimStudy2["E10"].feature_values[:, idx_BW]
        BW_E30_new_IC50_3     = MackSimStudy2["E30"].feature_values[:, idx_BW]
        if first:
            # ===== BW =====
            BW_vehicle_min_IC50_3,BW_vehicle_max_IC50_3 = BW_vehicle_new_IC50_3.copy(),BW_vehicle_new_IC50_3.copy()
            BW_E3_min_IC50_3,BW_E3_max_IC50_3 = BW_E3_new_IC50_3.copy(),BW_E3_new_IC50_3.copy()
            BW_E10_min_IC50_3,BW_E10_max_IC50_3 = BW_E10_new_IC50_3.copy(),BW_E10_new_IC50_3.copy()
            BW_E30_min_IC50_3,BW_E30_max_IC50_3 = BW_E30_new_IC50_3.copy(),BW_E30_new_IC50_3.copy()
            best_simAppetite_IC50_3 = MackSimStudy2
            first = False
        else:
            # ===== BW =====
            BW_vehicle_min_IC50_3 = np.minimum(BW_vehicle_min_IC50_3, BW_vehicle_new_IC50_3)
            BW_vehicle_max_IC50_3 = np.maximum(BW_vehicle_max_IC50_3, BW_vehicle_new_IC50_3)
            BW_E3_min_IC50_3 = np.minimum(BW_E3_min_IC50_3, BW_E3_new_IC50_3)
            BW_E3_max_IC50_3 = np.maximum(BW_E3_max_IC50_3, BW_E3_new_IC50_3)
            BW_E10_min_IC50_3 = np.minimum(BW_E10_min_IC50_3, BW_E10_new_IC50_3)
            BW_E10_max_IC50_3 = np.maximum(BW_E10_max_IC50_3, BW_E10_new_IC50_3)
            BW_E30_min_IC50_3 = np.minimum(BW_E30_min_IC50_3, BW_E30_new_IC50_3)
            BW_E30_max_IC50_3 = np.maximum(BW_E30_max_IC50_3, BW_E30_new_IC50_3)
    print("4/6 done")
    print("Simulation failed:", n_fail)
    n_fail = 0

    # 4) our fitted IC50
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, Best_PKParams, Best_GIparams, Best_BWparams,CalibratedActivity,CalibratedStress,CalibratedPotency) # IC50 appetite: 0.00205 nmol/L (PMID: 28698254); 0.093 nmol/L (PMID: 8405712); 0.65 nmol/L (PMID: 17138746)
        except Exception as e:
            n_fail = n_fail + 1
            continue
        # ===== lambda_delay =====
        lambda_delay_vehicle_new_IC50_1 = MackSimStudy2["vehicle"].feature_values[:, idx_lambda_delay]
        lambda_delay_E3_new_IC50_1      = MackSimStudy2["E3"].feature_values[:, idx_lambda_delay]
        # ===== EI_AC =====
        EI_vehicle_new = MackSimStudy2["vehicle"].feature_values[:, idx_EI_AC]
        EI_E3_new = MackSimStudy2["E3"].feature_values[:, idx_EI_AC]
        EI_E10_new = MackSimStudy2["E10"].feature_values[:, idx_EI_AC]
        EI_E30_new = MackSimStudy2["E30"].feature_values[:, idx_EI_AC]
        # ===== BW =====
        BW_vehicle_new = MackSimStudy2["vehicle"].feature_values[:, idx_BW]
        BW_E3_new = MackSimStudy2["E3"].feature_values[:, idx_BW]
        BW_E10_new = MackSimStudy2["E10"].feature_values[:, idx_BW]
        BW_E30_new = MackSimStudy2["E30"].feature_values[:, idx_BW]
        # ===== Appetite =====
        App_vehicle_new = MackSimStudy2["vehicle"].feature_values[:, idx_Appetite]
        App_E3_new = MackSimStudy2["E3"].feature_values[:, idx_Appetite]
        App_E10_new = MackSimStudy2["E10"].feature_values[:, idx_Appetite]
        App_E30_new = MackSimStudy2["E30"].feature_values[:, idx_Appetite]
        # ===== Eating =====
        Eat_vehicle_new = MackSimStudy2["vehicle"].feature_values[:, idx_Eating]
        Eat_E3_new = MackSimStudy2["E3"].feature_values[:, idx_Eating]
        Eat_E10_new = MackSimStudy2["E10"].feature_values[:, idx_Eating]
        Eat_E30_new = MackSimStudy2["E30"].feature_values[:, idx_Eating]
        if first:
            best_simAppetite = MackSimStudy2
            # ===== lambda_delay =====
            lambda_delay_vehicle_min_IC50_1,lambda_delay_vehicle_max_IC50_1 = lambda_delay_vehicle_new_IC50_1.copy(),lambda_delay_vehicle_new_IC50_1.copy()
            lambda_delay_E3_min_IC50_1,lambda_delay_E3_max_IC50_1 = lambda_delay_E3_new_IC50_1.copy(),lambda_delay_E3_new_IC50_1.copy()
            # ===== EI_AC =====
            EI_vehicle_min,EI_vehicle_max = EI_vehicle_new.copy(),EI_vehicle_new.copy()
            EI_E3_min,EI_E3_max = EI_E3_new.copy(),EI_E3_new.copy()
            EI_E10_min,EI_E10_max = EI_E10_new.copy(),EI_E10_new.copy()
            EI_E30_min,EI_E30_max = EI_E30_new.copy(),EI_E30_new.copy()
            # ===== Appetite =====
            App_vehicle_min,App_vehicle_max = App_vehicle_new.copy(),App_vehicle_new.copy()
            App_E3_min,App_E3_max = App_E3_new.copy(),App_E3_new.copy()
            App_E10_min,App_E10_max = App_E10_new.copy(),App_E10_new.copy()
            App_E30_min,App_E30_max = App_E30_new.copy(),App_E30_new.copy()
            # ===== Eating =====
            Eat_vehicle_min,Eat_vehicle_max = Eat_vehicle_new.copy(),Eat_vehicle_new.copy()
            Eat_E3_min,Eat_E3_max = Eat_E3_new.copy(),Eat_E3_new.copy()
            Eat_E10_min,Eat_E10_max = Eat_E10_new.copy(),Eat_E10_new.copy()
            Eat_E30_min,Eat_E30_max = Eat_E30_new.copy(),Eat_E30_new.copy()
            # ===== BW =====
            BW_vehicle_min,BW_vehicle_max = BW_vehicle_new.copy(),BW_vehicle_new.copy()
            BW_E3_min,BW_E3_max = BW_E3_new.copy(),BW_E3_new.copy()
            BW_E10_min,BW_E10_max = BW_E10_new.copy(),BW_E10_new.copy()
            BW_E30_min,BW_E30_max = BW_E30_new.copy(),BW_E30_new.copy()
            first = False
        else:
            # ----- lambda_delay -----
            lambda_delay_vehicle_min_IC50_1 = np.minimum(lambda_delay_vehicle_min_IC50_1, lambda_delay_vehicle_new_IC50_1)
            lambda_delay_vehicle_max_IC50_1 = np.maximum(lambda_delay_vehicle_max_IC50_1, lambda_delay_vehicle_new_IC50_1)
            lambda_delay_E3_min_IC50_1 = np.minimum(lambda_delay_E3_min_IC50_1, lambda_delay_E3_new_IC50_1)
            lambda_delay_E3_max_IC50_1 = np.maximum(lambda_delay_E3_max_IC50_1, lambda_delay_E3_new_IC50_1)
            # ===== EI_AC =====
            EI_vehicle_min = np.minimum(EI_vehicle_min, EI_vehicle_new)
            EI_vehicle_max = np.maximum(EI_vehicle_max, EI_vehicle_new)
            EI_E3_min = np.minimum(EI_E3_min, EI_E3_new)
            EI_E3_max = np.maximum(EI_E3_max, EI_E3_new)
            EI_E10_min = np.minimum(EI_E10_min, EI_E10_new)
            EI_E10_max = np.maximum(EI_E10_max, EI_E10_new)
            EI_E30_min = np.minimum(EI_E30_min, EI_E30_new)
            EI_E30_max = np.maximum(EI_E30_max, EI_E30_new)
            # ===== Appetite =====
            App_vehicle_min = np.minimum(App_vehicle_min, App_vehicle_new)
            App_vehicle_max = np.maximum(App_vehicle_max, App_vehicle_new)
            App_E3_min = np.minimum(App_E3_min, App_E3_new)
            App_E3_max = np.maximum(App_E3_max, App_E3_new)
            App_E10_min = np.minimum(App_E10_min, App_E10_new)
            App_E10_max = np.maximum(App_E10_max, App_E10_new)
            App_E30_min = np.minimum(App_E30_min, App_E30_new)
            App_E30_max = np.maximum(App_E30_max, App_E30_new)
            # ===== Eating =====
            Eat_vehicle_min = np.minimum(Eat_vehicle_min, Eat_vehicle_new)
            Eat_vehicle_max = np.maximum(Eat_vehicle_max, Eat_vehicle_new)
            Eat_E3_min = np.minimum(Eat_E3_min, Eat_E3_new)
            Eat_E3_max = np.maximum(Eat_E3_max, Eat_E3_new)
            Eat_E10_min = np.minimum(Eat_E10_min, Eat_E10_new)
            Eat_E10_max = np.maximum(Eat_E10_max, Eat_E10_new)
            Eat_E30_min = np.minimum(Eat_E30_min, Eat_E30_new)
            Eat_E30_max = np.maximum(Eat_E30_max, Eat_E30_new)
            # ===== BW =====
            BW_vehicle_min = np.minimum(BW_vehicle_min, BW_vehicle_new)
            BW_vehicle_max = np.maximum(BW_vehicle_max, BW_vehicle_new)
            BW_E3_min = np.minimum(BW_E3_min, BW_E3_new)
            BW_E3_max = np.maximum(BW_E3_max, BW_E3_new)
            BW_E10_min = np.minimum(BW_E10_min, BW_E10_new)
            BW_E10_max = np.maximum(BW_E10_max, BW_E10_new)
            BW_E30_min = np.minimum(BW_E30_min, BW_E30_new)
            BW_E30_max = np.maximum(BW_E30_max, BW_E30_new)
    print("5/6 done")
    print("Simulation failed:", n_fail)
    n_fail = 0

    # PK uncertainty
    first = True
    for VectorNumber in range(len(OptimisedPKParams)):
        PKParams       = np.asarray(OptimisedPKParams[VectorNumber][:-1], dtype=float)
        try:
            MackSimStudy2 = Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, Best_AppetiteParams, PKParams, Best_GIparams, Best_BWparams,CalibratedActivity,CalibratedStress,CalibratedPotency) # IC50 appetite: 0.00205 nmol/L (PMID: 28698254); 0.093 nmol/L (PMID: 8405712); 0.65 nmol/L (PMID: 17138746)
        except Exception as e:
            n_fail = n_fail + 1
            continue
        # ===== Extract new curves =====
        E_vehicle_new = MackSimStudy2["vehicle"].feature_values[:, idx_E]
        E_E3_new = MackSimStudy2["E3"].feature_values[:, idx_E]
        E_E10_new = MackSimStudy2["E10"].feature_values[:, idx_E]
        E_E30_new = MackSimStudy2["E30"].feature_values[:, idx_E]    
        if first:
            # ===== Exenatide =====
            E_vehicle_min,E_vehicle_max = E_vehicle_new.copy(),E_vehicle_new.copy()
            E_E3_min,E_E3_max = E_E3_new.copy(),E_E3_new.copy()
            E_E10_min,E_E10_max = E_E10_new.copy(),E_E10_new.copy()
            E_E30_min,E_E30_max = E_E30_new.copy(),E_E30_new.copy()
            first = False
        else:
            # ===== Exenatide =====
            E_vehicle_min = np.minimum(E_vehicle_min, E_vehicle_new)
            E_vehicle_max = np.maximum(E_vehicle_max, E_vehicle_new)
            E_E3_min = np.minimum(E_E3_min, E_E3_new)
            E_E3_max = np.maximum(E_E3_max, E_E3_new)
            E_E10_min = np.minimum(E_E10_min, E_E10_new)
            E_E10_max = np.maximum(E_E10_max, E_E10_new)
            E_E30_min = np.minimum(E_E30_min, E_E30_new)
            E_E30_max = np.maximum(E_E30_max, E_E30_new)
    print("6/6 done")
    print("Simulation failed:", n_fail)
    
    def compute_is_night(time, phase_shift_hours=16):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        IsNight = cycle > 0
        return IsNight
    IsNight = compute_is_night(time)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time)-1]))


    t_BW_data = np.array(data["MackStudy2"]["BW_Time"][0:5], dtype=float)

    colors = {
        "Vehicle": "blue",
        "E3": "teal",
        "E10": "goldenrod",
        "E30": "firebrick"
        }
    
    # ============================================================
    # Figure 1: Activity calibration
    # ============================================================
    fig1, axes1 = plt.subplots(2, 3, figsize=(18,5))
    ax_NoCal  = axes1[0,0]
    ax_IC50_1 = axes1[0,1]
    ax_lambda = axes1[0,2]
    # COLUMN 1: NoCal, VehicleBW only 
    ax_NoCal.fill_between(time,BW_vehicle_min_NoCal,BW_vehicle_max_NoCal,color=colors["Vehicle"],alpha=0.2, edgecolor="none")
    ax_NoCal.plot(time,best_simAppetite_NoCal["vehicle"].feature_values[:,idx_BW],color=colors["Vehicle"],lw=2,label="Vehicle")
    ax_NoCal.errorbar(t_BW_data[0:5],np.array(data["MackStudy2"]["VehicleBW"][0:5],dtype=float),yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][0:5],dtype=float),fmt='none',color=colors["Vehicle"],capsize=8,elinewidth=2,capthick=2,markersize=8)
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_NoCal.axvspan(time[edges[i]],time[edges[i+1]],color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    # COLUMN 2: IC50_1, VehicleBW only
    ax_IC50_1.fill_between(time,BW_vehicle_min_IC50_1,BW_vehicle_max_IC50_1,color=colors["Vehicle"],alpha=0.2, edgecolor="none")
    ax_IC50_1.plot(time,best_simAppetite_IC50_1["vehicle"].feature_values[:,idx_BW],color=colors["Vehicle"],lw=2,label="Simulation")
    ax_IC50_1.errorbar(t_BW_data[0:5],np.array(data["MackStudy2"]["VehicleBW"][0:5],dtype=float),yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][0:5],dtype=float),fmt='none',color=colors["Vehicle"],capsize=8,elinewidth=2,capthick=2,markersize=8,label="Predicted data")
    ax_IC50_1.errorbar(t_BW_data[1],np.array(data["MackStudy2"]["VehicleBW"][1],dtype=float),yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][1],dtype=float),fmt='x',color="black",capsize=0,elinewidth=0,capthick=0,markersize=18,markeredgewidth=4,label="Calibration: Initial stress")
    ax_IC50_1.errorbar(t_BW_data[-1],np.array(data["MackStudy2"]["VehicleBW"][-1],dtype=float),yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][-1],dtype=float),fmt='+',color="black",capsize=0,elinewidth=0,capthick=0,markersize=18,markeredgewidth=4,label="Calibration: Physical activity")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_IC50_1.axvspan(time[edges[i]],time[edges[i+1]],color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    # COLUMN 3: Activity calibration dynamics
    ax_lambda.axhline(y=Best_BWparams[-1], color="black", linestyle="--", linewidth=2, label="Used in blind prediction (Fig. S3A i)")
    ax_lambda.fill_between(time, lambda_delay_vehicle_min_IC50_1,  lambda_delay_vehicle_max_IC50_1, color=colors["Vehicle"],  alpha=0.15, linewidth=0)
    ax_lambda.plot(time,best_simAppetite_IC50_1["vehicle"].feature_values[:,idx_lambda_delay], lw=2,color=colors["Vehicle"],ls="--", label="Adaptation rate (from estimation data)")
    ax_lambda.errorbar([0],max(lambda_delay_vehicle_max_IC50_1),[0],fmt='x',color="black",capsize=0,elinewidth=0,capthick=0,markersize=18,markeredgewidth=4,label="Calibration: Initial stress")
    ax_lambda.errorbar([28],best_simAppetite_IC50_1["vehicle"].feature_values[-1,idx_lambda],[0],fmt='+',color="black",capsize=0,elinewidth=0,capthick=0,markersize=18,markeredgewidth=4,label="Calibration: Physical activity")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_lambda.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    # Formatting
    titles = ["Physical activity assumed\nfrom Guo and Hall HFD group", "Calibrating activity to vehicle group", "Calibration visualised"]
    row1_axes = [ax_NoCal,ax_IC50_1]
    for i in range(2):
        row1_axes[i].set_title(titles[i],fontsize=18)
        row1_axes[i].set_xlim(-1,29)
        row1_axes[i].set_ylim(20,40)
        row1_axes[i].set_xticks([0,7,14,21,28])
        row1_axes[i].set_yticks([20,25,30,35,40])
        row1_axes[i].set_xlabel("Time (days)",fontsize=18)
        row1_axes[i].grid(True)
        row1_axes[i].tick_params(axis='both',labelsize=18)
        row1_axes[i].set_ylabel("Total body mass\n(g)",fontsize=18)
    row1_axes[1].legend(loc="upper right",fontsize=12,framealpha=1)
    ax_lambda.set_title(titles[2],fontsize=18)
    ax_lambda.set_xlim(-1,29)
    ax_lambda.set_xticks([0,7,14,21,28])
    ax_lambda.set_ylim(0,1)
    ax_lambda.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax_lambda.set_xlabel("Time (days)",fontsize=18)
    ax_lambda.set_ylabel("Energy expendature\n(kcal/g/day)",fontsize=18)
    ax_lambda.grid(True)
    ax_lambda.tick_params(axis='both',labelsize=18)
    ax_lambda.legend(loc="upper right",fontsize=12,framealpha=1)
    fig1.tight_layout()
    plt.show()



    # ============================================================
    # Figure: BW comparison across IC50 values
    # ============================================================
    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 4), sharey=True)
    ax_IC50_1 = axes2[0]
    ax_IC50_2 = axes2[1]
    ax_IC50_3 = axes2[2]
    ax_FullCal = axes2[3]
    # COLUMN 1: IC50 = 0.00205 nmol/L
    ax_IC50_1.fill_between(time, BW_vehicle_min_IC50_1, BW_vehicle_max_IC50_1, color=colors["Vehicle"], alpha=0.2, edgecolor="none")
    ax_IC50_1.plot(time, best_simAppetite_IC50_1["vehicle"].feature_values[:, idx_BW], color=colors["Vehicle"], lw=2, label="Vehicle")
    ax_IC50_1.fill_between(time, BW_E3_min_IC50_1, BW_E3_max_IC50_1, color=colors["E3"], alpha=0.2, edgecolor="none")
    ax_IC50_1.plot(time, best_simAppetite_IC50_1["E3"].feature_values[:, idx_BW], color=colors["E3"], lw=2, label="Exenatide 3 µg/kg/day")
    ax_IC50_1.fill_between(time, BW_E10_min_IC50_1, BW_E10_max_IC50_1, color=colors["E10"], alpha=0.2, edgecolor="none")
    ax_IC50_1.plot(time, best_simAppetite_IC50_1["E10"].feature_values[:, idx_BW], color=colors["E10"], lw=2, label="Exenatide 10 µg/kg/day")
    ax_IC50_1.fill_between(time, BW_E30_min_IC50_1, BW_E30_max_IC50_1, color=colors["E30"], alpha=0.2, edgecolor="none")
    ax_IC50_1.plot(time, best_simAppetite_IC50_1["E30"].feature_values[:, idx_BW], color=colors["E30"], lw=2, label="Exenatide 30 µg/kg/day")
    ax_IC50_1.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["VehicleBW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][0:5], dtype=float), fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_1.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E3BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E3BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E3"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_1.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E10BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E10BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E10"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_1.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E30BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E30BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E30"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_1.errorbar(t_BW_data[1], np.array(data["MackStudy2"]["VehicleBW"][1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5, label="Physical activity calibration")
    ax_IC50_1.errorbar(t_BW_data[-1], np.array(data["MackStudy2"]["VehicleBW"][-1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][-1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5, label="Physical activity calibration")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_IC50_1.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    # COLUMN 2: IC50 = 0.093 nmol/L
    ax_IC50_2.fill_between(time, BW_vehicle_min_IC50_2, BW_vehicle_max_IC50_2, color=colors["Vehicle"], alpha=0.2, edgecolor="none")
    ax_IC50_2.plot(time, best_simAppetite_IC50_2["vehicle"].feature_values[:, idx_BW], color=colors["Vehicle"], lw=2, label="Vehicle")
    ax_IC50_2.fill_between(time, BW_E3_min_IC50_2, BW_E3_max_IC50_2, color=colors["E3"], alpha=0.2, edgecolor="none")
    ax_IC50_2.plot(time, best_simAppetite_IC50_2["E3"].feature_values[:, idx_BW], color=colors["E3"], lw=2, label="Exenatide 3 µg/kg/day")
    ax_IC50_2.fill_between(time, BW_E10_min_IC50_2, BW_E10_max_IC50_2, color=colors["E10"], alpha=0.2, edgecolor="none")
    ax_IC50_2.plot(time, best_simAppetite_IC50_2["E10"].feature_values[:, idx_BW], color=colors["E10"], lw=2, label="Exenatide 10 µg/kg/day")
    ax_IC50_2.fill_between(time, BW_E30_min_IC50_2, BW_E30_max_IC50_2, color=colors["E30"], alpha=0.2, edgecolor="none")
    ax_IC50_2.plot(time, best_simAppetite_IC50_2["E30"].feature_values[:, idx_BW], color=colors["E30"], lw=2, label="Exenatide 30 µg/kg/day")
    ax_IC50_2.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["VehicleBW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][0:5], dtype=float), fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_2.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E3BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E3BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E3"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_2.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E10BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E10BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E10"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_2.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E30BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E30BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E30"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_2.errorbar(t_BW_data[1], np.array(data["MackStudy2"]["VehicleBW"][1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5)
    ax_IC50_2.errorbar(t_BW_data[-1], np.array(data["MackStudy2"]["VehicleBW"][-1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][-1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5, label="Physical activity calibration")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_IC50_2.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    # COLUMN 3: IC50 = 0.65 nmol/L
    ax_IC50_3.fill_between(time, BW_vehicle_min_IC50_3, BW_vehicle_max_IC50_3, color=colors["Vehicle"], alpha=0.2, edgecolor="none")
    ax_IC50_3.plot(time, best_simAppetite_IC50_3["vehicle"].feature_values[:, idx_BW], color=colors["Vehicle"], lw=2, label="Vehicle")
    ax_IC50_3.fill_between(time, BW_E3_min_IC50_3, BW_E3_max_IC50_3, color=colors["E3"], alpha=0.2, edgecolor="none")
    ax_IC50_3.plot(time, best_simAppetite_IC50_3["E3"].feature_values[:, idx_BW], color=colors["E3"], lw=2, label="Exenatide 3 µg/kg/day")
    ax_IC50_3.fill_between(time, BW_E10_min_IC50_3, BW_E10_max_IC50_3, color=colors["E10"], alpha=0.2, edgecolor="none")
    ax_IC50_3.plot(time, best_simAppetite_IC50_3["E10"].feature_values[:, idx_BW], color=colors["E10"], lw=2, label="Exenatide 10 µg/kg/day")
    ax_IC50_3.fill_between(time, BW_E30_min_IC50_3, BW_E30_max_IC50_3, color=colors["E30"], alpha=0.2, edgecolor="none")
    ax_IC50_3.plot(time, best_simAppetite_IC50_3["E30"].feature_values[:, idx_BW], color=colors["E30"], lw=2, label="Exenatide 30 µg/kg/day")
    ax_IC50_3.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["VehicleBW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][0:5], dtype=float), fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_3.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E3BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E3BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E3"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_3.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E10BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E10BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E10"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_3.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E30BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E30BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E30"], capsize=8, elinewidth=2, capthick=2)
    ax_IC50_3.errorbar(t_BW_data[1], np.array(data["MackStudy2"]["VehicleBW"][1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5)
    ax_IC50_3.errorbar(t_BW_data[-1], np.array(data["MackStudy2"]["VehicleBW"][-1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][-1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5, label="Physical activity calibration")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_IC50_3.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    # COLUMN 4: Calibrated IC50 = 0.02 nmol/L
    ax_FullCal.fill_between(time, BW_vehicle_min, BW_vehicle_max, color=colors["Vehicle"], alpha=0.2, edgecolor="none")
    ax_FullCal.plot(time, best_simAppetite["vehicle"].feature_values[:, idx_BW], color=colors["Vehicle"], lw=2, label="Vehicle")
    ax_FullCal.fill_between(time, BW_E3_min, BW_E3_max, color=colors["E3"], alpha=0.2, edgecolor="none")
    ax_FullCal.plot(time, best_simAppetite["E3"].feature_values[:, idx_BW], color=colors["E3"], lw=2, label="Exenatide 3 µg/kg/day")
    ax_FullCal.fill_between(time, BW_E10_min, BW_E10_max, color=colors["E10"], alpha=0.2, edgecolor="none")
    ax_FullCal.plot(time, best_simAppetite["E10"].feature_values[:, idx_BW], color=colors["E10"], lw=2, label="Exenatide 10 µg/kg/day")
    ax_FullCal.fill_between(time, BW_E30_min, BW_E30_max, color=colors["E30"], alpha=0.2, edgecolor="none")
    ax_FullCal.plot(time, best_simAppetite["E30"].feature_values[:, idx_BW], color=colors["E30"], lw=2, label="Exenatide 30 µg/kg/day")
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["VehicleBW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][0:5], dtype=float), fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2)
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E3BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E3BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E3"], capsize=8, elinewidth=2, capthick=2)
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E10BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E10BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E10"], capsize=8, elinewidth=2, capthick=2)
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E30BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E30BW_stdev"][0:5], dtype=float), fmt='none', color=colors["E30"], capsize=8, elinewidth=2, capthick=2)
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E3BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E3BW_stdev"][0:5], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5)
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E10BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E10BW_stdev"][0:5], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5)
    ax_FullCal.errorbar(t_BW_data[0:5], np.array(data["MackStudy2"]["E30BW"][0:5], dtype=float), yerr=np.array(data["MackStudy2"]["E30BW_stdev"][0:5], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5)
    ax_FullCal.errorbar(t_BW_data[-1], np.array(data["MackStudy2"]["VehicleBW"][-1], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleBW_stdev"][-1], dtype=float), fmt='x', color="black", capsize=0, elinewidth=0, capthick=0, markersize=18, markeredgewidth=2.5, label="Physical activity calibration")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_FullCal.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    # FORMATTING
    titles = [
        "IC50 = 0.00205 nmol/L\nIwasaki et al. (PMID: 28698254)",
        "IC50 = 0.093 nmol/L\nThorens et al. (PMID: 8405712)",
        "IC50 = 0.65 nmol/L\nWild et al. (PMID: 17138746)",
        "IC50 = 0.02 nmol/L\nOur best estimate (fitted to data)"
    ]
    row_axes = [ax_IC50_1, ax_IC50_2, ax_IC50_3, ax_FullCal]
    for i in range(4):
        row_axes[i].set_title(titles[i], fontsize=18)
        row_axes[i].set_xlim(-1,29)
        row_axes[i].set_ylim(20,34)
        row_axes[i].set_xticks([0,7,14,21,28])
        row_axes[i].set_yticks([20,24,28,32])
        row_axes[i].set_xlabel("Time (days)", fontsize=18)
        row_axes[i].grid(True)
        row_axes[i].tick_params(axis="both", labelsize=18)
        row_axes[i].plot([0,28],[20,20], color='black', lw=6, solid_capstyle='butt')
    ax_IC50_1.set_ylabel("Total body mass\n(g)", fontsize=18)
    ax_IC50_2.tick_params(labelleft=False)
    ax_IC50_3.tick_params(labelleft=False)
    ax_FullCal.tick_params(labelleft=False)
    fig2.tight_layout()
    plt.show()


    # Figure: cumulative EI and exenatide exposure
    EI_vehicle_plot = weekly_cumulative_energy_intake_from_ei_ac(best_simAppetite["vehicle"].feature_values[:, idx_EI_AC], time)
    EI_E3_plot = weekly_cumulative_energy_intake_from_ei_ac(best_simAppetite["E3"].feature_values[:, idx_EI_AC], time)
    EI_E10_plot = weekly_cumulative_energy_intake_from_ei_ac(best_simAppetite["E10"].feature_values[:, idx_EI_AC], time)
    EI_E30_plot = weekly_cumulative_energy_intake_from_ei_ac(best_simAppetite["E30"].feature_values[:, idx_EI_AC], time)
    EI_vehicle_min_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_vehicle_min, time)
    EI_vehicle_max_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_vehicle_max, time)
    EI_E3_min_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_E3_min, time)
    EI_E3_max_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_E3_max, time)
    EI_E10_min_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_E10_min, time)
    EI_E10_max_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_E10_max, time)
    EI_E30_min_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_E30_min, time)
    EI_E30_max_plot = weekly_cumulative_energy_intake_from_ei_ac(EI_E30_max, time)
    fig, axes_row1 = plt.subplots(1, 2, figsize=(16,5))
    # Column 1: weekly cumulative EI
    axes_row1[0].fill_between(time[:-1], EI_vehicle_min_plot[:-1], EI_vehicle_max_plot[:-1], color=colors["Vehicle"], alpha=0.2, edgecolor="none")
    axes_row1[0].plot(time[:-1], EI_vehicle_plot[:-1], color=colors["Vehicle"], lw=2, label="Vehicle")
    axes_row1[0].errorbar([7,14,21,28], np.array(data["MackStudy2"]["VehicleEI"][0:4], dtype=float), yerr=np.array(data["MackStudy2"]["VehicleEI_stdev"][0:4], dtype=float), fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2)
    axes_row1[0].fill_between(time[:-1], EI_E3_min_plot[:-1], EI_E3_max_plot[:-1], color=colors["E3"], alpha=0.2, edgecolor="none")
    axes_row1[0].plot(time[:-1], EI_E3_plot[:-1], color=colors["E3"], lw=2, label="Exenatide 3 µg/kg/day")
    axes_row1[0].errorbar([7,14,21,28], np.array(data["MackStudy2"]["E3EI"][0:4], dtype=float), yerr=np.array(data["MackStudy2"]["E3EI_stdev"][0:4], dtype=float), fmt='none', color=colors["E3"], capsize=8, elinewidth=2, capthick=2)
    axes_row1[0].fill_between(time[:-1], EI_E10_min_plot[:-1], EI_E10_max_plot[:-1], color=colors["E10"], alpha=0.2, edgecolor="none")
    axes_row1[0].plot(time[:-1], EI_E10_plot[:-1], color=colors["E10"], lw=2, label="Exenatide 10 µg/kg/day")
    axes_row1[0].errorbar([7,14,21,28], np.array(data["MackStudy2"]["E10EI"][0:4], dtype=float), yerr=np.array(data["MackStudy2"]["E10EI_stdev"][0:4], dtype=float), fmt='none', color=colors["E10"], capsize=8, elinewidth=2, capthick=2)
    axes_row1[0].fill_between(time[:-1], EI_E30_min_plot[:-1], EI_E30_max_plot[:-1], color=colors["E30"], alpha=0.2, edgecolor="none")
    axes_row1[0].plot(time[:-1], EI_E30_plot[:-1], color=colors["E30"], lw=2, label="Exenatide 30 µg/kg/day")
    axes_row1[0].errorbar([7,14,21,28], np.array(data["MackStudy2"]["E30EI"][0:4], dtype=float), yerr=np.array(data["MackStudy2"]["E30EI_stdev"][0:4], dtype=float), fmt='none', color=colors["E30"], capsize=8, elinewidth=2, capthick=2)
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            axes_row1[0].axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    axes_row1[0].set_xticks([0,7,14,21,28])
    axes_row1[0].set_xlabel("Time (days)", fontsize=18)
    axes_row1[0].set_ylabel("Weekly cumulative energy intake\n(kcal)", fontsize=18)
    axes_row1[0].tick_params(labelsize=18)
    axes_row1[0].grid(True)
    axes_row1[0].set_ylim(-5,120)
    axes_row1[0].set_yticks([0,25,50,75,100])
    axes_row1[0].plot([0,28],[-5,-5], color='black', lw=6, solid_capstyle='butt')
    #axes_row1[0].legend(loc="best", framealpha=1, fontsize=12)
    # Column 2: Exenatide exposure
    axes_row1[1].fill_between(time, np.maximum(E_vehicle_min, 1e-6), np.maximum(E_vehicle_max, 1e-6), color=colors["Vehicle"], alpha=0.2, edgecolor="none")
    axes_row1[1].plot(time, np.maximum(best_simAppetite["vehicle"].feature_values[:, idx_E], 1e-6), color=colors["Vehicle"], lw=2, label="Vehicle")
    axes_row1[1].fill_between(time, np.maximum(E_E3_min, 1e-6), np.maximum(E_E3_max, 1e-6), color=colors["E3"], alpha=0.2, edgecolor="none")
    axes_row1[1].plot(time, np.maximum(best_simAppetite["E3"].feature_values[:, idx_E], 1e-6), color=colors["E3"], lw=2, label="Exenatide 3 µg/kg/day")
    axes_row1[1].fill_between(time, np.maximum(E_E10_min, 1e-6), np.maximum(E_E10_max, 1e-6), color=colors["E10"], alpha=0.2, edgecolor="none")
    axes_row1[1].plot(time, np.maximum(best_simAppetite["E10"].feature_values[:, idx_E], 1e-6), color=colors["E10"], lw=2, label="Exenatide 10 µg/kg/day")
    axes_row1[1].fill_between(time, np.maximum(E_E30_min, 1e-6), np.maximum(E_E30_max, 1e-6), color=colors["E30"], alpha=0.2, edgecolor="none")
    axes_row1[1].plot(time, np.maximum(best_simAppetite["E30"].feature_values[:, idx_E], 1e-6), color=colors["E30"], lw=2, label="Exenatide 30 µg/kg/day")
    axes_row1[1].axhline(y=0.00205, color="black", linestyle="--", linewidth=1, label="Iwasaki et al. IC50")
    axes_row1[1].axhline(y=0.02, color="black", linestyle="--", linewidth=1.5, label="Our IC50 (appetite)")
    axes_row1[1].axhline(y=0.093, color="black", linestyle="--", linewidth=1, label="Thorens et al. IC50")
    axes_row1[1].axhline(y=0.65, color="black", linestyle="--", linewidth=1, label="Wild et al. IC50")
    axes_row1[1].text(29.2,0.00205,"Iwasaki et al. IC50",va="center",fontsize=12)
    axes_row1[1].text(29.2,0.02,"Our IC50 (Appetite)",va="center",fontsize=12,fontweight="bold")
    axes_row1[1].text(29.2,0.11,"Thorens et al. IC50",va="center",fontsize=12)
    axes_row1[1].text(29.2,0.65,"Wild et al. IC50",va="center",fontsize=12)
    axes_row1[1].text(29.2,1.5,"Our IC50 (Insulin secretion)",va="center",fontsize=12,fontweight="bold")
    axes_row1[1].set_xlabel("Time (days)", fontsize=18)
    axes_row1[1].set_ylabel("Plasma exenatide\n(nmol/L)", fontsize=18)
    axes_row1[1].set_title("Exposure and tested IC50 values", fontsize=18)
    axes_row1[1].set_yscale("log")
    axes_row1[1].set_xlim(-1,29)
    axes_row1[1].set_ylim(0.001,0.7)
    axes_row1[1].set_xticks([0,7,14,21,28])
    axes_row1[1].set_yticks([0.001,0.01,0.1])
    axes_row1[1].plot([0,28],[0.001,0.001], color='black', lw=6, solid_capstyle='butt')
    axes_row1[1].tick_params(axis='both', which='major', labelsize=18)
    axes_row1[1].grid(True)
    fig.subplots_adjust(left=0.07,right=0.89,top=0.95,bottom=0.065,wspace=0.3,hspace=0.3)
    plt.show()


# --- He study ---
def plot_He_uncertainty(Model_Appetite, model_GI, model_BW, data, OptimisedAppetiteParams,OptimisedHeparams, OptimisedBWparams, FastPlot):
    
    OptimisedAppetiteParams = OptimisedAppetiteParams.to_numpy(dtype=float)
    OptimisedBWparams       = OptimisedBWparams.to_numpy(dtype=float)
    OptimisedHeparams       = OptimisedHeparams.to_numpy(dtype=float)

    n_select = 300
    if FastPlot:
        n_select = 10
    idx = np.linspace(0, OptimisedAppetiteParams.shape[0] - 1, n_select, dtype=int)
    OptimisedAppetiteParams = OptimisedAppetiteParams[idx]
    OptimisedHeparams       = OptimisedHeparams[idx]
    OptimisedBWparams       = OptimisedBWparams[idx]

    TimeRes = 24

    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):

        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        Heparams       = np.asarray(OptimisedHeparams[VectorNumber][:-1], dtype=float)
        BWParams       = np.asarray(OptimisedBWparams[VectorNumber][:-1], dtype=float)
        try:
            Hestudy = Simulate_HEStudy(Model_Appetite, model_GI, model_BW, data, AppetiteParams, Heparams, BWParams,TimeRes)
        except Exception as e:
            continue
        idx_BW       = Hestudy["control"].feature_names.index("Body_weight")
        idx_G        = Hestudy["control"].feature_names.index("Plasma_Glucose")
        idx_U_id     = Hestudy["control"].feature_names.index("f_IR_Uid")
        idx_EGP      = Hestudy["control"].feature_names.index("f_IR_EGP")
        idx_Ins      = Hestudy["control"].feature_names.index("f_IR_Ins")
        time         = Hestudy["control"].time_vector
        # BW
        BW_vehicle_new  = Hestudy["control"].feature_values[:, idx_BW]
        BW_HFD_new      = Hestudy["HFD"].feature_values[:, idx_BW]
        # Glucose
        def daily_min(values, points_per_day):
            values = np.asarray(values, dtype=float)
            n_days = (len(values) - 1) // points_per_day
            return values[:n_days*points_per_day].reshape(n_days, points_per_day).min(axis=1)
        G_vehicle_new = daily_min(Hestudy["control"].feature_values[:, idx_G],TimeRes)
        G_HFD_new     = daily_min(Hestudy["HFD"].feature_values[:, idx_G],TimeRes)
        # U_id
        Uid_vehicle_new = 1/(Hestudy["control"].feature_values[:, idx_U_id])
        Uid_HFD_new     = 1/(Hestudy["HFD"].feature_values[:, idx_U_id])
        # Ins
        Ins_vehicle_new = Hestudy["control"].feature_values[:, idx_Ins]
        Ins_HFD_new     = Hestudy["HFD"].feature_values[:, idx_Ins]
        # EGP
        EGP_vehicle_new = Hestudy["control"].feature_values[:, idx_EGP]
        EGP_HFD_new     = Hestudy["HFD"].feature_values[:, idx_EGP]

        if first:
            best_sim = Hestudy
            # BW
            BW_vehicle_min  = BW_vehicle_new.copy()
            BW_vehicle_max  = BW_vehicle_new.copy()
            BW_HFD_min      = BW_HFD_new.copy()
            BW_HFD_max      = BW_HFD_new.copy()
            # Glucose
            G_vehicle_min = G_vehicle_new.copy()
            G_vehicle_max = G_vehicle_new.copy()
            G_HFD_min      = G_HFD_new.copy()
            G_HFD_max      = G_HFD_new.copy()
            # U_id
            Uid_vehicle_min = Uid_vehicle_new.copy()
            Uid_vehicle_max = Uid_vehicle_new.copy()
            Uid_HFD_min     = Uid_HFD_new.copy()
            Uid_HFD_max     = Uid_HFD_new.copy()
            # Ins
            Ins_vehicle_min = Ins_vehicle_new.copy()
            Ins_vehicle_max = Ins_vehicle_new.copy()
            Ins_HFD_min      = Ins_HFD_new.copy()
            Ins_HFD_max      = Ins_HFD_new.copy()
            # EGP
            EGP_vehicle_min = EGP_vehicle_new.copy()
            EGP_vehicle_max = EGP_vehicle_new.copy()
            EGP_HFD_min      = EGP_HFD_new.copy()
            EGP_HFD_max      = EGP_HFD_new.copy()
            first = False
        else:
            # BW
            BW_vehicle_min = np.minimum(BW_vehicle_min, BW_vehicle_new)
            BW_vehicle_max = np.maximum(BW_vehicle_max, BW_vehicle_new)
            BW_HFD_min = np.minimum(BW_HFD_min, BW_HFD_new)
            BW_HFD_max = np.maximum(BW_HFD_max, BW_HFD_new)
            # Glucose
            G_vehicle_min = np.minimum(G_vehicle_min, G_vehicle_new)
            G_vehicle_max = np.maximum(G_vehicle_max, G_vehicle_new)
            G_HFD_min = np.minimum(G_HFD_min, G_HFD_new)
            G_HFD_max = np.maximum(G_HFD_max, G_HFD_new)
            # U_id
            Uid_vehicle_min = np.minimum(Uid_vehicle_min, Uid_vehicle_new)
            Uid_vehicle_max = np.maximum(Uid_vehicle_max, Uid_vehicle_new)
            Uid_HFD_min = np.minimum(Uid_HFD_min, Uid_HFD_new)
            Uid_HFD_max = np.maximum(Uid_HFD_max, Uid_HFD_new)
            # Ins
            Ins_vehicle_min = np.minimum(Ins_vehicle_min, Ins_vehicle_new)
            Ins_vehicle_max = np.maximum(Ins_vehicle_max, Ins_vehicle_new)
            Ins_HFD_min = np.minimum(Ins_HFD_min, Ins_HFD_new)
            Ins_HFD_max = np.maximum(Ins_HFD_max, Ins_HFD_new)
            # EGP
            EGP_vehicle_min = np.minimum(EGP_vehicle_min, EGP_vehicle_new)
            EGP_vehicle_max = np.maximum(EGP_vehicle_max, EGP_vehicle_new)
            EGP_HFD_min = np.minimum(EGP_HFD_min, EGP_HFD_new)
            EGP_HFD_max = np.maximum(EGP_HFD_max, EGP_HFD_new)
    def compute_is_night(time, phase_shift_hours=16):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        IsNight = cycle > 0
        return IsNight
    IsNight = compute_is_night(time)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time)-1]))
    time = best_sim["control"].time_vector / 7.0
    t_BW = np.array(data["HeStudy"]["time_BW"], dtype=float) / 7.0
    t_G = np.array(data["HeStudy"]["time_Glucose"], dtype=float) / 7.0

    colors = {
        "Vehicle": "blue",
        "HFD": "firebrick"
        }
    
    # ---------- Figure 1: He study validation - Total body mass, basal glucose, and disease progression ----------
    fig_he1, axes_he1 = plt.subplots(1, 3, figsize=(18, 5), sharex=False)
    ax_BW = axes_he1[0]
    ax_G = axes_he1[1]
    ax_DP = axes_he1[2]
    t_daily = np.arange(0, 148) / 7
    # Column 1: Total body mass
    ax_BW.fill_between(time, BW_vehicle_min, BW_vehicle_max, color=colors["Vehicle"], alpha=0.4, linewidth=0)
    ax_BW.plot(time, best_sim["control"].feature_values[:, idx_BW], lw=2, color=colors["Vehicle"], label="Chow diet")
    ax_BW.errorbar(t_BW, data["HeStudy"]["Control_BW"], yerr=data["HeStudy"]["Control_BW_stdev"], fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2, markersize=8)
    ax_BW.fill_between(time, BW_HFD_min, BW_HFD_max, color=colors["HFD"], alpha=0.4, linewidth=0)
    ax_BW.plot(time, best_sim["HFD"].feature_values[:, idx_BW], lw=2, color=colors["HFD"], label="High fat diet")
    ax_BW.errorbar(t_BW, data["HeStudy"]["HFD_BW"], yerr=data["HeStudy"]["HFD_BW_stdev"], fmt='none', color=colors["HFD"], capsize=8, elinewidth=2, capthick=2, markersize=8)
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_BW.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    ax_BW.set_xlabel("Time (weeks)", fontsize=18)
    ax_BW.set_ylabel("Total body mass\n(g)", fontsize=18)
    ax_BW.grid(True)
    ax_BW.legend(loc="best", framealpha=1, fontsize=14)
    ax_BW.tick_params(axis='both', which='major', labelsize=18)
    ax_BW.set_xlim(-1, 22)
    ax_BW.set_ylim(10, 50)
    ax_BW.set_xticks([0, 4, 8, 12, 16, 20])
    ax_BW.set_yticks([10, 20, 30, 40, 50])
    # Column 2: Plasma glucose
    ax_G.fill_between(t_daily, G_vehicle_min, G_vehicle_max, color=colors["Vehicle"], alpha=0.4, linewidth=0)
    ax_G.plot(t_daily, G_vehicle_new, lw=2, color=colors["Vehicle"], label="Chow diet")
    ax_G.errorbar(t_G, data["HeStudy"]["Control_Glucose"], yerr=data["HeStudy"]["Control_Glucose_stdev"], fmt='none', color=colors["Vehicle"], capsize=8, elinewidth=2, capthick=2, markersize=8)
    ax_G.fill_between(t_daily, G_HFD_min, G_HFD_max, color=colors["HFD"], alpha=0.4, linewidth=0)
    ax_G.plot(t_daily, G_HFD_new, lw=2, color=colors["HFD"], label="High fat diet")
    ax_G.errorbar(t_G, data["HeStudy"]["HFD_Glucose"], yerr=data["HeStudy"]["HFD_Glucose_stdev"], fmt='none', color=colors["HFD"], capsize=8, elinewidth=2, capthick=2, markersize=8)
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_G.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    ax_G.set_xlabel("Time (weeks)", fontsize=18)
    ax_G.set_ylabel("Basal plasma glucose\n(mM)", fontsize=18)
    ax_G.grid(True)
    ax_G.legend(loc="best", framealpha=1, fontsize=12)
    ax_G.tick_params(axis='both', which='major', labelsize=18)
    ax_G.set_xlim(-1, 22)
    ax_G.set_xticks([0, 4, 8, 12, 16, 20])
    # Column 3: Disease progression
    ax_DP.fill_between(time, Uid_vehicle_min, Uid_vehicle_max, color=colors["Vehicle"], alpha=0.25, linewidth=0)
    ax_DP.plot(time, 1/(best_sim["control"].feature_values[:, idx_U_id]), color=colors["Vehicle"], lw=3, linestyle="-")
    ax_DP.fill_between(time, EGP_vehicle_min, EGP_vehicle_max, color=colors["Vehicle"], alpha=0.20, linewidth=0)
    ax_DP.plot(time, best_sim["control"].feature_values[:, idx_EGP], color=colors["Vehicle"], lw=2.5, linestyle="--")
    ax_DP.fill_between(time, Ins_vehicle_min, Ins_vehicle_max, color=colors["Vehicle"], alpha=0.15, linewidth=0)
    ax_DP.plot(time, best_sim["control"].feature_values[:, idx_Ins], color=colors["Vehicle"], lw=2, linestyle=":")
    ax_DP.fill_between(time, Uid_HFD_min, Uid_HFD_max, color=colors["HFD"], alpha=0.25, linewidth=0)
    ax_DP.plot(time, 1/(best_sim["HFD"].feature_values[:, idx_U_id]), color=colors["HFD"], lw=3, linestyle="-")
    ax_DP.fill_between(time, EGP_HFD_min, EGP_HFD_max, color=colors["HFD"], alpha=0.20, linewidth=0)
    ax_DP.plot(time, best_sim["HFD"].feature_values[:, idx_EGP], color=colors["HFD"], lw=2.5, linestyle="--")
    ax_DP.fill_between(time, Ins_HFD_min, Ins_HFD_max, color=colors["HFD"], alpha=0.15, linewidth=0)
    ax_DP.plot(time, best_sim["HFD"].feature_values[:, idx_Ins], color=colors["HFD"], lw=2, linestyle=":")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_DP.axvspan(time[edges[i]], time[edges[i+1]], color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    ax_DP.set_xlabel("Time (weeks)", fontsize=18)
    ax_DP.set_ylabel("Disease progression\n(fold-change)", fontsize=18)
    ax_DP.set_yscale("log")
    ax_DP.set_ylim(0.05, 20)
    ax_DP.grid(True)
    #ax_DP.legend(loc="best", framealpha=1, fontsize=14)
    ax_DP.tick_params(axis='both', which='major', labelsize=18)
    ax_DP.set_xlim(-1, 22)
    ax_DP.set_xticks([0, 4, 8, 12, 16, 20])
    plt.setp(ax_BW.get_xticklabels(), fontsize=18)
    plt.setp(ax_BW.get_yticklabels(), fontsize=18)
    plt.setp(ax_G.get_xticklabels(), fontsize=18)
    plt.setp(ax_G.get_yticklabels(), fontsize=18)
    plt.setp(ax_DP.get_xticklabels(), fontsize=18)
    plt.setp(ax_DP.get_yticklabels(), fontsize=18)
    fig_he1.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# --- Gabery study ---
def plot_Gabery_uncertainty(Model_Appetite, model_BW, Model_PK, data, OptimisedAppetiteParams, OptimisedBWparams, OptimisedPKparams, FastPlot):

    OptimisedAppetiteParams = OptimisedAppetiteParams.to_numpy(dtype=float)
    OptimisedBWparams       = OptimisedBWparams.to_numpy(dtype=float)
    OptimisedPKparams       = OptimisedPKparams.to_numpy(dtype=float)

    Best_BWParams       = np.asarray(OptimisedBWparams[0][:-1], dtype=float)
    Best_PKParams       = np.asarray(OptimisedPKparams[0][:-1], dtype=float)
    Best_AppetiteParams = np.asarray(OptimisedAppetiteParams[0][:-1], dtype=float)

    def daily_food_intake_from_ei_ac(ei_ac, time, kcal_per_g):
        ei_ac = np.asarray(ei_ac, dtype=float)
        time = np.asarray(time, dtype=float)
        first_day = int(np.ceil(np.nanmin(time)))
        last_day = int(np.floor(np.nanmax(time)))
        day_edges = np.arange(first_day, last_day + 1, dtype=float)
        ei_edges = np.interp(day_edges, time, ei_ac)
        daily_ei_kcal = np.diff(ei_edges)
        daily_food_g = daily_ei_kcal / kcal_per_g
        daily_time = day_edges[:-1]
        return daily_time, daily_food_g
    Gabery_HFD_kcal_per_g = 4.73

    # Calculate cost
    RecalculateCost = False # Set to True ot false, if you want to recalculate the cost for all parameter vectors (this take a long time)
    def _validate_param(VectorNumber,IC50):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_GaberyStudy(Model_Appetite, model_BW, Model_PK, data, AppetiteParams, Best_BWParams, Best_PKParams, 0.12, 1, IC50, 2)
            idx_ei_ac = sims["Vehicle"].feature_names.index("EI_AC")
            idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
            t_data = np.asarray(data["GaberyStudy"]["time"][0:24], dtype=float)
            BW_Vehicle = np.interp(t_data, sims["Vehicle"].time_vector, sims["Vehicle"].feature_values[:, idx_bw])
            BW_Sema1 = np.interp(t_data, sims["Sema1"].time_vector, sims["Sema1"].feature_values[:, idx_bw])
            BW_Sema3 = np.interp(t_data, sims["Sema3"].time_vector, sims["Sema3"].feature_values[:, idx_bw])
            BW_Sema10 = np.interp(t_data, sims["Sema10"].time_vector, sims["Sema10"].feature_values[:, idx_bw])
            BW_Sema30 = np.interp(t_data, sims["Sema30"].time_vector, sims["Sema30"].feature_values[:, idx_bw])
            BW_Sema100 = np.interp(t_data, sims["Sema100"].time_vector, sims["Sema100"].feature_values[:, idx_bw])
            time_FI_Vehicle, FI_Vehicle = daily_food_intake_from_ei_ac(sims["Vehicle"].feature_values[:, idx_ei_ac], sims["Vehicle"].time_vector, Gabery_HFD_kcal_per_g)
            time_FI_Sema1, FI_Sema1 = daily_food_intake_from_ei_ac(sims["Sema1"].feature_values[:, idx_ei_ac], sims["Sema1"].time_vector, Gabery_HFD_kcal_per_g)
            time_FI_Sema3, FI_Sema3 = daily_food_intake_from_ei_ac(sims["Sema3"].feature_values[:, idx_ei_ac], sims["Sema3"].time_vector, Gabery_HFD_kcal_per_g)
            time_FI_Sema10, FI_Sema10 = daily_food_intake_from_ei_ac(sims["Sema10"].feature_values[:, idx_ei_ac], sims["Sema10"].time_vector, Gabery_HFD_kcal_per_g)
            time_FI_Sema30, FI_Sema30 = daily_food_intake_from_ei_ac(sims["Sema30"].feature_values[:, idx_ei_ac], sims["Sema30"].time_vector, Gabery_HFD_kcal_per_g)
            time_FI_Sema100, FI_Sema100 = daily_food_intake_from_ei_ac(sims["Sema100"].feature_values[:, idx_ei_ac], sims["Sema100"].time_vector, Gabery_HFD_kcal_per_g)
            FI_Vehicle_atData = np.interp(t_data, time_FI_Vehicle, FI_Vehicle)
            FI_Sema1_atData = np.interp(t_data, time_FI_Sema1, FI_Sema1)
            FI_Sema3_atData = np.interp(t_data, time_FI_Sema3, FI_Sema3)
            FI_Sema10_atData = np.interp(t_data, time_FI_Sema10, FI_Sema10)
            FI_Sema30_atData = np.interp(t_data, time_FI_Sema30, FI_Sema30)
            FI_Sema100_atData = np.interp(t_data, time_FI_Sema100, FI_Sema100)
            SimulatedValues = np.concatenate([BW_Vehicle, BW_Sema1, BW_Sema3, BW_Sema10, BW_Sema30, BW_Sema100, FI_Vehicle_atData, FI_Sema1_atData, FI_Sema3_atData, FI_Sema10_atData, FI_Sema30_atData, FI_Sema100_atData])
            DataMean = np.asarray(data["GaberyStudy"]["all_mean"], dtype=float)
            DataWeightedSD = np.asarray(data["GaberyStudy"]["all_weighted_sd"], dtype=float)
            if len(SimulatedValues) != len(DataMean) or len(DataMean) != len(DataWeightedSD):
                raise ValueError(f"Length mismatch: simulation={len(SimulatedValues)}, mean={len(DataMean)}, SD={len(DataWeightedSD)}")
            CostValidation = np.sum(((DataMean - SimulatedValues) / DataWeightedSD) ** 2)
            return CostValidation
        except Exception as error:
            print(f"Vector {VectorNumber} failed: {error}")
            return np.inf
    if RecalculateCost:
        ValidationCosts = Parallel(n_jobs=1, verbose=10)(delayed(_validate_param)(VectorNumber,2.8) for VectorNumber in range(len(OptimisedAppetiteParams)))
        BestVectorNumber = int(np.argmin(ValidationCosts))
        BestValidationCost = float(ValidationCosts[BestVectorNumber])
        print(f"Best validation cost, IC50 = 2.8: {BestValidationCost}")
        print(f"Best parameter vector index, IC50 = 2.8: {BestVectorNumber}")

        ValidationCosts = Parallel(n_jobs=1, verbose=10)(delayed(_validate_param)(VectorNumber,1.3) for VectorNumber in range(len(OptimisedAppetiteParams)))
        BestVectorNumber = int(np.argmin(ValidationCosts))
        BestValidationCost = float(ValidationCosts[BestVectorNumber])
        print(f"Best validation cost, IC50 = 1.3: {BestValidationCost}")
        print(f"Best parameter vector index, IC50 = 1.3: {BestVectorNumber}")


    n_select = 300
    if FastPlot:
        n_select = 10
    idx = np.linspace(0, OptimisedAppetiteParams.shape[0] - 1, n_select, dtype=int)
    OptimisedAppetiteParams = OptimisedAppetiteParams[idx]
    n_select = 500
    if FastPlot:
        n_select = 50
    idx = np.linspace(0, OptimisedPKparams.shape[0] - 1, n_select, dtype=int)
    OptimisedPKparams = OptimisedPKparams[idx]


    # Not calibrated
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):

        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_GaberyStudy(Model_Appetite, model_BW, Model_PK, data, AppetiteParams, Best_BWParams, Best_PKParams, Best_BWParams[-2],1, 2.8, 2)
        except Exception as e:
            print("Gabery simulation failed:", repr(e))
            continue
        idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
        Vehicle_BWNoCal_new = sims["Vehicle"].feature_values[:, idx_bw]
        if first:
            best_sim_NoCal = sims
            time_BWNoCal = sims["Vehicle"].time_vector
            Vehicle_BWNoCal_min, Vehicle_BWNoCal_max = Vehicle_BWNoCal_new.copy(), Vehicle_BWNoCal_new.copy()
            first = False
        else:
            Vehicle_BWNoCal_min = np.minimum(Vehicle_BWNoCal_min, Vehicle_BWNoCal_new)
            Vehicle_BWNoCal_max = np.maximum(Vehicle_BWNoCal_max, Vehicle_BWNoCal_new)

    # Calibrated AZ study potency
    PhysicalActivity = 0.12
    Recovery = 1
    IC50 = 2.8
    hill = 2
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):

        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_GaberyStudy(Model_Appetite, model_BW, Model_PK, data, AppetiteParams, Best_BWParams, Best_PKParams,PhysicalActivity,Recovery, IC50, hill)
        except Exception as e:
            print("Gabery simulation failed:", repr(e))
            continue
        idx_ei_ac = sims["Vehicle"].feature_names.index("EI_AC")
        idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
        idx_sema = sims["Vehicle"].feature_names.index("Plasma_Semaglutide")
        idx_lambda_delay = sims["Vehicle"].feature_names.index("lambda_delay")

        time_FI, Vehicle_FI_new = daily_food_intake_from_ei_ac(sims["Vehicle"].feature_values[:, idx_ei_ac], sims["Vehicle"].time_vector, Gabery_HFD_kcal_per_g)
        _, Sema1_FI_new = daily_food_intake_from_ei_ac(sims["Sema1"].feature_values[:, idx_ei_ac], sims["Sema1"].time_vector, Gabery_HFD_kcal_per_g)
        _, Sema3_FI_new = daily_food_intake_from_ei_ac(sims["Sema3"].feature_values[:, idx_ei_ac], sims["Sema3"].time_vector, Gabery_HFD_kcal_per_g)
        _, Sema10_FI_new = daily_food_intake_from_ei_ac(sims["Sema10"].feature_values[:, idx_ei_ac], sims["Sema10"].time_vector, Gabery_HFD_kcal_per_g)
        _, Sema30_FI_new = daily_food_intake_from_ei_ac(sims["Sema30"].feature_values[:, idx_ei_ac], sims["Sema30"].time_vector, Gabery_HFD_kcal_per_g)
        _, Sema100_FI_new = daily_food_intake_from_ei_ac(sims["Sema100"].feature_values[:, idx_ei_ac], sims["Sema100"].time_vector, Gabery_HFD_kcal_per_g)
        Vehicle_BW_new = sims["Vehicle"].feature_values[:, idx_bw]
        Sema1_BW_new = sims["Sema1"].feature_values[:, idx_bw]
        Sema3_BW_new = sims["Sema3"].feature_values[:, idx_bw]
        Sema10_BW_new = sims["Sema10"].feature_values[:, idx_bw]
        Sema30_BW_new = sims["Sema30"].feature_values[:, idx_bw]
        Sema100_BW_new = sims["Sema100"].feature_values[:, idx_bw]
        lambda_delay_vehicle_new = sims["Vehicle"].feature_values[:,idx_lambda_delay]

        if first:

            best_sim = sims
            time_BW = sims["Vehicle"].time_vector
            time_Sema = sims["Vehicle"].time_vector
            best_Vehicle_FI = Vehicle_FI_new.copy()
            best_Sema1_FI = Sema1_FI_new.copy()
            best_Sema3_FI = Sema3_FI_new.copy()
            best_Sema10_FI = Sema10_FI_new.copy()
            best_Sema30_FI = Sema30_FI_new.copy()
            best_Sema100_FI = Sema100_FI_new.copy()
            Vehicle_FI_min, Vehicle_FI_max = Vehicle_FI_new.copy(), Vehicle_FI_new.copy()
            Sema1_FI_min, Sema1_FI_max = Sema1_FI_new.copy(), Sema1_FI_new.copy()
            Sema3_FI_min, Sema3_FI_max = Sema3_FI_new.copy(), Sema3_FI_new.copy()
            Sema10_FI_min, Sema10_FI_max = Sema10_FI_new.copy(), Sema10_FI_new.copy()
            Sema30_FI_min, Sema30_FI_max = Sema30_FI_new.copy(), Sema30_FI_new.copy()
            Sema100_FI_min, Sema100_FI_max = Sema100_FI_new.copy(), Sema100_FI_new.copy()
            Vehicle_BW_min, Vehicle_BW_max = Vehicle_BW_new.copy(), Vehicle_BW_new.copy()
            Sema1_BW_min, Sema1_BW_max = Sema1_BW_new.copy(), Sema1_BW_new.copy()
            Sema3_BW_min, Sema3_BW_max = Sema3_BW_new.copy(), Sema3_BW_new.copy()
            Sema10_BW_min, Sema10_BW_max = Sema10_BW_new.copy(), Sema10_BW_new.copy()
            Sema30_BW_min, Sema30_BW_max = Sema30_BW_new.copy(), Sema30_BW_new.copy()
            Sema100_BW_min, Sema100_BW_max = Sema100_BW_new.copy(), Sema100_BW_new.copy()
            lambda_delay_vehicle_min,lambda_delay_vehicle_max,best_lambda_delay_vehicle = lambda_delay_vehicle_new.copy(),lambda_delay_vehicle_new.copy(),lambda_delay_vehicle_new.copy()

            first = False
        else:

            Vehicle_FI_min = np.minimum(Vehicle_FI_min, Vehicle_FI_new)
            Vehicle_FI_max = np.maximum(Vehicle_FI_max, Vehicle_FI_new)
            Sema1_FI_min = np.minimum(Sema1_FI_min, Sema1_FI_new)
            Sema1_FI_max = np.maximum(Sema1_FI_max, Sema1_FI_new)
            Sema3_FI_min = np.minimum(Sema3_FI_min, Sema3_FI_new)
            Sema3_FI_max = np.maximum(Sema3_FI_max, Sema3_FI_new)
            Sema10_FI_min = np.minimum(Sema10_FI_min, Sema10_FI_new)
            Sema10_FI_max = np.maximum(Sema10_FI_max, Sema10_FI_new)
            Sema30_FI_min = np.minimum(Sema30_FI_min, Sema30_FI_new)
            Sema30_FI_max = np.maximum(Sema30_FI_max, Sema30_FI_new)
            Sema100_FI_min = np.minimum(Sema100_FI_min, Sema100_FI_new)
            Sema100_FI_max = np.maximum(Sema100_FI_max, Sema100_FI_new)
            Vehicle_BW_min = np.minimum(Vehicle_BW_min, Vehicle_BW_new)
            Vehicle_BW_max = np.maximum(Vehicle_BW_max, Vehicle_BW_new)
            Sema1_BW_min = np.minimum(Sema1_BW_min, Sema1_BW_new)
            Sema1_BW_max = np.maximum(Sema1_BW_max, Sema1_BW_new)
            Sema3_BW_min = np.minimum(Sema3_BW_min, Sema3_BW_new)
            Sema3_BW_max = np.maximum(Sema3_BW_max, Sema3_BW_new)
            Sema10_BW_min = np.minimum(Sema10_BW_min, Sema10_BW_new)
            Sema10_BW_max = np.maximum(Sema10_BW_max, Sema10_BW_new)
            Sema30_BW_min = np.minimum(Sema30_BW_min, Sema30_BW_new)
            Sema30_BW_max = np.maximum(Sema30_BW_max, Sema30_BW_new)
            Sema100_BW_min = np.minimum(Sema100_BW_min, Sema100_BW_new)
            Sema100_BW_max = np.maximum(Sema100_BW_max, Sema100_BW_new)
            lambda_delay_vehicle_min,lambda_delay_vehicle_max = np.minimum(lambda_delay_vehicle_min,lambda_delay_vehicle_new),np.maximum(lambda_delay_vehicle_max,lambda_delay_vehicle_new)
    first = True
    for VectorNumber in range(len(OptimisedPKparams)):
        PKparams = np.asarray(OptimisedPKparams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_GaberyStudy(Model_Appetite, model_BW, Model_PK, data, Best_AppetiteParams, Best_BWParams, PKparams, PhysicalActivity, Recovery, IC50, hill)
        except Exception as e:
            print("Gabery simulation failed:", repr(e))
            continue
        idx_sema = sims["Vehicle"].feature_names.index("Plasma_Semaglutide")
        Vehicle_Sema_new = sims["Vehicle"].feature_values[:, idx_sema]
        Sema1_Sema_new = sims["Sema1"].feature_values[:, idx_sema]
        Sema3_Sema_new = sims["Sema3"].feature_values[:, idx_sema]
        Sema10_Sema_new = sims["Sema10"].feature_values[:, idx_sema]
        Sema30_Sema_new = sims["Sema30"].feature_values[:, idx_sema]
        Sema100_Sema_new = sims["Sema100"].feature_values[:, idx_sema]
        if first:
            time_Sema = sims["Vehicle"].time_vector
            Vehicle_Sema_min, Vehicle_Sema_max = Vehicle_Sema_new.copy(), Vehicle_Sema_new.copy()
            Sema1_Sema_min, Sema1_Sema_max = Sema1_Sema_new.copy(), Sema1_Sema_new.copy()
            Sema3_Sema_min, Sema3_Sema_max = Sema3_Sema_new.copy(), Sema3_Sema_new.copy()
            Sema10_Sema_min, Sema10_Sema_max = Sema10_Sema_new.copy(), Sema10_Sema_new.copy()
            Sema30_Sema_min, Sema30_Sema_max = Sema30_Sema_new.copy(), Sema30_Sema_new.copy()
            Sema100_Sema_min, Sema100_Sema_max = Sema100_Sema_new.copy(), Sema100_Sema_new.copy()
            first = False
        else:
            Vehicle_Sema_min = np.minimum(Vehicle_Sema_min, Vehicle_Sema_new)
            Vehicle_Sema_max = np.maximum(Vehicle_Sema_max, Vehicle_Sema_new)
            Sema1_Sema_min = np.minimum(Sema1_Sema_min, Sema1_Sema_new)
            Sema1_Sema_max = np.maximum(Sema1_Sema_max, Sema1_Sema_new)
            Sema3_Sema_min = np.minimum(Sema3_Sema_min, Sema3_Sema_new)
            Sema3_Sema_max = np.maximum(Sema3_Sema_max, Sema3_Sema_new)
            Sema10_Sema_min = np.minimum(Sema10_Sema_min, Sema10_Sema_new)
            Sema10_Sema_max = np.maximum(Sema10_Sema_max, Sema10_Sema_new)
            Sema30_Sema_min = np.minimum(Sema30_Sema_min, Sema30_Sema_new)
            Sema30_Sema_max = np.maximum(Sema30_Sema_max, Sema30_Sema_new)
            Sema100_Sema_min = np.minimum(Sema100_Sema_min, Sema100_Sema_new)
            Sema100_Sema_max = np.maximum(Sema100_Sema_max, Sema100_Sema_new)

    # Calibrated to the Gabery data
    IC50_2 = 1.3
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1],dtype=float)
        try:
            sims = Simulate_GaberyStudy(Model_Appetite,model_BW,Model_PK,data,AppetiteParams,Best_BWParams,Best_PKParams,PhysicalActivity,Recovery,IC50_2, hill)
        except Exception as e:
            print("Gabery simulation failed:",repr(e))
            continue
        idx_ei_ac = sims["Vehicle"].feature_names.index("EI_AC")
        idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
        idx_sema = sims["Vehicle"].feature_names.index("Plasma_Semaglutide")
        idx_lambda_delay = sims["Vehicle"].feature_names.index("lambda_delay")
        time_FI_IC50_6,Vehicle_FI_new_IC50_6 = daily_food_intake_from_ei_ac(sims["Vehicle"].feature_values[:,idx_ei_ac],sims["Vehicle"].time_vector,Gabery_HFD_kcal_per_g)
        _,Sema1_FI_new_IC50_6 = daily_food_intake_from_ei_ac(sims["Sema1"].feature_values[:,idx_ei_ac],sims["Sema1"].time_vector,Gabery_HFD_kcal_per_g)
        _,Sema3_FI_new_IC50_6 = daily_food_intake_from_ei_ac(sims["Sema3"].feature_values[:,idx_ei_ac],sims["Sema3"].time_vector,Gabery_HFD_kcal_per_g)
        _,Sema10_FI_new_IC50_6 = daily_food_intake_from_ei_ac(sims["Sema10"].feature_values[:,idx_ei_ac],sims["Sema10"].time_vector,Gabery_HFD_kcal_per_g)
        _,Sema30_FI_new_IC50_6 = daily_food_intake_from_ei_ac(sims["Sema30"].feature_values[:,idx_ei_ac],sims["Sema30"].time_vector,Gabery_HFD_kcal_per_g)
        _,Sema100_FI_new_IC50_6 = daily_food_intake_from_ei_ac(sims["Sema100"].feature_values[:,idx_ei_ac],sims["Sema100"].time_vector,Gabery_HFD_kcal_per_g)
        Vehicle_BW_new_IC50_6 = sims["Vehicle"].feature_values[:,idx_bw]
        Sema1_BW_new_IC50_6 = sims["Sema1"].feature_values[:,idx_bw]
        Sema3_BW_new_IC50_6 = sims["Sema3"].feature_values[:,idx_bw]
        Sema10_BW_new_IC50_6 = sims["Sema10"].feature_values[:,idx_bw]
        Sema30_BW_new_IC50_6 = sims["Sema30"].feature_values[:,idx_bw]
        Sema100_BW_new_IC50_6 = sims["Sema100"].feature_values[:,idx_bw]
        lambda_delay_vehicle_new_IC50_6 = sims["Vehicle"].feature_values[:,idx_lambda_delay]
        if first:
            time_BW_IC50_6 = sims["Vehicle"].time_vector
            best_Vehicle_FI_IC50_6 = Vehicle_FI_new_IC50_6.copy()
            best_Sema1_FI_IC50_6 = Sema1_FI_new_IC50_6.copy()
            best_Sema3_FI_IC50_6 = Sema3_FI_new_IC50_6.copy()
            best_Sema10_FI_IC50_6 = Sema10_FI_new_IC50_6.copy()
            best_Sema30_FI_IC50_6 = Sema30_FI_new_IC50_6.copy()
            best_Sema100_FI_IC50_6 = Sema100_FI_new_IC50_6.copy()
            Vehicle_FI_min_IC50_6,Vehicle_FI_max_IC50_6 = Vehicle_FI_new_IC50_6.copy(),Vehicle_FI_new_IC50_6.copy()
            Sema1_FI_min_IC50_6,Sema1_FI_max_IC50_6 = Sema1_FI_new_IC50_6.copy(),Sema1_FI_new_IC50_6.copy()
            Sema3_FI_min_IC50_6,Sema3_FI_max_IC50_6 = Sema3_FI_new_IC50_6.copy(),Sema3_FI_new_IC50_6.copy()
            Sema10_FI_min_IC50_6,Sema10_FI_max_IC50_6 = Sema10_FI_new_IC50_6.copy(),Sema10_FI_new_IC50_6.copy()
            Sema30_FI_min_IC50_6,Sema30_FI_max_IC50_6 = Sema30_FI_new_IC50_6.copy(),Sema30_FI_new_IC50_6.copy()
            Sema100_FI_min_IC50_6,Sema100_FI_max_IC50_6 = Sema100_FI_new_IC50_6.copy(),Sema100_FI_new_IC50_6.copy()
            Vehicle_BW_min_IC50_6,Vehicle_BW_max_IC50_6,best_Vehicle_BW_IC50_6 = Vehicle_BW_new_IC50_6.copy(),Vehicle_BW_new_IC50_6.copy(),Vehicle_BW_new_IC50_6.copy()
            Sema1_BW_min_IC50_6,Sema1_BW_max_IC50_6,best_Sema1_BW_IC50_6 = Sema1_BW_new_IC50_6.copy(),Sema1_BW_new_IC50_6.copy(),Sema1_BW_new_IC50_6.copy()
            Sema3_BW_min_IC50_6,Sema3_BW_max_IC50_6,best_Sema3_BW_IC50_6 = Sema3_BW_new_IC50_6.copy(),Sema3_BW_new_IC50_6.copy(),Sema3_BW_new_IC50_6.copy()
            Sema10_BW_min_IC50_6,Sema10_BW_max_IC50_6,best_Sema10_BW_IC50_6 = Sema10_BW_new_IC50_6.copy(),Sema10_BW_new_IC50_6.copy(),Sema10_BW_new_IC50_6.copy()
            Sema30_BW_min_IC50_6,Sema30_BW_max_IC50_6,best_Sema30_BW_IC50_6 = Sema30_BW_new_IC50_6.copy(),Sema30_BW_new_IC50_6.copy(),Sema30_BW_new_IC50_6.copy()
            Sema100_BW_min_IC50_6,Sema100_BW_max_IC50_6,best_Sema100_BW_IC50_6 = Sema100_BW_new_IC50_6.copy(),Sema100_BW_new_IC50_6.copy(),Sema100_BW_new_IC50_6.copy()
            lambda_delay_vehicle_min_IC50_6,lambda_delay_vehicle_max_IC50_6,best_lambda_delay_vehicle_IC50_6 = lambda_delay_vehicle_new_IC50_6.copy(),lambda_delay_vehicle_new_IC50_6.copy(),lambda_delay_vehicle_new_IC50_6.copy()
            first = False
        else:
            Vehicle_FI_min_IC50_6,Vehicle_FI_max_IC50_6 = np.minimum(Vehicle_FI_min_IC50_6,Vehicle_FI_new_IC50_6),np.maximum(Vehicle_FI_max_IC50_6,Vehicle_FI_new_IC50_6)
            Sema1_FI_min_IC50_6,Sema1_FI_max_IC50_6 = np.minimum(Sema1_FI_min_IC50_6,Sema1_FI_new_IC50_6),np.maximum(Sema1_FI_max_IC50_6,Sema1_FI_new_IC50_6)
            Sema3_FI_min_IC50_6,Sema3_FI_max_IC50_6 = np.minimum(Sema3_FI_min_IC50_6,Sema3_FI_new_IC50_6),np.maximum(Sema3_FI_max_IC50_6,Sema3_FI_new_IC50_6)
            Sema10_FI_min_IC50_6,Sema10_FI_max_IC50_6 = np.minimum(Sema10_FI_min_IC50_6,Sema10_FI_new_IC50_6),np.maximum(Sema10_FI_max_IC50_6,Sema10_FI_new_IC50_6)
            Sema30_FI_min_IC50_6,Sema30_FI_max_IC50_6 = np.minimum(Sema30_FI_min_IC50_6,Sema30_FI_new_IC50_6),np.maximum(Sema30_FI_max_IC50_6,Sema30_FI_new_IC50_6)
            Sema100_FI_min_IC50_6,Sema100_FI_max_IC50_6 = np.minimum(Sema100_FI_min_IC50_6,Sema100_FI_new_IC50_6),np.maximum(Sema100_FI_max_IC50_6,Sema100_FI_new_IC50_6)
            Vehicle_BW_min_IC50_6,Vehicle_BW_max_IC50_6 = np.minimum(Vehicle_BW_min_IC50_6,Vehicle_BW_new_IC50_6),np.maximum(Vehicle_BW_max_IC50_6,Vehicle_BW_new_IC50_6)
            Sema1_BW_min_IC50_6,Sema1_BW_max_IC50_6 = np.minimum(Sema1_BW_min_IC50_6,Sema1_BW_new_IC50_6),np.maximum(Sema1_BW_max_IC50_6,Sema1_BW_new_IC50_6)
            Sema3_BW_min_IC50_6,Sema3_BW_max_IC50_6 = np.minimum(Sema3_BW_min_IC50_6,Sema3_BW_new_IC50_6),np.maximum(Sema3_BW_max_IC50_6,Sema3_BW_new_IC50_6)
            Sema10_BW_min_IC50_6,Sema10_BW_max_IC50_6 = np.minimum(Sema10_BW_min_IC50_6,Sema10_BW_new_IC50_6),np.maximum(Sema10_BW_max_IC50_6,Sema10_BW_new_IC50_6)
            Sema30_BW_min_IC50_6,Sema30_BW_max_IC50_6 = np.minimum(Sema30_BW_min_IC50_6,Sema30_BW_new_IC50_6),np.maximum(Sema30_BW_max_IC50_6,Sema30_BW_new_IC50_6)
            Sema100_BW_min_IC50_6,Sema100_BW_max_IC50_6 = np.minimum(Sema100_BW_min_IC50_6,Sema100_BW_new_IC50_6),np.maximum(Sema100_BW_max_IC50_6,Sema100_BW_new_IC50_6)
            lambda_delay_vehicle_min_IC50_6,lambda_delay_vehicle_max_IC50_6 = np.minimum(lambda_delay_vehicle_min_IC50_6,lambda_delay_vehicle_new_IC50_6),np.maximum(lambda_delay_vehicle_max_IC50_6,lambda_delay_vehicle_new_IC50_6)
            
    # ---------- Day/night cycle ----------
    def compute_is_night(time, phase_shift_hours=16):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        return cycle > 0
    IsNight = compute_is_night(time_BW)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time_BW) - 1]))

    colors = {
        "Vehicle": "blue",
        "Sema1": "teal",
        "Sema3": "gold",
        "Sema10": "darkorange",
        "Sema30": "orangered",
        "Sema100": "firebrick"
    }

    # ======================
    # Calibration figure 1
    # ======================
    fig1,axes1 = plt.subplots(1,3,figsize=(18,5))
    ax_NoCal,ax_Cal,ax_lambda = axes1[0],axes1[1],axes1[2]
    # COLUMN 1: Not calibrated, Vehicle BW only
    ax_NoCal.fill_between(time_BWNoCal/7,Vehicle_BWNoCal_min,Vehicle_BWNoCal_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_NoCal.plot(time_BWNoCal/7,best_sim_NoCal["Vehicle"].feature_values[:,idx_bw],color=colors["Vehicle"],lw=2,label="Simulation")
    ax_NoCal.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float)/7,np.array(data["GaberyStudy"]["VehicleBW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["VehicleBW_stdev"][0:24],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2,label="Predicted data")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_NoCal.axvspan(time_BW[edges[i]]/7,time_BW[edges[i+1]]/7,color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    # COLUMN 2: Calibrated, Vehicle BW only
    ax_Cal.fill_between(time_BW/7,Vehicle_BW_min,Vehicle_BW_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_Cal.plot(time_BW/7,best_sim["Vehicle"].feature_values[:,idx_bw],color=colors["Vehicle"],lw=2,label="Simulation")
    ax_Cal.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float)/7,np.array(data["GaberyStudy"]["VehicleBW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["VehicleBW_stdev"][0:24],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2,label="Predicted data")
    ax_Cal.errorbar(np.array(data["GaberyStudy"]["time"][23],dtype=float)/7,np.array(data["GaberyStudy"]["VehicleBW"][23],dtype=float),fmt="+",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Physical activity",zorder=100)
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_Cal.axvspan(time_BW[edges[i]]/7,time_BW[edges[i+1]]/7,color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    # COLUMN 3: Physical-activity calibration
    ax_lambda.fill_between(time_BW/7,lambda_delay_vehicle_min,lambda_delay_vehicle_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_lambda.plot(time_BW/7,best_sim["Vehicle"].feature_values[:,idx_lambda_delay],color=colors["Vehicle"],lw=2,ls="--",label="Physical activity")
    ax_lambda.errorbar(np.array(data["GaberyStudy"]["time"][23],dtype=float)/7,PhysicalActivity,fmt="+",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Physical activity",zorder=100)
    ax_lambda.axhline(y=Best_BWParams[-2], color="black", linestyle="--", linewidth=2, label="Used in blind prediction (Fig. S3C i)")
    for i in range(len(edges)-1):
        if IsNight[edges[i]]:
            ax_lambda.axvspan(time_BW[edges[i]]/7,time_BW[edges[i+1]]/7,color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    # Formatting
    titles = ["Physical activity assumed\nfrom Guo and Hall chow group","Calibrating activity to vehicle group","Calibration visualised"]
    for i,ax in enumerate([ax_NoCal,ax_Cal]):
        ax.set_title(titles[i],fontsize=18)
        ax.set_xlabel("Time (weeks)",fontsize=18)
        ax.set_ylabel("Total body mass\n(g)",fontsize=18)
        ax.set_xlim(-0.1,24/7)
        ax.set_xticks([0,1,2,3])
        ax.set_ylim(25,50)
        ax.set_yticks([25,35,45])
        ax.tick_params(axis="both",which="major",labelsize=16)
        ax.grid(True)
        ax.legend(loc="best",fontsize=12,framealpha=1)
    ax_lambda.set_title(titles[2],fontsize=18)
    ax_lambda.set_xlabel("Time (weeks)",fontsize=18)
    ax_lambda.set_ylabel("Physical activity\n(kcal/g/day)",fontsize=18)
    ax_lambda.set_xlim(-0.1,24/7)
    ax_lambda.set_xticks([0,1, 2, 3])
    ax_lambda.set_ylim(0,1)
    ax_lambda.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax_lambda.tick_params(axis="both",which="major",labelsize=16)
    ax_lambda.grid(True)
    ax_lambda.legend(loc="best",fontsize=12,framealpha=1)
    fig1.tight_layout()
    plt.show()


    # ======================
    # Combined Fig. 2
    # ======================
    fig = plt.figure(figsize=(21,10))
    gs = fig.add_gridspec(2,3)
    ax_BW_IC50_3p9 = fig.add_subplot(gs[0,0])
    ax_FI_IC50_3p9 = fig.add_subplot(gs[1,0])
    ax_BW_IC50_6 = fig.add_subplot(gs[0,1])
    ax_FI_IC50_6 = fig.add_subplot(gs[1,1])
    ax_Sema = fig.add_subplot(gs[:,2])
    # COLUMN 1, ROW 1: BODY WEIGHT, prediction
    ax_BW_IC50_3p9.fill_between(time_BW,Vehicle_BW_min,Vehicle_BW_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_3p9.plot(time_BW,best_sim["Vehicle"].feature_values[:,idx_bw],color=colors["Vehicle"],lw=2,label="Vehicle")
    ax_BW_IC50_3p9.fill_between(time_BW,Sema1_BW_min,Sema1_BW_max,color=colors["Sema1"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_3p9.plot(time_BW,best_sim["Sema1"].feature_values[:,idx_bw],color=colors["Sema1"],lw=2,label="1 nmol/kg/day")
    ax_BW_IC50_3p9.fill_between(time_BW,Sema3_BW_min,Sema3_BW_max,color=colors["Sema3"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_3p9.plot(time_BW,best_sim["Sema3"].feature_values[:,idx_bw],color=colors["Sema3"],lw=2,label="3 nmol/kg/day")
    ax_BW_IC50_3p9.fill_between(time_BW,Sema10_BW_min,Sema10_BW_max,color=colors["Sema10"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_3p9.plot(time_BW,best_sim["Sema10"].feature_values[:,idx_bw],color=colors["Sema10"],lw=2,label="10 nmol/kg/day")
    ax_BW_IC50_3p9.fill_between(time_BW,Sema30_BW_min,Sema30_BW_max,color=colors["Sema30"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_3p9.plot(time_BW,best_sim["Sema30"].feature_values[:,idx_bw],color=colors["Sema30"],lw=2,label="30 nmol/kg/day")
    ax_BW_IC50_3p9.fill_between(time_BW,Sema100_BW_min,Sema100_BW_max,color=colors["Sema100"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_3p9.plot(time_BW,best_sim["Sema100"].feature_values[:,idx_bw],color=colors["Sema100"],lw=2,label="100 nmol/kg/day")
    ax_BW_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["VehicleBW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["VehicleBW_stdev"][0:24],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema1BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema1BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema1"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema3BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema3BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema3"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema10BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema10BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema10"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema30BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema30BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema30"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema100BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema100BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema100"],capsize=4,elinewidth=2,capthick=2)
    # COLUMN 1, ROW 2: FOOD INTAKE, prediction
    ax_FI_IC50_3p9.fill_between(time_FI,Vehicle_FI_min,Vehicle_FI_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_3p9.plot(time_FI,best_Vehicle_FI,color=colors["Vehicle"],lw=2,label="Vehicle")
    ax_FI_IC50_3p9.fill_between(time_FI,Sema1_FI_min,Sema1_FI_max,color=colors["Sema1"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_3p9.plot(time_FI,best_Sema1_FI,color=colors["Sema1"],lw=2,label="1 nmol/kg/day")
    ax_FI_IC50_3p9.fill_between(time_FI,Sema3_FI_min,Sema3_FI_max,color=colors["Sema3"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_3p9.plot(time_FI,best_Sema3_FI,color=colors["Sema3"],lw=2,label="3 nmol/kg/day")
    ax_FI_IC50_3p9.fill_between(time_FI,Sema10_FI_min,Sema10_FI_max,color=colors["Sema10"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_3p9.plot(time_FI,best_Sema10_FI,color=colors["Sema10"],lw=2,label="10 nmol/kg/day")
    ax_FI_IC50_3p9.fill_between(time_FI,Sema30_FI_min,Sema30_FI_max,color=colors["Sema30"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_3p9.plot(time_FI,best_Sema30_FI,color=colors["Sema30"],lw=2,label="30 nmol/kg/day")
    ax_FI_IC50_3p9.fill_between(time_FI,Sema100_FI_min,Sema100_FI_max,color=colors["Sema100"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_3p9.plot(time_FI,best_Sema100_FI,color=colors["Sema100"],lw=2,label="100 nmol/kg/day")
    ax_FI_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["VehicleFI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["VehicleFI_stdev"][0:24],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema1FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema1FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema1"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema3FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema3FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema3"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema10FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema10FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema10"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema30FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema30FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema30"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_3p9.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema100FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema100FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema100"],capsize=4,elinewidth=2,capthick=2)
    # COLUMN 2, ROW 1: BODY WEIGHT, calibrated
    ax_BW_IC50_6.fill_between(time_BW_IC50_6,Vehicle_BW_min_IC50_6,Vehicle_BW_max_IC50_6,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_6.plot(time_BW_IC50_6,best_Vehicle_BW_IC50_6,color=colors["Vehicle"],lw=2,label="Vehicle")
    ax_BW_IC50_6.fill_between(time_BW_IC50_6,Sema1_BW_min_IC50_6,Sema1_BW_max_IC50_6,color=colors["Sema1"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_6.plot(time_BW_IC50_6,best_Sema1_BW_IC50_6,color=colors["Sema1"],lw=2,label="1 nmol/kg/day")
    ax_BW_IC50_6.fill_between(time_BW_IC50_6,Sema3_BW_min_IC50_6,Sema3_BW_max_IC50_6,color=colors["Sema3"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_6.plot(time_BW_IC50_6,best_Sema3_BW_IC50_6,color=colors["Sema3"],lw=2,label="3 nmol/kg/day")
    ax_BW_IC50_6.fill_between(time_BW_IC50_6,Sema10_BW_min_IC50_6,Sema10_BW_max_IC50_6,color=colors["Sema10"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_6.plot(time_BW_IC50_6,best_Sema10_BW_IC50_6,color=colors["Sema10"],lw=2,label="10 nmol/kg/day")
    ax_BW_IC50_6.fill_between(time_BW_IC50_6,Sema30_BW_min_IC50_6,Sema30_BW_max_IC50_6,color=colors["Sema30"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_6.plot(time_BW_IC50_6,best_Sema30_BW_IC50_6,color=colors["Sema30"],lw=2,label="30 nmol/kg/day")
    ax_BW_IC50_6.fill_between(time_BW_IC50_6,Sema100_BW_min_IC50_6,Sema100_BW_max_IC50_6,color=colors["Sema100"],alpha=0.15,edgecolor="none")
    ax_BW_IC50_6.plot(time_BW_IC50_6,best_Sema100_BW_IC50_6,color=colors["Sema100"],lw=2,label="100 nmol/kg/day")
    ax_BW_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["VehicleBW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["VehicleBW_stdev"][0:24],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema1BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema1BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema1"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema3BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema3BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema3"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema10BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema10BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema10"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema30BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema30BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema30"],capsize=4,elinewidth=2,capthick=2)
    ax_BW_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema100BW"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema100BW_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema100"],capsize=4,elinewidth=2,capthick=2)
    # COLUMN 2, ROW 2: FOOD INTAKE, calibrated
    ax_FI_IC50_6.fill_between(time_FI_IC50_6,Vehicle_FI_min_IC50_6,Vehicle_FI_max_IC50_6,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_6.plot(time_FI_IC50_6,best_Vehicle_FI_IC50_6,color=colors["Vehicle"],lw=2,label="Vehicle")
    ax_FI_IC50_6.fill_between(time_FI_IC50_6,Sema1_FI_min_IC50_6,Sema1_FI_max_IC50_6,color=colors["Sema1"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_6.plot(time_FI_IC50_6,best_Sema1_FI_IC50_6,color=colors["Sema1"],lw=2,label="1 nmol/kg/day")
    ax_FI_IC50_6.fill_between(time_FI_IC50_6,Sema3_FI_min_IC50_6,Sema3_FI_max_IC50_6,color=colors["Sema3"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_6.plot(time_FI_IC50_6,best_Sema3_FI_IC50_6,color=colors["Sema3"],lw=2,label="3 nmol/kg/day")
    ax_FI_IC50_6.fill_between(time_FI_IC50_6,Sema10_FI_min_IC50_6,Sema10_FI_max_IC50_6,color=colors["Sema10"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_6.plot(time_FI_IC50_6,best_Sema10_FI_IC50_6,color=colors["Sema10"],lw=2,label="10 nmol/kg/day")
    ax_FI_IC50_6.fill_between(time_FI_IC50_6,Sema30_FI_min_IC50_6,Sema30_FI_max_IC50_6,color=colors["Sema30"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_6.plot(time_FI_IC50_6,best_Sema30_FI_IC50_6,color=colors["Sema30"],lw=2,label="30 nmol/kg/day")
    ax_FI_IC50_6.fill_between(time_FI_IC50_6,Sema100_FI_min_IC50_6,Sema100_FI_max_IC50_6,color=colors["Sema100"],alpha=0.15,edgecolor="none")
    ax_FI_IC50_6.plot(time_FI_IC50_6,best_Sema100_FI_IC50_6,color=colors["Sema100"],lw=2,label="100 nmol/kg/day")
    ax_FI_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["VehicleFI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["VehicleFI_stdev"][0:24],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema1FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema1FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema1"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema3FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema3FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema3"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema10FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema10FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema10"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema30FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema30FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema30"],capsize=4,elinewidth=2,capthick=2)
    ax_FI_IC50_6.errorbar(np.array(data["GaberyStudy"]["time"][0:24],dtype=float),np.array(data["GaberyStudy"]["Sema100FI"][0:24],dtype=float),yerr=np.array(data["GaberyStudy"]["Sema100FI_stdev"][0:24],dtype=float),fmt="none",color=colors["Sema100"],capsize=4,elinewidth=2,capthick=2)
    # COLUMN 3: SEMAGLUTIDE EXPOSURE
    ax_Sema.fill_between(time_Sema,np.maximum(Vehicle_Sema_min,1e-6),np.maximum(Vehicle_Sema_max,1e-6),color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_Sema.plot(time_Sema,np.maximum(best_sim["Vehicle"].feature_values[:,idx_sema],1e-6),color=colors["Vehicle"],lw=2,label="Vehicle")
    ax_Sema.fill_between(time_Sema,np.maximum(Sema1_Sema_min,1e-6),np.maximum(Sema1_Sema_max,1e-6),color=colors["Sema1"],alpha=0.15,edgecolor="none")
    ax_Sema.plot(time_Sema,np.maximum(best_sim["Sema1"].feature_values[:,idx_sema],1e-6),color=colors["Sema1"],lw=2,label="1 nmol/kg/day")
    ax_Sema.fill_between(time_Sema,np.maximum(Sema3_Sema_min,1e-6),np.maximum(Sema3_Sema_max,1e-6),color=colors["Sema3"],alpha=0.15,edgecolor="none")
    ax_Sema.plot(time_Sema,np.maximum(best_sim["Sema3"].feature_values[:,idx_sema],1e-6),color=colors["Sema3"],lw=2,label="3 nmol/kg/day")
    ax_Sema.fill_between(time_Sema,np.maximum(Sema10_Sema_min,1e-6),np.maximum(Sema10_Sema_max,1e-6),color=colors["Sema10"],alpha=0.15,edgecolor="none")
    ax_Sema.plot(time_Sema,np.maximum(best_sim["Sema10"].feature_values[:,idx_sema],1e-6),color=colors["Sema10"],lw=2,label="10 nmol/kg/day")
    ax_Sema.fill_between(time_Sema,np.maximum(Sema30_Sema_min,1e-6),np.maximum(Sema30_Sema_max,1e-6),color=colors["Sema30"],alpha=0.15,edgecolor="none")
    ax_Sema.plot(time_Sema,np.maximum(best_sim["Sema30"].feature_values[:,idx_sema],1e-6),color=colors["Sema30"],lw=2,label="30 nmol/kg/day")
    ax_Sema.fill_between(time_Sema,np.maximum(Sema100_Sema_min,1e-6),np.maximum(Sema100_Sema_max,1e-6),color=colors["Sema100"],alpha=0.15,edgecolor="none")
    ax_Sema.plot(time_Sema,np.maximum(best_sim["Sema100"].feature_values[:,idx_sema],1e-6),color=colors["Sema100"],lw=2,label="100 nmol/kg/day")
    ax_Sema.axhline(y=IC50,color="black",ls="--",lw=1.5)
    ax_Sema.axhline(y=IC50_2,color="black",ls=":",lw=2)
    # Formating
    for ax in [ax_BW_IC50_3p9,ax_FI_IC50_3p9,ax_BW_IC50_6,ax_FI_IC50_6,ax_Sema]:
        for i in range(len(edges)-1):
            if IsNight[edges[i]]:
                ax.axvspan(time_BW[edges[i]],time_BW[edges[i+1]],color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    dosing_times_BID = np.sort(np.concatenate([
        np.arange(3, 24, 1.0),
        np.arange(3 + 0.3, 24, 1.0)
    ]))
    for dose_time in dosing_times_BID:
        ax_BW_IC50_3p9.plot([dose_time-0.1, dose_time+0.1], [25,25], color="black", lw=8, solid_capstyle="butt")
        ax_BW_IC50_6.plot([dose_time-0.1, dose_time+0.1], [25,25], color="black", lw=8, solid_capstyle="butt")
        ax_FI_IC50_3p9.plot([dose_time-0.1, dose_time+0.1], [0,0], color="black", lw=8, solid_capstyle="butt")
        ax_FI_IC50_6.plot([dose_time-0.1, dose_time+0.1], [0,0], color="black", lw=8, solid_capstyle="butt")
        ax_Sema.plot([dose_time-0.1, dose_time+0.1], [1e-1,1e-1], color="black", lw=8, solid_capstyle="butt")
    # COLUMN 1
    ax_BW_IC50_3p9.set_xlabel("Time (days)",fontsize=18)
    ax_BW_IC50_3p9.set_ylabel("Total body mass\n(g)",fontsize=18)
    ax_BW_IC50_3p9.set_title("Potency from Fig. 6A",fontsize=18)
    ax_BW_IC50_3p9.set_xlim(-1,24)
    ax_BW_IC50_3p9.set_xticks([0,7,14,21])
    ax_BW_IC50_3p9.set_ylim(25,50)
    ax_BW_IC50_3p9.set_yticks([25,35,45])
    ax_BW_IC50_3p9.tick_params(axis="both",which="major",labelsize=16)
    ax_BW_IC50_3p9.grid(True)
    ax_FI_IC50_3p9.set_xlabel("Time (days)",fontsize=18)
    ax_FI_IC50_3p9.set_ylabel("Food intake\n(g/day)",fontsize=18)
    ax_FI_IC50_3p9.set_xlim(-1,24)
    ax_FI_IC50_3p9.set_xticks([0,7,14,21])
    ax_FI_IC50_3p9.set_ylim(0,4)
    ax_FI_IC50_3p9.set_yticks([0,1,2,3])
    ax_FI_IC50_3p9.tick_params(axis="both",which="major",labelsize=16)
    ax_FI_IC50_3p9.grid(True)
    # COLUMN 2
    ax_BW_IC50_6.set_xlabel("Time (days)",fontsize=18)
    ax_BW_IC50_6.set_ylabel("Total body mass\n(g)",fontsize=18)
    ax_BW_IC50_6.set_title("Potency fitted to data",fontsize=18)
    ax_BW_IC50_6.set_xlim(-1,24)
    ax_BW_IC50_6.set_xticks([0,7,14,21])
    ax_BW_IC50_6.set_ylim(25,50)
    ax_BW_IC50_6.set_yticks([25,35,45])
    ax_BW_IC50_6.tick_params(axis="both",which="major",labelsize=16)
    ax_BW_IC50_6.grid(True)
    ax_FI_IC50_6.set_xlabel("Time (days)",fontsize=18)
    ax_FI_IC50_6.set_ylabel("Food intake\n(g/day)",fontsize=18)
    ax_FI_IC50_6.set_xlim(-1,24)
    ax_FI_IC50_6.set_xticks([0,7,14,21])
    ax_FI_IC50_6.set_ylim(0,4)
    ax_FI_IC50_6.set_yticks([0,1,2,3])
    ax_FI_IC50_6.tick_params(axis="both",which="major",labelsize=16)
    ax_FI_IC50_6.grid(True)
    # COLUMN 3
    ax_Sema.set_xlabel("Time (days)",fontsize=18)
    ax_Sema.set_ylabel("Plasma semaglutide\n(nmol/L)",fontsize=18)
    ax_Sema.set_title("Semaglutide exposure",fontsize=18)
    ax_Sema.set_yscale("log")
    ax_Sema.set_xlim(-1,24)
    ax_Sema.set_xticks([0,7,14,21])
    ax_Sema.set_ylim(1e-1,1e3)
    ax_Sema.set_yticks([1e-1,1e0,1e1,1e2,1e3])
    ax_Sema.tick_params(axis="both",which="major",labelsize=16)
    ax_Sema.grid(True)
    ax_Sema.legend(loc="lower right",fontsize=10,framealpha=1)
    fig.subplots_adjust(left=0.07,right=0.993,top=0.96,bottom=0.065,wspace=0.3,hspace=0.3)
    plt.show()


# --- AZ study ---
def plot_AZstudy_uncertainty(Model_Appetite,model_BW, Model_PK,model_GI, data, OptimisedAppetiteParams,OptimisedBWparams, OptimisedPKparams, OptimisedGIparams, FastPlot):

    OptimisedAppetiteParams = OptimisedAppetiteParams.to_numpy(dtype=float)
    OptimisedBWparams       = OptimisedBWparams.to_numpy(dtype=float)
    OptimisedPKparams       = OptimisedPKparams.to_numpy(dtype=float)
    OptimisedGIparams       = OptimisedGIparams.to_numpy(dtype=float)

    Best_AppetiteParams = np.asarray(OptimisedAppetiteParams[0][:-1], dtype=float)
    Best_BWParams = np.asarray(OptimisedBWparams[0][:-1], dtype=float)
    Best_PKParams = np.asarray(OptimisedPKparams[0][:-1], dtype=float)
    Best_GIParams = np.asarray(OptimisedGIparams[0][:-1], dtype=float)

   # Calibrated to data, the following parameters are used for the simulations
    PhysicalActivity = 0.085
    Recovery = 2.3
    IC50 = 2.8
    hill = 2

    def weekly_cumulative_food_intake_from_ei_ac(ei_ac, time, kcal_per_g, week_length=7):
        ei_ac = np.asarray(ei_ac, dtype=float)
        time = np.asarray(time, dtype=float)
        weekly_food = np.zeros_like(ei_ac)
        week_idx = np.floor(time / week_length).astype(int)
        for w in np.unique(week_idx):
            idx = np.where(week_idx == w)[0]
            weekly_food[idx] = (ei_ac[idx] - ei_ac[idx[0]]) / kcal_per_g
        return weekly_food
    AZ_HFD_kcal_per_g = 5.24
    def daily_min(values, TimeRes_exposure):
        values = np.asarray(values, dtype=float)
        points_per_day = int(24 * TimeRes_exposure)
        n_days = (len(values) - 1) // points_per_day
        return values[:n_days*points_per_day].reshape(n_days, points_per_day).min(axis=1)
    TimeRes_exposure = 12

    RecalculateCost = False # True or false, use to recalculate the cost function for each parameter vector (this takes a long time))
    def _validate_param(VectorNumber):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_AZstudy(Model_Appetite, model_BW, Model_PK, model_GI, data, AppetiteParams, Best_BWParams, Best_PKParams, Best_GIParams, TimeRes_exposure, PhysicalActivity, Recovery, IC50, hill)
            idx_ei_ac = sims["Vehicle"].feature_names.index("EI_AC")
            idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
            idx_glucose = sims["Vehicle"].feature_names.index("Plasma_Glucose")
            idx_insulin = sims["Vehicle"].feature_names.index("Plasma_Insulin")
            idx_exposure = sims["Vehicle"].feature_names.index("Plasma_Exenatide")
            t_BW_regular = np.asarray(data["AZstudy"]["timeBW"][0:22], dtype=float)
            t_BW_Sema10 = np.asarray(data["AZstudy"]["timeBW"][0:34], dtype=float)
            t_FI_regular = np.asarray(data["AZstudy"]["timeFI"][0:3], dtype=float)
            t_FI_Sema10 = np.asarray(data["AZstudy"]["timeFI"][0:4], dtype=float)
            BW_Vehicle = np.interp(t_BW_regular, sims["Vehicle"].time_vector, sims["Vehicle"].feature_values[:, idx_bw])
            BW_Sema0p3 = np.interp(t_BW_regular, sims["Sema0p3"].time_vector, sims["Sema0p3"].feature_values[:, idx_bw])
            BW_Sema2 = np.interp(t_BW_regular, sims["Sema2"].time_vector, sims["Sema2"].feature_values[:, idx_bw])
            BW_Sema4 = np.interp(t_BW_regular, sims["Sema4_Q2D"].time_vector, sims["Sema4_Q2D"].feature_values[:, idx_bw])
            BW_Sema10 = np.interp(t_BW_Sema10, sims["Sema10"].time_vector, sims["Sema10"].feature_values[:, idx_bw])
            FI_Vehicle = weekly_cumulative_food_intake_from_ei_ac(sims["Vehicle"].feature_values[:, idx_ei_ac], sims["Vehicle"].time_vector, AZ_HFD_kcal_per_g)
            FI_Sema0p3 = weekly_cumulative_food_intake_from_ei_ac(sims["Sema0p3"].feature_values[:, idx_ei_ac], sims["Sema0p3"].time_vector, AZ_HFD_kcal_per_g)
            FI_Sema2 = weekly_cumulative_food_intake_from_ei_ac(sims["Sema2"].feature_values[:, idx_ei_ac], sims["Sema2"].time_vector, AZ_HFD_kcal_per_g)
            FI_Sema4 = weekly_cumulative_food_intake_from_ei_ac(sims["Sema4_Q2D"].feature_values[:, idx_ei_ac], sims["Sema4_Q2D"].time_vector, AZ_HFD_kcal_per_g)
            FI_Sema10 = weekly_cumulative_food_intake_from_ei_ac(sims["Sema10"].feature_values[:, idx_ei_ac], sims["Sema10"].time_vector, AZ_HFD_kcal_per_g)
            FI_Vehicle_atData = np.interp(t_FI_regular - (sims["Vehicle"].time_vector[1] - sims["Vehicle"].time_vector[0]), sims["Vehicle"].time_vector, FI_Vehicle)
            FI_Sema0p3_atData = np.interp(t_FI_regular - (sims["Sema0p3"].time_vector[1] - sims["Sema0p3"].time_vector[0]), sims["Sema0p3"].time_vector, FI_Sema0p3)
            FI_Sema2_atData = np.interp(t_FI_regular - (sims["Sema2"].time_vector[1] - sims["Sema2"].time_vector[0]), sims["Sema2"].time_vector, FI_Sema2)
            FI_Sema4_atData = np.interp(t_FI_regular - (sims["Sema4_Q2D"].time_vector[1] - sims["Sema4_Q2D"].time_vector[0]), sims["Sema4_Q2D"].time_vector, FI_Sema4)
            FI_Sema10_atData = np.interp(t_FI_Sema10 - (sims["Sema10"].time_vector[1] - sims["Sema10"].time_vector[0]), sims["Sema10"].time_vector, FI_Sema10)
            Glucose_Vehicle = daily_min(sims["Vehicle"].feature_values[:, idx_glucose], TimeRes_exposure)
            Glucose_Sema0p3 = daily_min(sims["Sema0p3"].feature_values[:, idx_glucose], TimeRes_exposure)
            Glucose_Sema2 = daily_min(sims["Sema2"].feature_values[:, idx_glucose], TimeRes_exposure)
            Glucose_Sema4 = daily_min(sims["Sema4_Q2D"].feature_values[:, idx_glucose], TimeRes_exposure)
            Glucose_Sema10 = daily_min(sims["Sema10"].feature_values[:, idx_glucose], TimeRes_exposure)
            Insulin_Vehicle = daily_min(sims["Vehicle"].feature_values[:, idx_insulin], TimeRes_exposure) / 1000
            Insulin_Sema0p3 = daily_min(sims["Sema0p3"].feature_values[:, idx_insulin], TimeRes_exposure) / 1000
            Insulin_Sema2 = daily_min(sims["Sema2"].feature_values[:, idx_insulin], TimeRes_exposure) / 1000
            Insulin_Sema4 = daily_min(sims["Sema4_Q2D"].feature_values[:, idx_insulin], TimeRes_exposure) / 1000
            Insulin_Sema10 = daily_min(sims["Sema10"].feature_values[:, idx_insulin], TimeRes_exposure) / 1000
            t_Glucose = float(np.asarray(data["AZstudy"]["timeGlucose"][0], dtype=float))
            t_Insulin = float(np.asarray(data["AZstudy"]["timeInsulin"][0], dtype=float))
            GlucoseValues = np.array([np.interp(t_Glucose, np.arange(len(Glucose_Vehicle), dtype=float), Glucose_Vehicle), np.interp(t_Glucose, np.arange(len(Glucose_Sema0p3), dtype=float), Glucose_Sema0p3), np.interp(t_Glucose, np.arange(len(Glucose_Sema2), dtype=float), Glucose_Sema2), np.interp(t_Glucose, np.arange(len(Glucose_Sema4), dtype=float), Glucose_Sema4), np.interp(t_Glucose, np.arange(len(Glucose_Sema10), dtype=float), Glucose_Sema10)])
            InsulinValues = np.array([np.interp(t_Insulin, np.arange(len(Insulin_Vehicle), dtype=float), Insulin_Vehicle), np.interp(t_Insulin, np.arange(len(Insulin_Sema0p3), dtype=float), Insulin_Sema0p3), np.interp(t_Insulin, np.arange(len(Insulin_Sema2), dtype=float), Insulin_Sema2), np.interp(t_Insulin, np.arange(len(Insulin_Sema4), dtype=float), Insulin_Sema4), np.interp(t_Insulin, np.arange(len(Insulin_Sema10), dtype=float), Insulin_Sema10)])
            t_Exposure = float(np.asarray(data["AZstudy"]["timeExposure"][0], dtype=float))
            t_ExposureLateEnd = float(np.asarray(data["AZstudy"]["timeExposureLateEnd"][0], dtype=float))
            ExposureValues = np.array([np.interp(t_Exposure, sims["Sema2"].time_vector, sims["Sema2"].feature_values[:, idx_exposure]), np.interp(t_Exposure, sims["Sema4_Q2D"].time_vector, sims["Sema4_Q2D"].feature_values[:, idx_exposure]), np.interp(t_ExposureLateEnd, sims["Sema10"].time_vector, sims["Sema10"].feature_values[:, idx_exposure])])
            SimulatedValues = np.concatenate([BW_Vehicle, BW_Sema0p3, BW_Sema2, BW_Sema4, BW_Sema10, FI_Vehicle_atData, FI_Sema0p3_atData, FI_Sema2_atData, FI_Sema4_atData, FI_Sema10_atData, InsulinValues, GlucoseValues, ExposureValues])
            DataMean = np.asarray(data["AZstudy"]["all_mean"][0:151], dtype=float)
            DataWeightedSD = np.asarray(data["AZstudy"]["all_weighted_sd"][0:151], dtype=float)
            if len(SimulatedValues) != len(DataMean) or len(DataMean) != len(DataWeightedSD):
                raise ValueError(f"Length mismatch: simulation={len(SimulatedValues)}, mean={len(DataMean)}, SD={len(DataWeightedSD)}")
            CostValidation = np.sum(((DataMean - SimulatedValues) / DataWeightedSD) ** 2)
            return CostValidation
        except Exception as error:
            print(f"Vector {VectorNumber} failed: {error}")
            return np.inf
    if RecalculateCost:
        ValidationCosts = Parallel(n_jobs=1, verbose=10)(delayed(_validate_param)(VectorNumber) for VectorNumber in range(len(OptimisedAppetiteParams)))
        BestVectorNumber = int(np.argmin(ValidationCosts))
        BestValidationCost = float(ValidationCosts[BestVectorNumber])
        print(f"Best validation cost: {BestValidationCost}")
        print(f"Best parameter vector index: {BestVectorNumber}")

    n_select = 300
    if FastPlot:
        n_select = 10
    idx = np.linspace(0, OptimisedAppetiteParams.shape[0] - 1, n_select, dtype=int)
    OptimisedAppetiteParams = OptimisedAppetiteParams[idx]
    OptimisedBWparams       = OptimisedBWparams[idx]
    OptimisedGIparams       = OptimisedGIparams[idx]

    n_select = 500
    if FastPlot:
        n_select = 100
    idx = np.linspace(0, OptimisedPKparams.shape[0] - 1, n_select, dtype=int)
    OptimisedPKparams = OptimisedPKparams[idx]


    # NotCalibrated
    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_AZstudy(Model_Appetite,model_BW,Model_PK,model_GI, data, AppetiteParams,Best_BWParams,Best_PKParams,Best_GIParams, TimeRes_exposure, Best_BWParams[-1],1,1,1)
        except Exception as e:
            print("AZ simulation failed:", repr(e))
            continue
        idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
        Vehicle_BWNoCal_new = sims["Vehicle"].feature_values[:, idx_bw]
        if first:
            Vehicle_BWNoCal_min,Vehicle_BWNoCal_max,best_Vehicle_BWNoCal = Vehicle_BWNoCal_new.copy(),Vehicle_BWNoCal_new.copy(),Vehicle_BWNoCal_new.copy()
            first = False
        else:
            Vehicle_BWNoCal_min = np.minimum(Vehicle_BWNoCal_min, Vehicle_BWNoCal_new)
            Vehicle_BWNoCal_max = np.maximum(Vehicle_BWNoCal_max, Vehicle_BWNoCal_new)


    first = True
    for VectorNumber in range(len(OptimisedAppetiteParams)):
        AppetiteParams = np.asarray(OptimisedAppetiteParams[VectorNumber][:-1], dtype=float)
        try:
            sims = Simulate_AZstudy(Model_Appetite,model_BW,Model_PK,model_GI, data, AppetiteParams,Best_BWParams,Best_PKParams,Best_GIParams, TimeRes_exposure, PhysicalActivity,Recovery,IC50,hill)
        except Exception as e:
            print("AZ simulation failed:", repr(e))
            continue
        idx_ei_ac = sims["Vehicle"].feature_names.index("EI_AC")
        idx_bw = sims["Vehicle"].feature_names.index("Body_weight")
        idx_glucose = sims["Vehicle"].feature_names.index("Plasma_Glucose")
        idx_insulin = sims["Vehicle"].feature_names.index("Plasma_Insulin")
        idx_lambda       = sims["Vehicle"].feature_names.index("lambda")
        idx_lambda_delay = sims["Vehicle"].feature_names.index("lambda_delay")
        # FI
        Vehicle_FI_new = weekly_cumulative_food_intake_from_ei_ac(sims["Vehicle"].feature_values[:, idx_ei_ac], sims["Vehicle"].time_vector, AZ_HFD_kcal_per_g)
        Sema0p3_FI_new = weekly_cumulative_food_intake_from_ei_ac(sims["Sema0p3"].feature_values[:, idx_ei_ac], sims["Sema0p3"].time_vector, AZ_HFD_kcal_per_g)
        Sema2_FI_new = weekly_cumulative_food_intake_from_ei_ac(sims["Sema2"].feature_values[:, idx_ei_ac], sims["Sema2"].time_vector, AZ_HFD_kcal_per_g)
        Sema4_FI_new = weekly_cumulative_food_intake_from_ei_ac(sims["Sema4_Q2D"].feature_values[:, idx_ei_ac], sims["Sema4_Q2D"].time_vector, AZ_HFD_kcal_per_g)
        Sema10_FI_new = weekly_cumulative_food_intake_from_ei_ac(sims["Sema10"].feature_values[:, idx_ei_ac], sims["Sema10"].time_vector, AZ_HFD_kcal_per_g)
        # BW
        Vehicle_BW_new = sims["Vehicle"].feature_values[:, idx_bw]
        Sema0p3_BW_new = sims["Sema0p3"].feature_values[:, idx_bw]
        Sema2_BW_new = sims["Sema2"].feature_values[:, idx_bw]
        Sema4_BW_new = sims["Sema4_Q2D"].feature_values[:, idx_bw]
        Sema10_BW_new = sims["Sema10"].feature_values[:, idx_bw]
        # Glucose: one minimum value per day
        Vehicle_Glucose_new = daily_min(sims["Vehicle"].feature_values[:, idx_glucose], TimeRes_exposure)
        Sema0p3_Glucose_new = daily_min(sims["Sema0p3"].feature_values[:, idx_glucose], TimeRes_exposure)
        Sema2_Glucose_new = daily_min(sims["Sema2"].feature_values[:, idx_glucose], TimeRes_exposure)
        Sema4_Glucose_new = daily_min(sims["Sema4_Q2D"].feature_values[:, idx_glucose], TimeRes_exposure)
        Sema10_Glucose_new = daily_min(sims["Sema10"].feature_values[:, idx_glucose], TimeRes_exposure)
        # Insulin: one minimum value per day
        Vehicle_Insulin_new = daily_min(sims["Vehicle"].feature_values[:, idx_insulin], TimeRes_exposure)/1000
        Sema0p3_Insulin_new = daily_min(sims["Sema0p3"].feature_values[:, idx_insulin], TimeRes_exposure)/1000
        Sema2_Insulin_new = daily_min(sims["Sema2"].feature_values[:, idx_insulin], TimeRes_exposure)/1000
        Sema4_Insulin_new = daily_min(sims["Sema4_Q2D"].feature_values[:, idx_insulin], TimeRes_exposure)/1000
        Sema10_Insulin_new = daily_min(sims["Sema10"].feature_values[:, idx_insulin], TimeRes_exposure)/1000
        # ===== lambda_delay =====
        lambda_delay_vehicle_new = sims["Vehicle"].feature_values[:, idx_lambda_delay]
        if first:
            best_sims = sims
            # FI
            Vehicle_FI_min,Vehicle_FI_max,best_Vehicle_FI = Vehicle_FI_new.copy(),Vehicle_FI_new.copy(),Vehicle_FI_new.copy()
            Sema0p3_FI_min,Sema0p3_FI_max,best_Sema0p3_FI = Sema0p3_FI_new.copy(),Sema0p3_FI_new.copy(),Sema0p3_FI_new.copy()
            Sema2_FI_min,Sema2_FI_max,best_Sema2_FI = Sema2_FI_new.copy(),Sema2_FI_new.copy(),Sema2_FI_new.copy()
            Sema4_FI_min,Sema4_FI_max,best_Sema4_FI = Sema4_FI_new.copy(),Sema4_FI_new.copy(),Sema4_FI_new.copy()
            Sema10_FI_min,Sema10_FI_max,best_Sema10_FI = Sema10_FI_new.copy(),Sema10_FI_new.copy(),Sema10_FI_new.copy()
            # BW
            Vehicle_BW_min,Vehicle_BW_max,best_Vehicle_BW = Vehicle_BW_new.copy(),Vehicle_BW_new.copy(),Vehicle_BW_new.copy()
            Sema0p3_BW_min,Sema0p3_BW_max,best_Sema0p3_BW = Sema0p3_BW_new.copy(),Sema0p3_BW_new.copy(),Sema0p3_BW_new.copy()
            Sema2_BW_min,Sema2_BW_max,best_Sema2_BW = Sema2_BW_new.copy(),Sema2_BW_new.copy(),Sema2_BW_new.copy()
            Sema4_BW_min,Sema4_BW_max,best_Sema4_BW = Sema4_BW_new.copy(),Sema4_BW_new.copy(),Sema4_BW_new.copy()
            Sema10_BW_min,Sema10_BW_max,best_Sema10_BW = Sema10_BW_new.copy(),Sema10_BW_new.copy(),Sema10_BW_new.copy()
            # Glucose
            Vehicle_Glucose_min,Vehicle_Glucose_max,best_Vehicle_Glucose = Vehicle_Glucose_new.copy(),Vehicle_Glucose_new.copy(),Vehicle_Glucose_new.copy()
            Sema0p3_Glucose_min,Sema0p3_Glucose_max,best_Sema0p3_Glucose = Sema0p3_Glucose_new.copy(),Sema0p3_Glucose_new.copy(),Sema0p3_Glucose_new.copy()
            Sema2_Glucose_min,Sema2_Glucose_max,best_Sema2_Glucose = Sema2_Glucose_new.copy(),Sema2_Glucose_new.copy(),Sema2_Glucose_new.copy()
            Sema4_Glucose_min,Sema4_Glucose_max,best_Sema4_Glucose = Sema4_Glucose_new.copy(),Sema4_Glucose_new.copy(),Sema4_Glucose_new.copy()
            Sema10_Glucose_min,Sema10_Glucose_max,best_Sema10_Glucose = Sema10_Glucose_new.copy(),Sema10_Glucose_new.copy(),Sema10_Glucose_new.copy()
            # Insulin
            Vehicle_Insulin_min,Vehicle_Insulin_max,best_Vehicle_Insulin = Vehicle_Insulin_new.copy(),Vehicle_Insulin_new.copy(),Vehicle_Insulin_new.copy()
            Sema0p3_Insulin_min,Sema0p3_Insulin_max,best_Sema0p3_Insulin = Sema0p3_Insulin_new.copy(),Sema0p3_Insulin_new.copy(),Sema0p3_Insulin_new.copy()
            Sema2_Insulin_min,Sema2_Insulin_max,best_Sema2_Insulin = Sema2_Insulin_new.copy(),Sema2_Insulin_new.copy(),Sema2_Insulin_new.copy()
            Sema4_Insulin_min,Sema4_Insulin_max,best_Sema4_Insulin = Sema4_Insulin_new.copy(),Sema4_Insulin_new.copy(),Sema4_Insulin_new.copy()
            Sema10_Insulin_min,Sema10_Insulin_max,best_Sema10_Insulin = Sema10_Insulin_new.copy(),Sema10_Insulin_new.copy(),Sema10_Insulin_new.copy()
            # ===== lambda_delay =====
            lambda_delay_vehicle_min,lambda_delay_vehicle_max = lambda_delay_vehicle_new.copy(),lambda_delay_vehicle_new.copy()
            first = False
        else:
            # FI
            Vehicle_FI_min = np.minimum(Vehicle_FI_min, Vehicle_FI_new)
            Vehicle_FI_max = np.maximum(Vehicle_FI_max, Vehicle_FI_new)
            Sema0p3_FI_min = np.minimum(Sema0p3_FI_min, Sema0p3_FI_new)
            Sema0p3_FI_max = np.maximum(Sema0p3_FI_max, Sema0p3_FI_new)
            Sema2_FI_min = np.minimum(Sema2_FI_min, Sema2_FI_new)
            Sema2_FI_max = np.maximum(Sema2_FI_max, Sema2_FI_new)
            Sema4_FI_min = np.minimum(Sema4_FI_min, Sema4_FI_new)
            Sema4_FI_max = np.maximum(Sema4_FI_max, Sema4_FI_new)
            Sema10_FI_min = np.minimum(Sema10_FI_min, Sema10_FI_new)
            Sema10_FI_max = np.maximum(Sema10_FI_max, Sema10_FI_new)
            # BW
            Vehicle_BW_min = np.minimum(Vehicle_BW_min, Vehicle_BW_new)
            Vehicle_BW_max = np.maximum(Vehicle_BW_max, Vehicle_BW_new)
            Sema0p3_BW_min = np.minimum(Sema0p3_BW_min, Sema0p3_BW_new)
            Sema0p3_BW_max = np.maximum(Sema0p3_BW_max, Sema0p3_BW_new)
            Sema2_BW_min = np.minimum(Sema2_BW_min, Sema2_BW_new)
            Sema2_BW_max = np.maximum(Sema2_BW_max, Sema2_BW_new)
            Sema4_BW_min = np.minimum(Sema4_BW_min, Sema4_BW_new)
            Sema4_BW_max = np.maximum(Sema4_BW_max, Sema4_BW_new)
            Sema10_BW_min = np.minimum(Sema10_BW_min, Sema10_BW_new)
            Sema10_BW_max = np.maximum(Sema10_BW_max, Sema10_BW_new)
            # Glucose
            Vehicle_Glucose_min = np.minimum(Vehicle_Glucose_min, Vehicle_Glucose_new)
            Vehicle_Glucose_max = np.maximum(Vehicle_Glucose_max, Vehicle_Glucose_new)
            Sema0p3_Glucose_min = np.minimum(Sema0p3_Glucose_min, Sema0p3_Glucose_new)
            Sema0p3_Glucose_max = np.maximum(Sema0p3_Glucose_max, Sema0p3_Glucose_new)
            Sema2_Glucose_min = np.minimum(Sema2_Glucose_min, Sema2_Glucose_new)
            Sema2_Glucose_max = np.maximum(Sema2_Glucose_max, Sema2_Glucose_new)
            Sema4_Glucose_min = np.minimum(Sema4_Glucose_min, Sema4_Glucose_new)
            Sema4_Glucose_max = np.maximum(Sema4_Glucose_max, Sema4_Glucose_new)
            Sema10_Glucose_min = np.minimum(Sema10_Glucose_min, Sema10_Glucose_new)
            Sema10_Glucose_max = np.maximum(Sema10_Glucose_max, Sema10_Glucose_new)
            # Insulin
            Vehicle_Insulin_min = np.minimum(Vehicle_Insulin_min, Vehicle_Insulin_new)
            Vehicle_Insulin_max = np.maximum(Vehicle_Insulin_max, Vehicle_Insulin_new)
            Sema0p3_Insulin_min = np.minimum(Sema0p3_Insulin_min, Sema0p3_Insulin_new)
            Sema0p3_Insulin_max = np.maximum(Sema0p3_Insulin_max, Sema0p3_Insulin_new)
            Sema2_Insulin_min = np.minimum(Sema2_Insulin_min, Sema2_Insulin_new)
            Sema2_Insulin_max = np.maximum(Sema2_Insulin_max, Sema2_Insulin_new)
            Sema4_Insulin_min = np.minimum(Sema4_Insulin_min, Sema4_Insulin_new)
            Sema4_Insulin_max = np.maximum(Sema4_Insulin_max, Sema4_Insulin_new)
            Sema10_Insulin_min = np.minimum(Sema10_Insulin_min, Sema10_Insulin_new)
            Sema10_Insulin_max = np.maximum(Sema10_Insulin_max, Sema10_Insulin_new)
            # ----- lambda_delay -----
            lambda_delay_vehicle_min = np.minimum(lambda_delay_vehicle_min, lambda_delay_vehicle_new)
            lambda_delay_vehicle_max = np.maximum(lambda_delay_vehicle_max, lambda_delay_vehicle_new)

    time_regular = best_sims["Vehicle"].time_vector
    time_Sema10 = best_sims["Sema10"].time_vector
    timeDay = np.arange(len(best_Vehicle_Glucose), dtype=float)
    timeDay_Sema10 = np.arange(len(best_Sema10_Glucose), dtype=float)

    # ---------- Day/night cycle ----------
    def compute_is_night(time, phase_shift_hours=16):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        return cycle > 0
    IsNight = compute_is_night(time_Sema10)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time_Sema10) - 1]))

    colors = {
        "Vehicle": "blue",
        "Sema0p3": "teal",
        "Sema2": "darkorange",
        "Sema4_Q2D": "goldenrod",
        "Sema10": "firebrick"
    }

    # ======================
    # Figure 1
    # ======================
    def add_night_shading(ax,time_days,phase_shift_hours=16):
        time_days = np.asarray(time_days,dtype=float)
        IsNight_local = np.sin(2*np.pi*(time_days*24+phase_shift_hours)/24) > 0
        change_idx_local = np.where(np.diff(IsNight_local.astype(int)) != 0)[0]+1
        edges_local = np.concatenate(([0],change_idx_local,[len(time_days)-1]))
        for i in range(len(edges_local)-1):
            if IsNight_local[edges_local[i]]:
                ax.axvspan(time_days[edges_local[i]]/7,time_days[edges_local[i+1]]/7,color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    time_NoCal = np.asarray(time_regular,dtype=float)
    time_Cal = np.asarray(best_sims["Vehicle"].time_vector,dtype=float)
    Vehicle_BWNoCal_min = np.asarray(Vehicle_BWNoCal_min,dtype=float)
    Vehicle_BWNoCal_max = np.asarray(Vehicle_BWNoCal_max,dtype=float)
    best_Vehicle_BWNoCal = np.asarray(best_Vehicle_BWNoCal,dtype=float)
    Vehicle_BW_min = np.asarray(Vehicle_BW_min,dtype=float)
    Vehicle_BW_max = np.asarray(Vehicle_BW_max,dtype=float)
    best_Vehicle_BW = np.asarray(best_Vehicle_BW,dtype=float)
    lambda_delay_vehicle_min = np.asarray(lambda_delay_vehicle_min,dtype=float)
    lambda_delay_vehicle_max = np.asarray(lambda_delay_vehicle_max,dtype=float)
    best_lambda_delay_vehicle = np.asarray(best_sims["Vehicle"].feature_values[:,idx_lambda_delay],dtype=float)
    fig1,axes1 = plt.subplots(1,3,figsize=(18,5))
    ax_NoCal,ax_Cal,ax_lambda = axes1[0],axes1[1],axes1[2]
    # COLUMN 1: Not calibrated, Vehicle BW only
    ax_NoCal.fill_between(time_NoCal/7,Vehicle_BWNoCal_min,Vehicle_BWNoCal_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_NoCal.plot(time_NoCal/7,best_Vehicle_BWNoCal,color=colors["Vehicle"],lw=2,label="Simulation")
    ax_NoCal.errorbar(np.array(data["AZstudy"]["timeBW"][0:22],dtype=float)/7,np.array(data["AZstudy"]["VehicleBW"][0:22],dtype=float),yerr=np.array(data["AZstudy"]["VehicleBW_stdev"][0:22],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2,label="Predicted data",zorder=10)
    add_night_shading(ax_NoCal,time_NoCal)
    # COLUMN 2: Calibrated, Vehicle BW only
    ax_Cal.fill_between(time_Cal/7,Vehicle_BW_min,Vehicle_BW_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_Cal.plot(time_Cal/7,best_Vehicle_BW,color=colors["Vehicle"],lw=2,label="Simulation")
    ax_Cal.errorbar(np.array(data["AZstudy"]["timeBW"][0:22],dtype=float)/7,np.array(data["AZstudy"]["VehicleBW"][0:22],dtype=float),yerr=np.array(data["AZstudy"]["VehicleBW_stdev"][0:22],dtype=float),fmt="none",color=colors["Vehicle"],capsize=4,elinewidth=2,capthick=2,label="Predicted data",zorder=10)
    ax_Cal.errorbar(np.array(data["AZstudy"]["timeBW"][7],dtype=float)/7,np.array(data["AZstudy"]["VehicleBW"][7],dtype=float),fmt="x",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Initial stress",zorder=100)
    ax_Cal.errorbar(np.array(data["AZstudy"]["timeBW"][21],dtype=float)/7,np.array(data["AZstudy"]["VehicleBW"][21],dtype=float),fmt="+",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Physical activity",zorder=100)
    add_night_shading(ax_Cal,time_Cal)
    # COLUMN 3: Physical-activity calibration dynamics
    ax_lambda.fill_between(time_Cal/7,lambda_delay_vehicle_min,lambda_delay_vehicle_max,color=colors["Vehicle"],alpha=0.15,edgecolor="none")
    ax_lambda.plot(time_Cal/7,best_lambda_delay_vehicle,color=colors["Vehicle"],lw=2,ls="--",label="Adaptation rate (from estimation data)")
    ax_lambda.errorbar(time_Cal[0]/7,Recovery*PhysicalActivity,fmt="x",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Initial stress",zorder=100)
    ax_lambda.errorbar(np.array(data["AZstudy"]["timeBW"][22],dtype=float)/7,PhysicalActivity,fmt="+",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Physical activity",zorder=100)
    ax_lambda.axhline(y=Best_BWParams[-1], color="black", linestyle="--", linewidth=2, label="Used in blind prediction (Fig. S3B i)")
    add_night_shading(ax_lambda,time_Cal)
    # Formatting
    titles = ["Physical activity assumed\nfrom Guo and Hall HFD group","Calibrating activity to vehicle group","Calibration visualised"]
    for i,ax in enumerate([ax_NoCal,ax_Cal]):
        ax.set_title(titles[i],fontsize=18)
        ax.set_xlabel("Time (weeks)",fontsize=18)
        ax.set_ylabel("Total body mass\n(g)",fontsize=18)
        ax.set_xlim(-0.1,3.2)
        ax.set_xticks([0,1,2,3])
        ax.set_ylim(40,55)
        ax.set_yticks([40,45,50,55])
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=16)
    ax_NoCal.legend(loc="best",fontsize=16,framealpha=1)
    ax_Cal.legend(loc="best",fontsize=16,framealpha=1)
    ax_lambda.set_title(titles[2],fontsize=18)
    ax_lambda.set_xlabel("Time (weeks)",fontsize=18)
    ax_lambda.set_ylabel("Physical activity\n(kcal/g/day)",fontsize=18)
    ax_lambda.set_xlim(-0.1,3.2)
    ax_lambda.set_xticks([0,1,2,3])
    ax_lambda.set_ylim(0,1)
    ax_lambda.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax_lambda.grid(True)
    ax_lambda.tick_params(axis="both",which="major",labelsize=16)
    ax_lambda.legend(loc="best",fontsize=16,framealpha=1)
    fig1.tight_layout()
    plt.show()


    # ======================
    # Figure 2
    # ======================
    fig, axes = plt.subplots(2,2,figsize=(16,11))
    ax_bw = axes[0,0]
    ax_fi = axes[0,1]
    ax_glucose = axes[1,0]
    ax_insulin = axes[1,1]
    # ROW 1+2 COL 1: BODY WEIGHT
    ax_bw.fill_between(time_regular/7, Vehicle_BW_min, Vehicle_BW_max, color=colors["Vehicle"], alpha=0.15, edgecolor="none")
    ax_bw.plot(time_regular/7, best_Vehicle_BW, color=colors["Vehicle"], lw=2, label="Vehicle")
    ax_bw.fill_between(time_regular/7, Sema0p3_BW_min, Sema0p3_BW_max, color=colors["Sema0p3"], alpha=0.15, edgecolor="none")
    ax_bw.plot(time_regular/7, best_Sema0p3_BW, color=colors["Sema0p3"], lw=2, label="0.3 nmol/kg QD")
    ax_bw.fill_between(time_regular/7, Sema2_BW_min, Sema2_BW_max, color=colors["Sema2"], alpha=0.15, edgecolor="none")
    ax_bw.plot(time_regular/7, best_Sema2_BW, color=colors["Sema2"], lw=2, label="2 nmol/kg QD")
    ax_bw.fill_between(time_regular/7, Sema4_BW_min, Sema4_BW_max, color=colors["Sema4_Q2D"], alpha=0.15, edgecolor="none")
    ax_bw.plot(time_regular/7, best_Sema4_BW, color=colors["Sema4_Q2D"], lw=2, label="4 nmol/kg Q2D")
    ax_bw.fill_between(time_Sema10/7, Sema10_BW_min, Sema10_BW_max, color=colors["Sema10"], alpha=0.15, edgecolor="none")
    ax_bw.plot(time_Sema10/7, best_Sema10_BW, color=colors["Sema10"], lw=2, label="10 nmol/kg QD")
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][0:22], dtype=float)/7, np.array(data["AZstudy"]["VehicleBW"][0:22], dtype=float), yerr=np.array(data["AZstudy"]["VehicleBW_stdev"][0:22], dtype=float),fmt="none", color=colors["Vehicle"], capsize=4, elinewidth=2, capthick=2)
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][0:22], dtype=float)/7, np.array(data["AZstudy"]["Sema0p3BW"][0:22], dtype=float), yerr=np.array(data["AZstudy"]["Sema0p3BW_stdev"][0:22], dtype=float), fmt="none", color=colors["Sema0p3"], capsize=4, elinewidth=2, capthick=2)
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][0:22], dtype=float)/7, np.array(data["AZstudy"]["Sema2BW"][0:22], dtype=float), yerr=np.array(data["AZstudy"]["Sema2BW_stdev"][0:22], dtype=float), fmt="none", color=colors["Sema2"], capsize=4, elinewidth=2, capthick=2)
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][0:22], dtype=float)/7, np.array(data["AZstudy"]["Sema4_Q2DBW"][0:22], dtype=float), yerr=np.array(data["AZstudy"]["Sema4_Q2DBW_stdev"][0:22], dtype=float), fmt="none", color=colors["Sema4_Q2D"], capsize=4, elinewidth=2, capthick=2)
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][0:34], dtype=float)/7, np.array(data["AZstudy"]["Sema10BW"][0:34], dtype=float), yerr=np.array(data["AZstudy"]["Sema10BW_stdev"][0:34], dtype=float), fmt="none", color=colors["Sema10"], capsize=4, elinewidth=2, capthick=2)
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][7],dtype=float)/7,np.array(data["AZstudy"]["VehicleBW"][7],dtype=float),fmt="x",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Initial stress",zorder=100)
    ax_bw.errorbar(np.array(data["AZstudy"]["timeBW"][21],dtype=float)/7,np.array(data["AZstudy"]["VehicleBW"][21],dtype=float),fmt="x",color="black",capsize=0,elinewidth=0,capthick=0,markersize=20,markeredgewidth=5,label="Calibration: Physical activity",zorder=100)
    for day in range(22):
        ax_bw.plot([day/7 - 0.1/7, day/7 + 0.1/7], [40, 40], color='black', lw=18, solid_capstyle='butt')
    ax_bw.set_xlabel("Time (weeks)", fontsize=18)
    ax_bw.set_ylabel("Total body mass\n(g)", fontsize=18)
    ax_bw.set_xlim(-0.1, 4.8)
    ax_bw.set_xticks([0,1,2,3,4])
    ax_bw.set_ylim(40, 55)
    ax_bw.set_yticks([40,45,50,55])
    # ROW 1+2 COL 2: FOOD INTAKE
    ax_fi.fill_between(time_regular/7, Vehicle_FI_min, Vehicle_FI_max, color=colors["Vehicle"], alpha=0.15, edgecolor="none")
    ax_fi.plot(time_regular/7, best_Vehicle_FI, color=colors["Vehicle"], lw=2)
    ax_fi.fill_between(time_regular/7, Sema0p3_FI_min, Sema0p3_FI_max, color=colors["Sema0p3"], alpha=0.15, edgecolor="none")
    ax_fi.plot(time_regular/7, best_Sema0p3_FI, color=colors["Sema0p3"], lw=2)
    ax_fi.fill_between(time_regular/7, Sema2_FI_min, Sema2_FI_max, color=colors["Sema2"], alpha=0.15, edgecolor="none")
    ax_fi.plot(time_regular/7, best_Sema2_FI, color=colors["Sema2"], lw=2)
    ax_fi.fill_between(time_regular/7, Sema4_FI_min, Sema4_FI_max, color=colors["Sema4_Q2D"], alpha=0.15, edgecolor="none")
    ax_fi.plot(time_regular/7, best_Sema4_FI, color=colors["Sema4_Q2D"], lw=2)
    ax_fi.fill_between(time_Sema10/7, Sema10_FI_min, Sema10_FI_max, color=colors["Sema10"], alpha=0.15, edgecolor="none")
    ax_fi.plot(time_Sema10/7, best_Sema10_FI, color=colors["Sema10"], lw=2)
    ax_fi.errorbar(np.array(data["AZstudy"]["timeFI"][0:3], dtype=float)/7, np.array(data["AZstudy"]["VehicleFI"][0:3], dtype=float), yerr=np.array(data["AZstudy"]["VehicleFI_stdev"][0:3], dtype=float), fmt="none", color=colors["Vehicle"], capsize=4, elinewidth=2, capthick=2)
    ax_fi.errorbar(np.array(data["AZstudy"]["timeFI"][0:3], dtype=float)/7, np.array(data["AZstudy"]["Sema0p3FI"][0:3], dtype=float), yerr=np.array(data["AZstudy"]["Sema0p3FI_stdev"][0:3], dtype=float), fmt="none", color=colors["Sema0p3"], capsize=4, elinewidth=2, capthick=2)
    ax_fi.errorbar(np.array(data["AZstudy"]["timeFI"][0:3], dtype=float)/7, np.array(data["AZstudy"]["Sema2FI"][0:3], dtype=float), yerr=np.array(data["AZstudy"]["Sema2FI_stdev"][0:3], dtype=float), fmt="none",color=colors["Sema2"], capsize=4, elinewidth=2, capthick=2)
    ax_fi.errorbar(np.array(data["AZstudy"]["timeFI"][0:3], dtype=float)/7, np.array(data["AZstudy"]["Sema4_Q2DFI"][0:3], dtype=float), yerr=np.array(data["AZstudy"]["Sema4_Q2DFI_stdev"][0:3], dtype=float), fmt="none", color=colors["Sema4_Q2D"], capsize=4, elinewidth=2, capthick=2)
    ax_fi.errorbar(np.array(data["AZstudy"]["timeFI"][0:4], dtype=float)/7, np.array(data["AZstudy"]["Sema10FI"][0:4], dtype=float), yerr=np.array(data["AZstudy"]["Sema10FI_stdev"][0:4], dtype=float), fmt="none",color=colors["Sema10"], capsize=4, elinewidth=2, capthick=2)
    for day in range(22):
        ax_fi.plot([day/7 - 0.1/7, day/7 + 0.1/7], [-0.5, -0.5], color='black', lw=18, solid_capstyle='butt')
    ax_fi.set_xlabel("Time (weeks)", fontsize=18)
    ax_fi.set_ylabel("Weekly cumulative food intake\n(g)", fontsize=18)
    ax_fi.set_xlim(-0.1, 5.1)
    ax_fi.set_xticks([0,1,2,3,4,5])
    ax_fi.set_ylim(-0.5, 35)
    ax_fi.set_yticks([0,10,20,30])
    # ROW 2 COL 1: GLUCOSE
    ax_glucose.fill_between(timeDay/7, Vehicle_Glucose_min, Vehicle_Glucose_max, color=colors["Vehicle"], alpha=0.15, edgecolor="none")
    ax_glucose.plot(timeDay/7, best_Vehicle_Glucose, color=colors["Vehicle"], lw=2)
    ax_glucose.fill_between(timeDay/7, Sema0p3_Glucose_min, Sema0p3_Glucose_max, color=colors["Sema0p3"], alpha=0.15, edgecolor="none")
    ax_glucose.plot(timeDay/7, best_Sema0p3_Glucose, color=colors["Sema0p3"], lw=2)
    ax_glucose.fill_between(timeDay/7, Sema2_Glucose_min, Sema2_Glucose_max, color=colors["Sema2"], alpha=0.15, edgecolor="none")
    ax_glucose.plot(timeDay/7, best_Sema2_Glucose, color=colors["Sema2"], lw=2)
    ax_glucose.fill_between(timeDay/7, Sema4_Glucose_min, Sema4_Glucose_max, color=colors["Sema4_Q2D"], alpha=0.15, edgecolor="none")
    ax_glucose.plot(timeDay/7, best_Sema4_Glucose, color=colors["Sema4_Q2D"], lw=2)
    ax_glucose.fill_between(timeDay_Sema10/7, Sema10_Glucose_min, Sema10_Glucose_max, color=colors["Sema10"], alpha=0.15, edgecolor="none")
    ax_glucose.plot(timeDay_Sema10/7, best_Sema10_Glucose, color=colors["Sema10"], lw=2)
    ax_glucose.errorbar(np.array(data["AZstudy"]["timeGlucose"][0], dtype=float)/7, np.array(data["AZstudy"]["VehicleGlucose"][0], dtype=float), yerr=np.array(data["AZstudy"]["VehicleGlucose_stdev"][0], dtype=float), fmt="none", color=colors["Vehicle"], capsize=4, elinewidth=2, capthick=2)
    ax_glucose.errorbar(np.array(data["AZstudy"]["timeGlucose"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema0p3Glucose"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema0p3Glucose_stdev"][0], dtype=float), fmt="none", color=colors["Sema0p3"], capsize=4, elinewidth=2, capthick=2)
    ax_glucose.errorbar(np.array(data["AZstudy"]["timeGlucose"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema2Glucose"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema2Glucose_stdev"][0], dtype=float), fmt="none", color=colors["Sema2"], capsize=4, elinewidth=2, capthick=2)
    ax_glucose.errorbar(np.array(data["AZstudy"]["timeGlucose"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema4_Q2DGlucose"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema4_Q2DGlucose_stdev"][0], dtype=float), fmt="none", color=colors["Sema4_Q2D"], capsize=4, elinewidth=2, capthick=2)
    ax_glucose.errorbar(np.array(data["AZstudy"]["timeGlucose"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema10Glucose"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema10Glucose_stdev"][0], dtype=float), fmt="none", color=colors["Sema10"], capsize=4, elinewidth=2, capthick=2)
    for day in range(22):
        ax_glucose.plot([day/7 - 0.1/7, day/7 + 0.1/7], [0, 0], color='black', lw=18, solid_capstyle='butt')
    ax_glucose.set_xlabel("Time (weeks)", fontsize=18)
    ax_glucose.set_ylabel("Basal plasma glucose\n(mmol/L)", fontsize=18)
    ax_glucose.set_xlim(-0.1, 5.1)
    ax_glucose.set_xticks([0,1,2,3,4,5])
    ax_glucose.set_ylim(0, 20)
    ax_glucose.set_yticks([0,5,10,15,20])
    # ROW 2 COL 2: INSULIN
    ax_insulin.fill_between(timeDay/7, Vehicle_Insulin_min, Vehicle_Insulin_max, color=colors["Vehicle"], alpha=0.15, edgecolor="none")
    ax_insulin.plot(timeDay/7, best_Vehicle_Insulin, color=colors["Vehicle"], lw=2)
    ax_insulin.fill_between(timeDay/7, Sema0p3_Insulin_min, Sema0p3_Insulin_max, color=colors["Sema0p3"], alpha=0.15, edgecolor="none")
    ax_insulin.plot(timeDay/7, best_Sema0p3_Insulin, color=colors["Sema0p3"], lw=2)
    ax_insulin.fill_between(timeDay/7, Sema2_Insulin_min, Sema2_Insulin_max, color=colors["Sema2"], alpha=0.15, edgecolor="none")
    ax_insulin.plot(timeDay/7, best_Sema2_Insulin, color=colors["Sema2"], lw=2)
    ax_insulin.fill_between(timeDay/7, Sema4_Insulin_min, Sema4_Insulin_max, color=colors["Sema4_Q2D"], alpha=0.15, edgecolor="none")
    ax_insulin.plot(timeDay/7, best_Sema4_Insulin, color=colors["Sema4_Q2D"], lw=2)
    ax_insulin.fill_between(timeDay_Sema10/7, Sema10_Insulin_min, Sema10_Insulin_max, color=colors["Sema10"], alpha=0.15, edgecolor="none")
    ax_insulin.plot(timeDay_Sema10/7, best_Sema10_Insulin, color=colors["Sema10"], lw=2)
    ax_insulin.errorbar(np.array(data["AZstudy"]["timeInsulin"][0], dtype=float)/7, np.array(data["AZstudy"]["VehicleInsulin"][0], dtype=float), yerr=np.array(data["AZstudy"]["VehicleInsulin_stdev"][0], dtype=float), fmt="none", color=colors["Vehicle"], capsize=4, elinewidth=2, capthick=2)
    ax_insulin.errorbar(np.array(data["AZstudy"]["timeInsulin"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema0p3Insulin"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema0p3Insulin_stdev"][0], dtype=float), fmt="none", color=colors["Sema0p3"], capsize=4, elinewidth=2, capthick=2)
    ax_insulin.errorbar(np.array(data["AZstudy"]["timeInsulin"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema2Insulin"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema2Insulin_stdev"][0], dtype=float), fmt="none", color=colors["Sema2"], capsize=4, elinewidth=2, capthick=2)
    ax_insulin.errorbar(np.array(data["AZstudy"]["timeInsulin"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema4_Q2DInsulin"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema4_Q2DInsulin_stdev"][0], dtype=float), fmt="none", color=colors["Sema4_Q2D"], capsize=4, elinewidth=2, capthick=2)
    ax_insulin.errorbar(np.array(data["AZstudy"]["timeInsulin"][0], dtype=float)/7, np.array(data["AZstudy"]["Sema10Insulin"][0], dtype=float), yerr=np.array(data["AZstudy"]["Sema10Insulin_stdev"][0], dtype=float), fmt="none", color=colors["Sema10"], capsize=4, elinewidth=2, capthick=2)
    for day in range(22):
        ax_insulin.plot([day/7 - 0.1/7, day/7 + 0.1/7], [0, 0], color='black', lw=18, solid_capstyle='butt')
    ax_insulin.set_xlabel("Time (weeks)", fontsize=18)
    ax_insulin.set_ylabel("Basal plasma insulin\n(nmol/L)", fontsize=18)
    ax_insulin.set_xlim(-0.1, 5.1)
    ax_insulin.set_xticks([0,1,2,3,4,5])
    ax_insulin.set_ylim(0, 2)
    ax_insulin.set_yticks([0,0.5,1,1.5,2])
    for ax in [ax_bw, ax_fi, ax_glucose, ax_insulin]:
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True)
    for ax in [ax_bw, ax_fi, ax_glucose, ax_insulin]:
        for i in range(len(edges) - 1):
            if IsNight[edges[i]]:
                ax.axvspan(time_Sema10[edges[i]]/7, time_Sema10[edges[i + 1]]/7, color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    fig.subplots_adjust(left=0.07,right=0.993,top=0.98,bottom=0.065,wspace=0.3,hspace=0.3)
    plt.show()


# --- Predict populations study ---
def plot_DigitalPopulations(Model_Appetite,model_BW,Model_PK,model_GI,data,OptimisedAppetiteparams,OptimisedBWparams,OptimisedSemaglutidePKparams,OptimisedHeparams, FastPlot):


    OptimisedAppetiteparams = OptimisedAppetiteparams.to_numpy(dtype=float)
    OptimisedBWparams = OptimisedBWparams.to_numpy(dtype=float)
    OptimisedSemaglutidePKparams = OptimisedSemaglutidePKparams.to_numpy(dtype=float)
    OptimisedHeparams = OptimisedHeparams.to_numpy(dtype=float)
    Best_BWParams = np.asarray(OptimisedBWparams[0][:-1], dtype=float)
    Best_PKParams = np.asarray(OptimisedSemaglutidePKparams[0][:-1], dtype=float)
    Best_Heparams = np.asarray(OptimisedHeparams[0][:-1], dtype=float)

    n_select = 200
    if FastPlot:
        n_select = 10
    idx = np.linspace(0, OptimisedAppetiteparams.shape[0]-1, n_select, dtype=int)
    OptimisedAppetiteparams = OptimisedAppetiteparams[idx]
    OptimisedBWparams = OptimisedBWparams[idx]
    OptimisedHeparams = OptimisedHeparams[idx]

    def daily_average(values, TimeRes_exposure):
        values = np.asarray(values, dtype=float)
        points_per_day = int(24 * TimeRes_exposure)
        n_days = (len(values)-1) // points_per_day
        return values[:n_days*points_per_day].reshape(n_days, points_per_day).mean(axis=1)
    def last_day_meals_per_day_and_meal_size(ei_ac, eating_array, time, TimeRes):
        ei_ac = np.asarray(ei_ac, dtype=float)
        eating_array = np.asarray(eating_array, dtype=float)
        time = np.asarray(time, dtype=float)
        daily_ei = daily_energy_intake_from_ei_ac(ei_ac, time)
        meal_starts = (eating_array[1:] > 0.5) & (eating_array[:-1] <= 0.5)
        meal_start_idx = np.where(meal_starts)[0] + 1
        if eating_array[0] > 0.5:
            meal_start_idx = np.insert(meal_start_idx, 0, 0)
        meal_start_days = np.floor(time[meal_start_idx])
        last_day = int(np.floor(np.nanmax(time)) - 1)
        last_day_mask = meal_start_days == last_day
        total_meals = np.sum(last_day_mask)
        meal_start_hours = (time[meal_start_idx] % 1.0) * 24
        night_meals = np.sum(
            ((meal_start_hours < 9.0) | (meal_start_hours >= 21.0))
            & last_day_mask
        )
        daily_ei_last_day = daily_ei[-1]
        meals_per_day = float(total_meals)
        mean_meal_size = daily_ei_last_day / total_meals if total_meals > 0 else 0.0
        night_meal_percent = 100 * night_meals / total_meals if total_meals > 0 else 0.0
        return meals_per_day, mean_meal_size, daily_ei, night_meal_percent

    TimeRes_exposure = 12

    x_fedstate  = np.linspace(0, 30, 30 + 1)

    first = True
    for VectorNumber in range(len(OptimisedAppetiteparams)):
        AppetiteParams = np.asarray(OptimisedAppetiteparams[VectorNumber][:-1], dtype=float)

        try:
            sims = Simulate_DigitalPopulations(Model_Appetite, model_BW, Model_PK, model_GI, AppetiteParams, Best_BWParams, Best_PKParams, Best_Heparams, TimeRes_exposure, 2.8)
        except Exception as e:
            print("Digital population simulation failed:", repr(e))
            continue

        idx_ei_ac = sims["HealthyChow_D12450J"].feature_names.index("EI_AC")
        idx_bw = sims["HealthyChow_D12450J"].feature_names.index("Body_weight")
        idx_glucose = sims["HealthyChow_D12450J"].feature_names.index("Plasma_Glucose")
        idx_eating = sims["HealthyChow_D12450J"].feature_names.index("Eating")
        idx_daynight = sims["HealthyChow_D12450J"].feature_names.index("DayNightCycle")
        HealthyChow_D12450J_EIappetite_new =  np.exp(-sims["HealthyChow_D12450J"].parameter_values[sims["HealthyChow_D12450J"].parameter_names.index("EI_HungerRate")] * (x_fedstate-15))
        NonLipid_satiety_new = sims["HealthyChow_D12450J"].parameter_values[sims["HealthyChow_D12450J"].parameter_names.index("StomachFullnessCarbs")]
        Lipid_satiety_new = sims["HealthyChow_D12450J"].parameter_values[sims["HealthyChow_D12450J"].parameter_names.index("StomachFullnessFats")]
        NonLipidandLipid_ratio_new = sims["HealthyChow_D12450J"].parameter_values[sims["HealthyChow_D12450J"].parameter_names.index("StomachFullnessCarbs")]/sims["HealthyChow_D12450J"].parameter_values[sims["HealthyChow_D12450J"].parameter_names.index("StomachFullnessFats")]
        HealthyChow_D12450J_DN_new = sims["HealthyChow_D12450J"].feature_values[:, idx_daynight]
        # ---------- Total body mass ----------
        HealthyChow_D12450J_BW_new = sims["HealthyChow_D12450J"].feature_values[:,idx_bw]
        HealthyHFD_D12492_BW_new = sims["HealthyHFD_D12492"].feature_values[:,idx_bw]
        HealthyHFD_D12331_BW_new = sims["HealthyHFD_D12331"].feature_values[:,idx_bw]
        HealthyControl_D124512_BW_new = sims["HealthyControl_D124512"].feature_values[:,idx_bw]
        HealthyControl_TekladLM_485_BW_new = sims["HealthyControl_TekladLM_485"].feature_values[:,idx_bw]
        ObeseChow_D12450J_BW_new = sims["ObeseChow_D12450J"].feature_values[:,idx_bw]
        ObeseHFD_D12492_BW_new = sims["ObeseHFD_D12492"].feature_values[:,idx_bw]
        ObeseHFD_Sema10_BW_new = sims["ObeseHFD_Sema10"].feature_values[:,idx_bw]
        ObeseHFD_Sema2_BW_new = sims["ObeseHFD_Sema2"].feature_values[:,idx_bw]
        # ---------- Daily average glucose ----------
        HealthyChow_D12450J_GlucoseDaily_new = daily_average(sims["HealthyChow_D12450J"].feature_values[:,idx_glucose],TimeRes_exposure)
        HealthyHFD_D12492_GlucoseDaily_new = daily_average(sims["HealthyHFD_D12492"].feature_values[:,idx_glucose],TimeRes_exposure)
        HealthyHFD_D12331_GlucoseDaily_new = daily_average(sims["HealthyHFD_D12331"].feature_values[:,idx_glucose],TimeRes_exposure)
        HealthyControl_D124512_GlucoseDaily_new = daily_average(sims["HealthyControl_D124512"].feature_values[:,idx_glucose],TimeRes_exposure)
        HealthyControl_TekladLM_485_GlucoseDaily_new = daily_average(sims["HealthyControl_TekladLM_485"].feature_values[:,idx_glucose],TimeRes_exposure)
        ObeseChow_D12450J_GlucoseDaily_new = daily_average(sims["ObeseChow_D12450J"].feature_values[:,idx_glucose],TimeRes_exposure)
        ObeseHFD_D12492_GlucoseDaily_new = daily_average(sims["ObeseHFD_D12492"].feature_values[:,idx_glucose],TimeRes_exposure)
        ObeseHFD_Sema10_GlucoseDaily_new = daily_average(sims["ObeseHFD_Sema10"].feature_values[:,idx_glucose],TimeRes_exposure)
        ObeseHFD_Sema2_GlucoseDaily_new = daily_average(sims["ObeseHFD_Sema2"].feature_values[:,idx_glucose],TimeRes_exposure)
        # ---------- Full-resolution glucose ----------
        HealthyChow_D12450J_Glucose_new = sims["HealthyChow_D12450J"].feature_values[:,idx_glucose]
        HealthyHFD_D12492_Glucose_new = sims["HealthyHFD_D12492"].feature_values[:,idx_glucose]
        HealthyHFD_D12331_Glucose_new = sims["HealthyHFD_D12331"].feature_values[:,idx_glucose]
        HealthyControl_D124512_Glucose_new = sims["HealthyControl_D124512"].feature_values[:,idx_glucose]
        HealthyControl_TekladLM_485_Glucose_new = sims["HealthyControl_TekladLM_485"].feature_values[:,idx_glucose]
        ObeseChow_D12450J_Glucose_new = sims["ObeseChow_D12450J"].feature_values[:,idx_glucose]
        ObeseHFD_D12492_Glucose_new = sims["ObeseHFD_D12492"].feature_values[:,idx_glucose]
        ObeseHFD_Sema10_Glucose_new = sims["ObeseHFD_Sema10"].feature_values[:,idx_glucose]
        ObeseHFD_Sema2_Glucose_new = sims["ObeseHFD_Sema2"].feature_values[:,idx_glucose]
        # ---------- Meals, meal size, daily energy intake and night meals ----------
        HealthyChow_D12450J_meals_new,HealthyChow_D12450J_meal_size_new,HealthyChow_D12450J_daily_ei_new,HealthyChow_D12450J_night_meals_new = last_day_meals_per_day_and_meal_size(sims["HealthyChow_D12450J"].feature_values[:,idx_ei_ac],sims["HealthyChow_D12450J"].feature_values[:,idx_eating],sims["HealthyChow_D12450J"].time_vector,TimeRes_exposure)
        HealthyHFD_D12492_meals_new,HealthyHFD_D12492_meal_size_new,HealthyHFD_D12492_daily_ei_new,HealthyHFD_D12492_night_meals_new = last_day_meals_per_day_and_meal_size(sims["HealthyHFD_D12492"].feature_values[:,idx_ei_ac],sims["HealthyHFD_D12492"].feature_values[:,idx_eating],sims["HealthyHFD_D12492"].time_vector,TimeRes_exposure)
        HealthyHFD_D12331_meals_new,HealthyHFD_D12331_meal_size_new,HealthyHFD_D12331_daily_ei_new,HealthyHFD_D12331_night_meals_new = last_day_meals_per_day_and_meal_size(sims["HealthyHFD_D12331"].feature_values[:,idx_ei_ac],sims["HealthyHFD_D12331"].feature_values[:,idx_eating],sims["HealthyHFD_D12331"].time_vector,TimeRes_exposure)
        HealthyControl_D124512_meals_new,HealthyControl_D124512_meal_size_new,HealthyControl_D124512_daily_ei_new,HealthyControl_D124512_night_meals_new = last_day_meals_per_day_and_meal_size(sims["HealthyControl_D124512"].feature_values[:,idx_ei_ac],sims["HealthyControl_D124512"].feature_values[:,idx_eating],sims["HealthyControl_D124512"].time_vector,TimeRes_exposure)
        HealthyControl_TekladLM_485_meals_new,HealthyControl_TekladLM_485_meal_size_new,HealthyControl_TekladLM_485_daily_ei_new,HealthyControl_TekladLM_485_night_meals_new = last_day_meals_per_day_and_meal_size(sims["HealthyControl_TekladLM_485"].feature_values[:,idx_ei_ac],sims["HealthyControl_TekladLM_485"].feature_values[:,idx_eating],sims["HealthyControl_TekladLM_485"].time_vector,TimeRes_exposure)
        ObeseChow_D12450J_meals_new,ObeseChow_D12450J_meal_size_new,ObeseChow_D12450J_daily_ei_new,ObeseChow_D12450J_night_meals_new = last_day_meals_per_day_and_meal_size(sims["ObeseChow_D12450J"].feature_values[:,idx_ei_ac],sims["ObeseChow_D12450J"].feature_values[:,idx_eating],sims["ObeseChow_D12450J"].time_vector,TimeRes_exposure)
        ObeseHFD_D12492_meals_new,ObeseHFD_D12492_meal_size_new,ObeseHFD_D12492_daily_ei_new,ObeseHFD_D12492_night_meals_new = last_day_meals_per_day_and_meal_size(sims["ObeseHFD_D12492"].feature_values[:,idx_ei_ac],sims["ObeseHFD_D12492"].feature_values[:,idx_eating],sims["ObeseHFD_D12492"].time_vector,TimeRes_exposure)
        ObeseHFD_Sema10_meals_new,ObeseHFD_Sema10_meal_size_new,ObeseHFD_Sema10_daily_ei_new,ObeseHFD_Sema10_night_meals_new = last_day_meals_per_day_and_meal_size(sims["ObeseHFD_Sema10"].feature_values[:,idx_ei_ac],sims["ObeseHFD_Sema10"].feature_values[:,idx_eating],sims["ObeseHFD_Sema10"].time_vector,TimeRes_exposure)
        ObeseHFD_Sema2_meals_new,ObeseHFD_Sema2_meal_size_new,ObeseHFD_Sema2_daily_ei_new,ObeseHFD_Sema2_night_meals_new = last_day_meals_per_day_and_meal_size(sims["ObeseHFD_Sema2"].feature_values[:,idx_ei_ac],sims["ObeseHFD_Sema2"].feature_values[:,idx_eating],sims["ObeseHFD_Sema2"].time_vector,TimeRes_exposure)
        # ---------- Convert meal outputs to arrays ----------
        HealthyChow_D12450J_meals_new = np.asarray(HealthyChow_D12450J_meals_new,dtype=float)
        HealthyHFD_D12492_meals_new = np.asarray(HealthyHFD_D12492_meals_new,dtype=float)
        HealthyHFD_D12331_meals_new = np.asarray(HealthyHFD_D12331_meals_new,dtype=float)
        HealthyControl_D124512_meals_new = np.asarray(HealthyControl_D124512_meals_new,dtype=float)
        HealthyControl_TekladLM_485_meals_new = np.asarray(HealthyControl_TekladLM_485_meals_new,dtype=float)
        ObeseChow_D12450J_meals_new = np.asarray(ObeseChow_D12450J_meals_new,dtype=float)
        ObeseHFD_D12492_meals_new = np.asarray(ObeseHFD_D12492_meals_new,dtype=float)
        ObeseHFD_Sema10_meals_new = np.asarray(ObeseHFD_Sema10_meals_new,dtype=float)
        ObeseHFD_Sema2_meals_new = np.asarray(ObeseHFD_Sema2_meals_new,dtype=float)
        HealthyChow_D12450J_meal_size_new = np.asarray(HealthyChow_D12450J_meal_size_new,dtype=float)
        HealthyHFD_D12492_meal_size_new = np.asarray(HealthyHFD_D12492_meal_size_new,dtype=float)
        HealthyHFD_D12331_meal_size_new = np.asarray(HealthyHFD_D12331_meal_size_new,dtype=float)
        HealthyControl_D124512_meal_size_new = np.asarray(HealthyControl_D124512_meal_size_new,dtype=float)
        HealthyControl_TekladLM_485_meal_size_new = np.asarray(HealthyControl_TekladLM_485_meal_size_new,dtype=float)
        ObeseChow_D12450J_meal_size_new = np.asarray(ObeseChow_D12450J_meal_size_new,dtype=float)
        ObeseHFD_D12492_meal_size_new = np.asarray(ObeseHFD_D12492_meal_size_new,dtype=float)
        ObeseHFD_Sema10_meal_size_new = np.asarray(ObeseHFD_Sema10_meal_size_new,dtype=float)
        ObeseHFD_Sema2_meal_size_new = np.asarray(ObeseHFD_Sema2_meal_size_new,dtype=float)
        HealthyChow_D12450J_daily_ei_new = np.asarray(HealthyChow_D12450J_daily_ei_new,dtype=float)
        HealthyHFD_D12492_daily_ei_new = np.asarray(HealthyHFD_D12492_daily_ei_new,dtype=float)
        HealthyHFD_D12331_daily_ei_new = np.asarray(HealthyHFD_D12331_daily_ei_new,dtype=float)
        HealthyControl_D124512_daily_ei_new = np.asarray(HealthyControl_D124512_daily_ei_new,dtype=float)
        HealthyControl_TekladLM_485_daily_ei_new = np.asarray(HealthyControl_TekladLM_485_daily_ei_new,dtype=float)
        ObeseChow_D12450J_daily_ei_new = np.asarray(ObeseChow_D12450J_daily_ei_new,dtype=float)
        ObeseHFD_D12492_daily_ei_new = np.asarray(ObeseHFD_D12492_daily_ei_new,dtype=float)
        ObeseHFD_Sema10_daily_ei_new = np.asarray(ObeseHFD_Sema10_daily_ei_new,dtype=float)
        ObeseHFD_Sema2_daily_ei_new = np.asarray(ObeseHFD_Sema2_daily_ei_new,dtype=float)
        HealthyChow_D12450J_night_meals_new = np.asarray(HealthyChow_D12450J_night_meals_new,dtype=float)
        HealthyHFD_D12492_night_meals_new = np.asarray(HealthyHFD_D12492_night_meals_new,dtype=float)
        HealthyHFD_D12331_night_meals_new = np.asarray(HealthyHFD_D12331_night_meals_new,dtype=float)
        HealthyControl_D124512_night_meals_new = np.asarray(HealthyControl_D124512_night_meals_new,dtype=float)
        HealthyControl_TekladLM_485_night_meals_new = np.asarray(HealthyControl_TekladLM_485_night_meals_new,dtype=float)
        ObeseChow_D12450J_night_meals_new = np.asarray(ObeseChow_D12450J_night_meals_new,dtype=float)
        ObeseHFD_D12492_night_meals_new = np.asarray(ObeseHFD_D12492_night_meals_new,dtype=float)
        ObeseHFD_Sema10_night_meals_new = np.asarray(ObeseHFD_Sema10_night_meals_new,dtype=float)
        ObeseHFD_Sema2_night_meals_new = np.asarray(ObeseHFD_Sema2_night_meals_new,dtype=float)

        if first:
            best_sim = sims
            time_regular = sims["HealthyChow_D12450J"].time_vector
            time_Glucose = np.arange(len(HealthyChow_D12450J_GlucoseDaily_new), dtype=float)

            # ---------- Appetite components ----------
            best_HealthyChow_D12450J_DN,HealthyChow_D12450J_DN_min,HealthyChow_D12450J_DN_max = HealthyChow_D12450J_DN_new.copy(),HealthyChow_D12450J_DN_new.copy(),HealthyChow_D12450J_DN_new.copy()
            best_HealthyChow_D12450J_NonLipid_satiety,HealthyChow_D12450J_NonLipid_satiety_min,HealthyChow_D12450J_NonLipid_satiety_max = NonLipid_satiety_new.copy()      ,NonLipid_satiety_new.copy()      ,NonLipid_satiety_new.copy()
            best_HealthyChow_D12450J_Lipid_satiety,HealthyChow_D12450J_Lipid_satiety_min,HealthyChow_D12450J_Lipid_satiety_max = Lipid_satiety_new.copy()         ,Lipid_satiety_new.copy()         ,Lipid_satiety_new.copy()
            best_HealthyChow_D12450J_NonLipidandLipid_ratio,HealthyChow_D12450J_NonLipidandLipid_ratio_min,HealthyChow_D12450J_NonLipidandLipid_ratio_max = NonLipidandLipid_ratio_new.copy(),NonLipidandLipid_ratio_new.copy(),NonLipidandLipid_ratio_new.copy()
            best_HealthyChow_D12450J_EIappetite,HealthyChow_D12450J_EIappetite_min,HealthyChow_D12450J_EIappetite_max = HealthyChow_D12450J_EIappetite_new.copy(),HealthyChow_D12450J_EIappetite_new.copy(),HealthyChow_D12450J_EIappetite_new.copy()
            # ---------- Total body mass ----------
            best_HealthyChow_D12450J_BW,HealthyChow_D12450J_BW_min,HealthyChow_D12450J_BW_max = HealthyChow_D12450J_BW_new.copy(),HealthyChow_D12450J_BW_new.copy(),HealthyChow_D12450J_BW_new.copy()
            best_HealthyHFD_D12492_BW,HealthyHFD_D12492_BW_min,HealthyHFD_D12492_BW_max = HealthyHFD_D12492_BW_new.copy(),HealthyHFD_D12492_BW_new.copy(),HealthyHFD_D12492_BW_new.copy()
            best_HealthyHFD_D12331_BW,HealthyHFD_D12331_BW_min,HealthyHFD_D12331_BW_max = HealthyHFD_D12331_BW_new.copy(),HealthyHFD_D12331_BW_new.copy(),HealthyHFD_D12331_BW_new.copy()
            best_HealthyControl_D124512_BW,HealthyControl_D124512_BW_min,HealthyControl_D124512_BW_max = HealthyControl_D124512_BW_new.copy(),HealthyControl_D124512_BW_new.copy(),HealthyControl_D124512_BW_new.copy()
            best_HealthyControl_TekladLM_485_BW,HealthyControl_TekladLM_485_BW_min,HealthyControl_TekladLM_485_BW_max = HealthyControl_TekladLM_485_BW_new.copy(),HealthyControl_TekladLM_485_BW_new.copy(),HealthyControl_TekladLM_485_BW_new.copy()
            best_ObeseChow_D12450J_BW,ObeseChow_D12450J_BW_min,ObeseChow_D12450J_BW_max = ObeseChow_D12450J_BW_new.copy(),ObeseChow_D12450J_BW_new.copy(),ObeseChow_D12450J_BW_new.copy()
            best_ObeseHFD_D12492_BW,ObeseHFD_D12492_BW_min,ObeseHFD_D12492_BW_max = ObeseHFD_D12492_BW_new.copy(),ObeseHFD_D12492_BW_new.copy(),ObeseHFD_D12492_BW_new.copy()
            best_ObeseHFD_Sema10_BW,ObeseHFD_Sema10_BW_min,ObeseHFD_Sema10_BW_max = ObeseHFD_Sema10_BW_new.copy(),ObeseHFD_Sema10_BW_new.copy(),ObeseHFD_Sema10_BW_new.copy()
            best_ObeseHFD_Sema2_BW,ObeseHFD_Sema2_BW_min,ObeseHFD_Sema2_BW_max = ObeseHFD_Sema2_BW_new.copy(),ObeseHFD_Sema2_BW_new.copy(),ObeseHFD_Sema2_BW_new.copy()
            # ---------- Daily average glucose ----------
            best_HealthyChow_D12450J_GlucoseDaily,HealthyChow_D12450J_GlucoseDaily_min,HealthyChow_D12450J_GlucoseDaily_max = HealthyChow_D12450J_GlucoseDaily_new.copy(),HealthyChow_D12450J_GlucoseDaily_new.copy(),HealthyChow_D12450J_GlucoseDaily_new.copy()
            best_HealthyHFD_D12492_GlucoseDaily,HealthyHFD_D12492_GlucoseDaily_min,HealthyHFD_D12492_GlucoseDaily_max = HealthyHFD_D12492_GlucoseDaily_new.copy(),HealthyHFD_D12492_GlucoseDaily_new.copy(),HealthyHFD_D12492_GlucoseDaily_new.copy()
            best_HealthyHFD_D12331_GlucoseDaily,HealthyHFD_D12331_GlucoseDaily_min,HealthyHFD_D12331_GlucoseDaily_max = HealthyHFD_D12331_GlucoseDaily_new.copy(),HealthyHFD_D12331_GlucoseDaily_new.copy(),HealthyHFD_D12331_GlucoseDaily_new.copy()
            best_HealthyControl_D124512_GlucoseDaily,HealthyControl_D124512_GlucoseDaily_min,HealthyControl_D124512_GlucoseDaily_max = HealthyControl_D124512_GlucoseDaily_new.copy(),HealthyControl_D124512_GlucoseDaily_new.copy(),HealthyControl_D124512_GlucoseDaily_new.copy()
            best_HealthyControl_TekladLM_485_GlucoseDaily,HealthyControl_TekladLM_485_GlucoseDaily_min,HealthyControl_TekladLM_485_GlucoseDaily_max = HealthyControl_TekladLM_485_GlucoseDaily_new.copy(),HealthyControl_TekladLM_485_GlucoseDaily_new.copy(),HealthyControl_TekladLM_485_GlucoseDaily_new.copy()
            best_ObeseChow_D12450J_GlucoseDaily,ObeseChow_D12450J_GlucoseDaily_min,ObeseChow_D12450J_GlucoseDaily_max = ObeseChow_D12450J_GlucoseDaily_new.copy(),ObeseChow_D12450J_GlucoseDaily_new.copy(),ObeseChow_D12450J_GlucoseDaily_new.copy()
            best_ObeseHFD_D12492_GlucoseDaily,ObeseHFD_D12492_GlucoseDaily_min,ObeseHFD_D12492_GlucoseDaily_max = ObeseHFD_D12492_GlucoseDaily_new.copy(),ObeseHFD_D12492_GlucoseDaily_new.copy(),ObeseHFD_D12492_GlucoseDaily_new.copy()
            best_ObeseHFD_Sema10_GlucoseDaily,ObeseHFD_Sema10_GlucoseDaily_min,ObeseHFD_Sema10_GlucoseDaily_max = ObeseHFD_Sema10_GlucoseDaily_new.copy(),ObeseHFD_Sema10_GlucoseDaily_new.copy(),ObeseHFD_Sema10_GlucoseDaily_new.copy()
            best_ObeseHFD_Sema2_GlucoseDaily,ObeseHFD_Sema2_GlucoseDaily_min,ObeseHFD_Sema2_GlucoseDaily_max = ObeseHFD_Sema2_GlucoseDaily_new.copy(),ObeseHFD_Sema2_GlucoseDaily_new.copy(),ObeseHFD_Sema2_GlucoseDaily_new.copy()
            # ---------- Full-resolution glucose ----------
            best_HealthyChow_D12450J_Glucose,HealthyChow_D12450J_Glucose_min,HealthyChow_D12450J_Glucose_max = HealthyChow_D12450J_Glucose_new.copy(),HealthyChow_D12450J_Glucose_new.copy(),HealthyChow_D12450J_Glucose_new.copy()
            best_HealthyHFD_D12492_Glucose,HealthyHFD_D12492_Glucose_min,HealthyHFD_D12492_Glucose_max = HealthyHFD_D12492_Glucose_new.copy(),HealthyHFD_D12492_Glucose_new.copy(),HealthyHFD_D12492_Glucose_new.copy()
            best_HealthyHFD_D12331_Glucose,HealthyHFD_D12331_Glucose_min,HealthyHFD_D12331_Glucose_max = HealthyHFD_D12331_Glucose_new.copy(),HealthyHFD_D12331_Glucose_new.copy(),HealthyHFD_D12331_Glucose_new.copy()
            best_HealthyControl_D124512_Glucose,HealthyControl_D124512_Glucose_min,HealthyControl_D124512_Glucose_max = HealthyControl_D124512_Glucose_new.copy(),HealthyControl_D124512_Glucose_new.copy(),HealthyControl_D124512_Glucose_new.copy()
            best_HealthyControl_TekladLM_485_Glucose,HealthyControl_TekladLM_485_Glucose_min,HealthyControl_TekladLM_485_Glucose_max = HealthyControl_TekladLM_485_Glucose_new.copy(),HealthyControl_TekladLM_485_Glucose_new.copy(),HealthyControl_TekladLM_485_Glucose_new.copy()
            best_ObeseChow_D12450J_Glucose,ObeseChow_D12450J_Glucose_min,ObeseChow_D12450J_Glucose_max = ObeseChow_D12450J_Glucose_new.copy(),ObeseChow_D12450J_Glucose_new.copy(),ObeseChow_D12450J_Glucose_new.copy()
            best_ObeseHFD_D12492_Glucose,ObeseHFD_D12492_Glucose_min,ObeseHFD_D12492_Glucose_max = ObeseHFD_D12492_Glucose_new.copy(),ObeseHFD_D12492_Glucose_new.copy(),ObeseHFD_D12492_Glucose_new.copy()
            best_ObeseHFD_Sema10_Glucose,ObeseHFD_Sema10_Glucose_min,ObeseHFD_Sema10_Glucose_max = ObeseHFD_Sema10_Glucose_new.copy(),ObeseHFD_Sema10_Glucose_new.copy(),ObeseHFD_Sema10_Glucose_new.copy()
            best_ObeseHFD_Sema2_Glucose,ObeseHFD_Sema2_Glucose_min,ObeseHFD_Sema2_Glucose_max = ObeseHFD_Sema2_Glucose_new.copy(),ObeseHFD_Sema2_Glucose_new.copy(),ObeseHFD_Sema2_Glucose_new.copy()
            # ---------- Meals per day ----------
            best_HealthyChow_D12450J_meals,HealthyChow_D12450J_meals_min,HealthyChow_D12450J_meals_max = HealthyChow_D12450J_meals_new.copy(),HealthyChow_D12450J_meals_new.copy(),HealthyChow_D12450J_meals_new.copy()
            best_HealthyHFD_D12492_meals,HealthyHFD_D12492_meals_min,HealthyHFD_D12492_meals_max = HealthyHFD_D12492_meals_new.copy(),HealthyHFD_D12492_meals_new.copy(),HealthyHFD_D12492_meals_new.copy()
            best_HealthyHFD_D12331_meals,HealthyHFD_D12331_meals_min,HealthyHFD_D12331_meals_max = HealthyHFD_D12331_meals_new.copy(),HealthyHFD_D12331_meals_new.copy(),HealthyHFD_D12331_meals_new.copy()
            best_HealthyControl_D124512_meals,HealthyControl_D124512_meals_min,HealthyControl_D124512_meals_max = HealthyControl_D124512_meals_new.copy(),HealthyControl_D124512_meals_new.copy(),HealthyControl_D124512_meals_new.copy()
            best_HealthyControl_TekladLM_485_meals,HealthyControl_TekladLM_485_meals_min,HealthyControl_TekladLM_485_meals_max = HealthyControl_TekladLM_485_meals_new.copy(),HealthyControl_TekladLM_485_meals_new.copy(),HealthyControl_TekladLM_485_meals_new.copy()
            best_ObeseChow_D12450J_meals,ObeseChow_D12450J_meals_min,ObeseChow_D12450J_meals_max = ObeseChow_D12450J_meals_new.copy(),ObeseChow_D12450J_meals_new.copy(),ObeseChow_D12450J_meals_new.copy()
            best_ObeseHFD_D12492_meals,ObeseHFD_D12492_meals_min,ObeseHFD_D12492_meals_max = ObeseHFD_D12492_meals_new.copy(),ObeseHFD_D12492_meals_new.copy(),ObeseHFD_D12492_meals_new.copy()
            best_ObeseHFD_Sema10_meals,ObeseHFD_Sema10_meals_min,ObeseHFD_Sema10_meals_max = ObeseHFD_Sema10_meals_new.copy(),ObeseHFD_Sema10_meals_new.copy(),ObeseHFD_Sema10_meals_new.copy()
            best_ObeseHFD_Sema2_meals,ObeseHFD_Sema2_meals_min,ObeseHFD_Sema2_meals_max = ObeseHFD_Sema2_meals_new.copy(),ObeseHFD_Sema2_meals_new.copy(),ObeseHFD_Sema2_meals_new.copy()
            # ---------- Meal size ----------
            best_HealthyChow_D12450J_meal_size,HealthyChow_D12450J_meal_size_min,HealthyChow_D12450J_meal_size_max = HealthyChow_D12450J_meal_size_new.copy(),HealthyChow_D12450J_meal_size_new.copy(),HealthyChow_D12450J_meal_size_new.copy()
            best_HealthyHFD_D12492_meal_size,HealthyHFD_D12492_meal_size_min,HealthyHFD_D12492_meal_size_max = HealthyHFD_D12492_meal_size_new.copy(),HealthyHFD_D12492_meal_size_new.copy(),HealthyHFD_D12492_meal_size_new.copy()
            best_HealthyHFD_D12331_meal_size,HealthyHFD_D12331_meal_size_min,HealthyHFD_D12331_meal_size_max = HealthyHFD_D12331_meal_size_new.copy(),HealthyHFD_D12331_meal_size_new.copy(),HealthyHFD_D12331_meal_size_new.copy()
            best_HealthyControl_D124512_meal_size,HealthyControl_D124512_meal_size_min,HealthyControl_D124512_meal_size_max = HealthyControl_D124512_meal_size_new.copy(),HealthyControl_D124512_meal_size_new.copy(),HealthyControl_D124512_meal_size_new.copy()
            best_HealthyControl_TekladLM_485_meal_size,HealthyControl_TekladLM_485_meal_size_min,HealthyControl_TekladLM_485_meal_size_max = HealthyControl_TekladLM_485_meal_size_new.copy(),HealthyControl_TekladLM_485_meal_size_new.copy(),HealthyControl_TekladLM_485_meal_size_new.copy()
            best_ObeseChow_D12450J_meal_size,ObeseChow_D12450J_meal_size_min,ObeseChow_D12450J_meal_size_max = ObeseChow_D12450J_meal_size_new.copy(),ObeseChow_D12450J_meal_size_new.copy(),ObeseChow_D12450J_meal_size_new.copy()
            best_ObeseHFD_D12492_meal_size,ObeseHFD_D12492_meal_size_min,ObeseHFD_D12492_meal_size_max = ObeseHFD_D12492_meal_size_new.copy(),ObeseHFD_D12492_meal_size_new.copy(),ObeseHFD_D12492_meal_size_new.copy()
            best_ObeseHFD_Sema10_meal_size,ObeseHFD_Sema10_meal_size_min,ObeseHFD_Sema10_meal_size_max = ObeseHFD_Sema10_meal_size_new.copy(),ObeseHFD_Sema10_meal_size_new.copy(),ObeseHFD_Sema10_meal_size_new.copy()
            best_ObeseHFD_Sema2_meal_size,ObeseHFD_Sema2_meal_size_min,ObeseHFD_Sema2_meal_size_max = ObeseHFD_Sema2_meal_size_new.copy(),ObeseHFD_Sema2_meal_size_new.copy(),ObeseHFD_Sema2_meal_size_new.copy()
            # ---------- Daily energy intake ----------
            best_HealthyChow_D12450J_daily_ei,HealthyChow_D12450J_daily_ei_min,HealthyChow_D12450J_daily_ei_max = HealthyChow_D12450J_daily_ei_new.copy(),HealthyChow_D12450J_daily_ei_new.copy(),HealthyChow_D12450J_daily_ei_new.copy()
            best_HealthyHFD_D12492_daily_ei,HealthyHFD_D12492_daily_ei_min,HealthyHFD_D12492_daily_ei_max = HealthyHFD_D12492_daily_ei_new.copy(),HealthyHFD_D12492_daily_ei_new.copy(),HealthyHFD_D12492_daily_ei_new.copy()
            best_HealthyHFD_D12331_daily_ei,HealthyHFD_D12331_daily_ei_min,HealthyHFD_D12331_daily_ei_max = HealthyHFD_D12331_daily_ei_new.copy(),HealthyHFD_D12331_daily_ei_new.copy(),HealthyHFD_D12331_daily_ei_new.copy()
            best_HealthyControl_D124512_daily_ei,HealthyControl_D124512_daily_ei_min,HealthyControl_D124512_daily_ei_max = HealthyControl_D124512_daily_ei_new.copy(),HealthyControl_D124512_daily_ei_new.copy(),HealthyControl_D124512_daily_ei_new.copy()
            best_HealthyControl_TekladLM_485_daily_ei,HealthyControl_TekladLM_485_daily_ei_min,HealthyControl_TekladLM_485_daily_ei_max = HealthyControl_TekladLM_485_daily_ei_new.copy(),HealthyControl_TekladLM_485_daily_ei_new.copy(),HealthyControl_TekladLM_485_daily_ei_new.copy()
            best_ObeseChow_D12450J_daily_ei,ObeseChow_D12450J_daily_ei_min,ObeseChow_D12450J_daily_ei_max = ObeseChow_D12450J_daily_ei_new.copy(),ObeseChow_D12450J_daily_ei_new.copy(),ObeseChow_D12450J_daily_ei_new.copy()
            best_ObeseHFD_D12492_daily_ei,ObeseHFD_D12492_daily_ei_min,ObeseHFD_D12492_daily_ei_max = ObeseHFD_D12492_daily_ei_new.copy(),ObeseHFD_D12492_daily_ei_new.copy(),ObeseHFD_D12492_daily_ei_new.copy()
            best_ObeseHFD_Sema10_daily_ei,ObeseHFD_Sema10_daily_ei_min,ObeseHFD_Sema10_daily_ei_max = ObeseHFD_Sema10_daily_ei_new.copy(),ObeseHFD_Sema10_daily_ei_new.copy(),ObeseHFD_Sema10_daily_ei_new.copy()
            best_ObeseHFD_Sema2_daily_ei,ObeseHFD_Sema2_daily_ei_min,ObeseHFD_Sema2_daily_ei_max = ObeseHFD_Sema2_daily_ei_new.copy(),ObeseHFD_Sema2_daily_ei_new.copy(),ObeseHFD_Sema2_daily_ei_new.copy()
            # ---------- Night meals ----------
            best_HealthyChow_D12450J_night_meals,HealthyChow_D12450J_night_meals_min,HealthyChow_D12450J_night_meals_max = HealthyChow_D12450J_night_meals_new.copy(),HealthyChow_D12450J_night_meals_new.copy(),HealthyChow_D12450J_night_meals_new.copy()
            best_HealthyHFD_D12492_night_meals,HealthyHFD_D12492_night_meals_min,HealthyHFD_D12492_night_meals_max = HealthyHFD_D12492_night_meals_new.copy(),HealthyHFD_D12492_night_meals_new.copy(),HealthyHFD_D12492_night_meals_new.copy()
            best_HealthyHFD_D12331_night_meals,HealthyHFD_D12331_night_meals_min,HealthyHFD_D12331_night_meals_max = HealthyHFD_D12331_night_meals_new.copy(),HealthyHFD_D12331_night_meals_new.copy(),HealthyHFD_D12331_night_meals_new.copy()
            best_HealthyControl_D124512_night_meals,HealthyControl_D124512_night_meals_min,HealthyControl_D124512_night_meals_max = HealthyControl_D124512_night_meals_new.copy(),HealthyControl_D124512_night_meals_new.copy(),HealthyControl_D124512_night_meals_new.copy()
            best_HealthyControl_TekladLM_485_night_meals,HealthyControl_TekladLM_485_night_meals_min,HealthyControl_TekladLM_485_night_meals_max = HealthyControl_TekladLM_485_night_meals_new.copy(),HealthyControl_TekladLM_485_night_meals_new.copy(),HealthyControl_TekladLM_485_night_meals_new.copy()
            best_ObeseChow_D12450J_night_meals,ObeseChow_D12450J_night_meals_min,ObeseChow_D12450J_night_meals_max = ObeseChow_D12450J_night_meals_new.copy(),ObeseChow_D12450J_night_meals_new.copy(),ObeseChow_D12450J_night_meals_new.copy()
            best_ObeseHFD_D12492_night_meals,ObeseHFD_D12492_night_meals_min,ObeseHFD_D12492_night_meals_max = ObeseHFD_D12492_night_meals_new.copy(),ObeseHFD_D12492_night_meals_new.copy(),ObeseHFD_D12492_night_meals_new.copy()
            best_ObeseHFD_Sema10_night_meals,ObeseHFD_Sema10_night_meals_min,ObeseHFD_Sema10_night_meals_max = ObeseHFD_Sema10_night_meals_new.copy(),ObeseHFD_Sema10_night_meals_new.copy(),ObeseHFD_Sema10_night_meals_new.copy()
            best_ObeseHFD_Sema2_night_meals,ObeseHFD_Sema2_night_meals_min,ObeseHFD_Sema2_night_meals_max = ObeseHFD_Sema2_night_meals_new.copy(),ObeseHFD_Sema2_night_meals_new.copy(),ObeseHFD_Sema2_night_meals_new.copy()
            first = False
        else:

            # ---------- Appetite components ----------
            HealthyChow_D12450J_NonLipid_satiety_min,HealthyChow_D12450J_NonLipid_satiety_max = np.minimum(HealthyChow_D12450J_NonLipid_satiety_min,NonLipid_satiety_new),np.maximum(HealthyChow_D12450J_NonLipid_satiety_max,NonLipid_satiety_new)
            HealthyChow_D12450J_Lipid_satiety_min,HealthyChow_D12450J_Lipid_satiety_max = np.minimum(HealthyChow_D12450J_Lipid_satiety_min,Lipid_satiety_new),np.maximum(HealthyChow_D12450J_Lipid_satiety_max,Lipid_satiety_new)
            HealthyChow_D12450J_NonLipidandLipid_ratio_min,HealthyChow_D12450J_NonLipidandLipid_ratio_max = np.minimum(HealthyChow_D12450J_NonLipidandLipid_ratio_min,NonLipidandLipid_ratio_new),np.maximum(HealthyChow_D12450J_NonLipidandLipid_ratio_max,NonLipidandLipid_ratio_new)
            HealthyChow_D12450J_EIappetite_min,HealthyChow_D12450J_EIappetite_max = np.minimum(HealthyChow_D12450J_EIappetite_min,HealthyChow_D12450J_EIappetite_new),np.maximum(HealthyChow_D12450J_EIappetite_max,HealthyChow_D12450J_EIappetite_new)
            HealthyChow_D12450J_DN_min,HealthyChow_D12450J_DN_max = np.minimum(HealthyChow_D12450J_DN_min,HealthyChow_D12450J_DN_new),np.maximum(HealthyChow_D12450J_DN_max,HealthyChow_D12450J_DN_new)
            # ---------- Total body mass ----------
            HealthyChow_D12450J_BW_min,HealthyChow_D12450J_BW_max = np.minimum(HealthyChow_D12450J_BW_min,HealthyChow_D12450J_BW_new),np.maximum(HealthyChow_D12450J_BW_max,HealthyChow_D12450J_BW_new)
            HealthyHFD_D12492_BW_min,HealthyHFD_D12492_BW_max = np.minimum(HealthyHFD_D12492_BW_min,HealthyHFD_D12492_BW_new),np.maximum(HealthyHFD_D12492_BW_max,HealthyHFD_D12492_BW_new)
            HealthyHFD_D12331_BW_min,HealthyHFD_D12331_BW_max = np.minimum(HealthyHFD_D12331_BW_min,HealthyHFD_D12331_BW_new),np.maximum(HealthyHFD_D12331_BW_max,HealthyHFD_D12331_BW_new)
            HealthyControl_D124512_BW_min,HealthyControl_D124512_BW_max = np.minimum(HealthyControl_D124512_BW_min,HealthyControl_D124512_BW_new),np.maximum(HealthyControl_D124512_BW_max,HealthyControl_D124512_BW_new)
            HealthyControl_TekladLM_485_BW_min,HealthyControl_TekladLM_485_BW_max = np.minimum(HealthyControl_TekladLM_485_BW_min,HealthyControl_TekladLM_485_BW_new),np.maximum(HealthyControl_TekladLM_485_BW_max,HealthyControl_TekladLM_485_BW_new)
            ObeseChow_D12450J_BW_min,ObeseChow_D12450J_BW_max = np.minimum(ObeseChow_D12450J_BW_min,ObeseChow_D12450J_BW_new),np.maximum(ObeseChow_D12450J_BW_max,ObeseChow_D12450J_BW_new)
            ObeseHFD_D12492_BW_min,ObeseHFD_D12492_BW_max = np.minimum(ObeseHFD_D12492_BW_min,ObeseHFD_D12492_BW_new),np.maximum(ObeseHFD_D12492_BW_max,ObeseHFD_D12492_BW_new)
            ObeseHFD_Sema10_BW_min,ObeseHFD_Sema10_BW_max = np.minimum(ObeseHFD_Sema10_BW_min,ObeseHFD_Sema10_BW_new),np.maximum(ObeseHFD_Sema10_BW_max,ObeseHFD_Sema10_BW_new)
            ObeseHFD_Sema2_BW_min,ObeseHFD_Sema2_BW_max = np.minimum(ObeseHFD_Sema2_BW_min,ObeseHFD_Sema2_BW_new),np.maximum(ObeseHFD_Sema2_BW_max,ObeseHFD_Sema2_BW_new)
            # ---------- Daily average glucose ----------
            HealthyChow_D12450J_GlucoseDaily_min,HealthyChow_D12450J_GlucoseDaily_max = np.minimum(HealthyChow_D12450J_GlucoseDaily_min,HealthyChow_D12450J_GlucoseDaily_new),np.maximum(HealthyChow_D12450J_GlucoseDaily_max,HealthyChow_D12450J_GlucoseDaily_new)
            HealthyHFD_D12492_GlucoseDaily_min,HealthyHFD_D12492_GlucoseDaily_max = np.minimum(HealthyHFD_D12492_GlucoseDaily_min,HealthyHFD_D12492_GlucoseDaily_new),np.maximum(HealthyHFD_D12492_GlucoseDaily_max,HealthyHFD_D12492_GlucoseDaily_new)
            HealthyHFD_D12331_GlucoseDaily_min,HealthyHFD_D12331_GlucoseDaily_max = np.minimum(HealthyHFD_D12331_GlucoseDaily_min,HealthyHFD_D12331_GlucoseDaily_new),np.maximum(HealthyHFD_D12331_GlucoseDaily_max,HealthyHFD_D12331_GlucoseDaily_new)
            HealthyControl_D124512_GlucoseDaily_min,HealthyControl_D124512_GlucoseDaily_max = np.minimum(HealthyControl_D124512_GlucoseDaily_min,HealthyControl_D124512_GlucoseDaily_new),np.maximum(HealthyControl_D124512_GlucoseDaily_max,HealthyControl_D124512_GlucoseDaily_new)
            HealthyControl_TekladLM_485_GlucoseDaily_min,HealthyControl_TekladLM_485_GlucoseDaily_max = np.minimum(HealthyControl_TekladLM_485_GlucoseDaily_min,HealthyControl_TekladLM_485_GlucoseDaily_new),np.maximum(HealthyControl_TekladLM_485_GlucoseDaily_max,HealthyControl_TekladLM_485_GlucoseDaily_new)
            ObeseChow_D12450J_GlucoseDaily_min,ObeseChow_D12450J_GlucoseDaily_max = np.minimum(ObeseChow_D12450J_GlucoseDaily_min,ObeseChow_D12450J_GlucoseDaily_new),np.maximum(ObeseChow_D12450J_GlucoseDaily_max,ObeseChow_D12450J_GlucoseDaily_new)
            ObeseHFD_D12492_GlucoseDaily_min,ObeseHFD_D12492_GlucoseDaily_max = np.minimum(ObeseHFD_D12492_GlucoseDaily_min,ObeseHFD_D12492_GlucoseDaily_new),np.maximum(ObeseHFD_D12492_GlucoseDaily_max,ObeseHFD_D12492_GlucoseDaily_new)
            ObeseHFD_Sema10_GlucoseDaily_min,ObeseHFD_Sema10_GlucoseDaily_max = np.minimum(ObeseHFD_Sema10_GlucoseDaily_min,ObeseHFD_Sema10_GlucoseDaily_new),np.maximum(ObeseHFD_Sema10_GlucoseDaily_max,ObeseHFD_Sema10_GlucoseDaily_new)
            ObeseHFD_Sema2_GlucoseDaily_min,ObeseHFD_Sema2_GlucoseDaily_max = np.minimum(ObeseHFD_Sema2_GlucoseDaily_min,ObeseHFD_Sema2_GlucoseDaily_new),np.maximum(ObeseHFD_Sema2_GlucoseDaily_max,ObeseHFD_Sema2_GlucoseDaily_new)
            # ---------- Full-resolution glucose ----------
            HealthyChow_D12450J_Glucose_min,HealthyChow_D12450J_Glucose_max = np.minimum(HealthyChow_D12450J_Glucose_min,HealthyChow_D12450J_Glucose_new),np.maximum(HealthyChow_D12450J_Glucose_max,HealthyChow_D12450J_Glucose_new)
            HealthyHFD_D12492_Glucose_min,HealthyHFD_D12492_Glucose_max = np.minimum(HealthyHFD_D12492_Glucose_min,HealthyHFD_D12492_Glucose_new),np.maximum(HealthyHFD_D12492_Glucose_max,HealthyHFD_D12492_Glucose_new)
            HealthyHFD_D12331_Glucose_min,HealthyHFD_D12331_Glucose_max = np.minimum(HealthyHFD_D12331_Glucose_min,HealthyHFD_D12331_Glucose_new),np.maximum(HealthyHFD_D12331_Glucose_max,HealthyHFD_D12331_Glucose_new)
            HealthyControl_D124512_Glucose_min,HealthyControl_D124512_Glucose_max = np.minimum(HealthyControl_D124512_Glucose_min,HealthyControl_D124512_Glucose_new),np.maximum(HealthyControl_D124512_Glucose_max,HealthyControl_D124512_Glucose_new)
            HealthyControl_TekladLM_485_Glucose_min,HealthyControl_TekladLM_485_Glucose_max = np.minimum(HealthyControl_TekladLM_485_Glucose_min,HealthyControl_TekladLM_485_Glucose_new),np.maximum(HealthyControl_TekladLM_485_Glucose_max,HealthyControl_TekladLM_485_Glucose_new)
            ObeseChow_D12450J_Glucose_min,ObeseChow_D12450J_Glucose_max = np.minimum(ObeseChow_D12450J_Glucose_min,ObeseChow_D12450J_Glucose_new),np.maximum(ObeseChow_D12450J_Glucose_max,ObeseChow_D12450J_Glucose_new)
            ObeseHFD_D12492_Glucose_min,ObeseHFD_D12492_Glucose_max = np.minimum(ObeseHFD_D12492_Glucose_min,ObeseHFD_D12492_Glucose_new),np.maximum(ObeseHFD_D12492_Glucose_max,ObeseHFD_D12492_Glucose_new)
            ObeseHFD_Sema10_Glucose_min,ObeseHFD_Sema10_Glucose_max = np.minimum(ObeseHFD_Sema10_Glucose_min,ObeseHFD_Sema10_Glucose_new),np.maximum(ObeseHFD_Sema10_Glucose_max,ObeseHFD_Sema10_Glucose_new)
            ObeseHFD_Sema2_Glucose_min,ObeseHFD_Sema2_Glucose_max = np.minimum(ObeseHFD_Sema2_Glucose_min,ObeseHFD_Sema2_Glucose_new),np.maximum(ObeseHFD_Sema2_Glucose_max,ObeseHFD_Sema2_Glucose_new)
            # ---------- Meals per day ----------
            HealthyChow_D12450J_meals_min,HealthyChow_D12450J_meals_max = np.minimum(HealthyChow_D12450J_meals_min,HealthyChow_D12450J_meals_new),np.maximum(HealthyChow_D12450J_meals_max,HealthyChow_D12450J_meals_new)
            HealthyHFD_D12492_meals_min,HealthyHFD_D12492_meals_max = np.minimum(HealthyHFD_D12492_meals_min,HealthyHFD_D12492_meals_new),np.maximum(HealthyHFD_D12492_meals_max,HealthyHFD_D12492_meals_new)
            HealthyHFD_D12331_meals_min,HealthyHFD_D12331_meals_max = np.minimum(HealthyHFD_D12331_meals_min,HealthyHFD_D12331_meals_new),np.maximum(HealthyHFD_D12331_meals_max,HealthyHFD_D12331_meals_new)
            HealthyControl_D124512_meals_min,HealthyControl_D124512_meals_max = np.minimum(HealthyControl_D124512_meals_min,HealthyControl_D124512_meals_new),np.maximum(HealthyControl_D124512_meals_max,HealthyControl_D124512_meals_new)
            HealthyControl_TekladLM_485_meals_min,HealthyControl_TekladLM_485_meals_max = np.minimum(HealthyControl_TekladLM_485_meals_min,HealthyControl_TekladLM_485_meals_new),np.maximum(HealthyControl_TekladLM_485_meals_max,HealthyControl_TekladLM_485_meals_new)
            ObeseChow_D12450J_meals_min,ObeseChow_D12450J_meals_max = np.minimum(ObeseChow_D12450J_meals_min,ObeseChow_D12450J_meals_new),np.maximum(ObeseChow_D12450J_meals_max,ObeseChow_D12450J_meals_new)
            ObeseHFD_D12492_meals_min,ObeseHFD_D12492_meals_max = np.minimum(ObeseHFD_D12492_meals_min,ObeseHFD_D12492_meals_new),np.maximum(ObeseHFD_D12492_meals_max,ObeseHFD_D12492_meals_new)
            ObeseHFD_Sema10_meals_min,ObeseHFD_Sema10_meals_max = np.minimum(ObeseHFD_Sema10_meals_min,ObeseHFD_Sema10_meals_new),np.maximum(ObeseHFD_Sema10_meals_max,ObeseHFD_Sema10_meals_new)
            ObeseHFD_Sema2_meals_min,ObeseHFD_Sema2_meals_max = np.minimum(ObeseHFD_Sema2_meals_min,ObeseHFD_Sema2_meals_new),np.maximum(ObeseHFD_Sema2_meals_max,ObeseHFD_Sema2_meals_new)
            # ---------- Meal size ----------
            HealthyChow_D12450J_meal_size_min,HealthyChow_D12450J_meal_size_max = np.minimum(HealthyChow_D12450J_meal_size_min,HealthyChow_D12450J_meal_size_new),np.maximum(HealthyChow_D12450J_meal_size_max,HealthyChow_D12450J_meal_size_new)
            HealthyHFD_D12492_meal_size_min,HealthyHFD_D12492_meal_size_max = np.minimum(HealthyHFD_D12492_meal_size_min,HealthyHFD_D12492_meal_size_new),np.maximum(HealthyHFD_D12492_meal_size_max,HealthyHFD_D12492_meal_size_new)
            HealthyHFD_D12331_meal_size_min,HealthyHFD_D12331_meal_size_max = np.minimum(HealthyHFD_D12331_meal_size_min,HealthyHFD_D12331_meal_size_new),np.maximum(HealthyHFD_D12331_meal_size_max,HealthyHFD_D12331_meal_size_new)
            HealthyControl_D124512_meal_size_min,HealthyControl_D124512_meal_size_max = np.minimum(HealthyControl_D124512_meal_size_min,HealthyControl_D124512_meal_size_new),np.maximum(HealthyControl_D124512_meal_size_max,HealthyControl_D124512_meal_size_new)
            HealthyControl_TekladLM_485_meal_size_min,HealthyControl_TekladLM_485_meal_size_max = np.minimum(HealthyControl_TekladLM_485_meal_size_min,HealthyControl_TekladLM_485_meal_size_new),np.maximum(HealthyControl_TekladLM_485_meal_size_max,HealthyControl_TekladLM_485_meal_size_new)
            ObeseChow_D12450J_meal_size_min,ObeseChow_D12450J_meal_size_max = np.minimum(ObeseChow_D12450J_meal_size_min,ObeseChow_D12450J_meal_size_new),np.maximum(ObeseChow_D12450J_meal_size_max,ObeseChow_D12450J_meal_size_new)
            ObeseHFD_D12492_meal_size_min,ObeseHFD_D12492_meal_size_max = np.minimum(ObeseHFD_D12492_meal_size_min,ObeseHFD_D12492_meal_size_new),np.maximum(ObeseHFD_D12492_meal_size_max,ObeseHFD_D12492_meal_size_new)
            ObeseHFD_Sema10_meal_size_min,ObeseHFD_Sema10_meal_size_max = np.minimum(ObeseHFD_Sema10_meal_size_min,ObeseHFD_Sema10_meal_size_new),np.maximum(ObeseHFD_Sema10_meal_size_max,ObeseHFD_Sema10_meal_size_new)
            ObeseHFD_Sema2_meal_size_min,ObeseHFD_Sema2_meal_size_max = np.minimum(ObeseHFD_Sema2_meal_size_min,ObeseHFD_Sema2_meal_size_new),np.maximum(ObeseHFD_Sema2_meal_size_max,ObeseHFD_Sema2_meal_size_new)
            # ---------- Daily energy intake ----------
            HealthyChow_D12450J_daily_ei_min,HealthyChow_D12450J_daily_ei_max = np.minimum(HealthyChow_D12450J_daily_ei_min,HealthyChow_D12450J_daily_ei_new),np.maximum(HealthyChow_D12450J_daily_ei_max,HealthyChow_D12450J_daily_ei_new)
            HealthyHFD_D12492_daily_ei_min,HealthyHFD_D12492_daily_ei_max = np.minimum(HealthyHFD_D12492_daily_ei_min,HealthyHFD_D12492_daily_ei_new),np.maximum(HealthyHFD_D12492_daily_ei_max,HealthyHFD_D12492_daily_ei_new)
            HealthyHFD_D12331_daily_ei_min,HealthyHFD_D12331_daily_ei_max = np.minimum(HealthyHFD_D12331_daily_ei_min,HealthyHFD_D12331_daily_ei_new),np.maximum(HealthyHFD_D12331_daily_ei_max,HealthyHFD_D12331_daily_ei_new)
            HealthyControl_D124512_daily_ei_min,HealthyControl_D124512_daily_ei_max = np.minimum(HealthyControl_D124512_daily_ei_min,HealthyControl_D124512_daily_ei_new),np.maximum(HealthyControl_D124512_daily_ei_max,HealthyControl_D124512_daily_ei_new)
            HealthyControl_TekladLM_485_daily_ei_min,HealthyControl_TekladLM_485_daily_ei_max = np.minimum(HealthyControl_TekladLM_485_daily_ei_min,HealthyControl_TekladLM_485_daily_ei_new),np.maximum(HealthyControl_TekladLM_485_daily_ei_max,HealthyControl_TekladLM_485_daily_ei_new)
            ObeseChow_D12450J_daily_ei_min,ObeseChow_D12450J_daily_ei_max = np.minimum(ObeseChow_D12450J_daily_ei_min,ObeseChow_D12450J_daily_ei_new),np.maximum(ObeseChow_D12450J_daily_ei_max,ObeseChow_D12450J_daily_ei_new)
            ObeseHFD_D12492_daily_ei_min,ObeseHFD_D12492_daily_ei_max = np.minimum(ObeseHFD_D12492_daily_ei_min,ObeseHFD_D12492_daily_ei_new),np.maximum(ObeseHFD_D12492_daily_ei_max,ObeseHFD_D12492_daily_ei_new)
            ObeseHFD_Sema10_daily_ei_min,ObeseHFD_Sema10_daily_ei_max = np.minimum(ObeseHFD_Sema10_daily_ei_min,ObeseHFD_Sema10_daily_ei_new),np.maximum(ObeseHFD_Sema10_daily_ei_max,ObeseHFD_Sema10_daily_ei_new)
            ObeseHFD_Sema2_daily_ei_min,ObeseHFD_Sema2_daily_ei_max = np.minimum(ObeseHFD_Sema2_daily_ei_min,ObeseHFD_Sema2_daily_ei_new),np.maximum(ObeseHFD_Sema2_daily_ei_max,ObeseHFD_Sema2_daily_ei_new)
            # ---------- Night meals ----------
            HealthyChow_D12450J_night_meals_min,HealthyChow_D12450J_night_meals_max = np.minimum(HealthyChow_D12450J_night_meals_min,HealthyChow_D12450J_night_meals_new),np.maximum(HealthyChow_D12450J_night_meals_max,HealthyChow_D12450J_night_meals_new)
            HealthyHFD_D12492_night_meals_min,HealthyHFD_D12492_night_meals_max = np.minimum(HealthyHFD_D12492_night_meals_min,HealthyHFD_D12492_night_meals_new),np.maximum(HealthyHFD_D12492_night_meals_max,HealthyHFD_D12492_night_meals_new)
            HealthyHFD_D12331_night_meals_min,HealthyHFD_D12331_night_meals_max = np.minimum(HealthyHFD_D12331_night_meals_min,HealthyHFD_D12331_night_meals_new),np.maximum(HealthyHFD_D12331_night_meals_max,HealthyHFD_D12331_night_meals_new)
            HealthyControl_D124512_night_meals_min,HealthyControl_D124512_night_meals_max = np.minimum(HealthyControl_D124512_night_meals_min,HealthyControl_D124512_night_meals_new),np.maximum(HealthyControl_D124512_night_meals_max,HealthyControl_D124512_night_meals_new)
            HealthyControl_TekladLM_485_night_meals_min,HealthyControl_TekladLM_485_night_meals_max = np.minimum(HealthyControl_TekladLM_485_night_meals_min,HealthyControl_TekladLM_485_night_meals_new),np.maximum(HealthyControl_TekladLM_485_night_meals_max,HealthyControl_TekladLM_485_night_meals_new)
            ObeseChow_D12450J_night_meals_min,ObeseChow_D12450J_night_meals_max = np.minimum(ObeseChow_D12450J_night_meals_min,ObeseChow_D12450J_night_meals_new),np.maximum(ObeseChow_D12450J_night_meals_max,ObeseChow_D12450J_night_meals_new)
            ObeseHFD_D12492_night_meals_min,ObeseHFD_D12492_night_meals_max = np.minimum(ObeseHFD_D12492_night_meals_min,ObeseHFD_D12492_night_meals_new),np.maximum(ObeseHFD_D12492_night_meals_max,ObeseHFD_D12492_night_meals_new)
            ObeseHFD_Sema10_night_meals_min,ObeseHFD_Sema10_night_meals_max = np.minimum(ObeseHFD_Sema10_night_meals_min,ObeseHFD_Sema10_night_meals_new),np.maximum(ObeseHFD_Sema10_night_meals_max,ObeseHFD_Sema10_night_meals_new)
            ObeseHFD_Sema2_night_meals_min,ObeseHFD_Sema2_night_meals_max = np.minimum(ObeseHFD_Sema2_night_meals_min,ObeseHFD_Sema2_night_meals_new),np.maximum(ObeseHFD_Sema2_night_meals_max,ObeseHFD_Sema2_night_meals_new)
    if first:
        raise RuntimeError("All digital population simulations failed.")
    
    def compute_is_night(time, phase_shift_hours=4):
        hours = time * 24
        cycle = np.sin(2 * np.pi * (hours + phase_shift_hours) / 24)
        return cycle > 0
    IsNight = compute_is_night(time_regular)
    change_idx = np.where(np.diff(IsNight.astype(int)) != 0)[0] + 1
    edges = np.concatenate(([0], change_idx, [len(time_regular) - 1]))

    time = best_sim["HealthyChow_D12450J"].time_vector
    time_daily_ei = np.arange(len(best_HealthyChow_D12450J_daily_ei), dtype=float)


    print(
        f"Best fit for non-lipids compared to lipid (saturation/kcal): {best_HealthyChow_D12450J_NonLipidandLipid_ratio:.3f} "
        f"(min: {HealthyChow_D12450J_NonLipidandLipid_ratio_min:.3f} - "
        f"max: {HealthyChow_D12450J_NonLipidandLipid_ratio_max:.3f})"
    )

   # ---------- Fig. 4: Appetite dependencies ----------
    fig_dn, axes_dn = plt.subplots(1, 4, figsize=(21, 5))
    #  Long-term dependency (8 weeks)
    axes_dn[0].fill_between(time/7,HealthyChow_D12450J_DN_min,HealthyChow_D12450J_DN_max,color="black",alpha=0.2)
    axes_dn[0].plot(time/7,best_HealthyChow_D12450J_DN,color="black",lw=2)
    for i in range(len(edges) - 1):
        if IsNight[edges[i]]:
            axes_dn[0].axvspan(time[edges[i]]/7, time[edges[i + 1]]/7, color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    axes_dn[0].set_xlim(0,4)
    axes_dn[0].set_ylim(0,20)
    axes_dn[0].set_yticks([0,5,10,15,20])
    axes_dn[0].set_xticks([0,1,2,3,4])
    axes_dn[0].set_xlabel("Time (weeks)",fontsize=18)
    axes_dn[0].set_ylabel("Time-dependent appetite\n(1/day)",fontsize=18)
    #  Time dependency 
    first_day_mask = (time >= 0) & (time <= 1)
    axes_dn[1].fill_between(time[first_day_mask],HealthyChow_D12450J_DN_min[first_day_mask],HealthyChow_D12450J_DN_max[first_day_mask],color="black",alpha=0.2)
    axes_dn[1].plot(time[first_day_mask],best_HealthyChow_D12450J_DN[first_day_mask],color="black",lw=2)
    for k in range(len(edges)-1):
        if IsNight[edges[k]] and time[edges[k]] <= 1:
            axes_dn[1].axvspan(max(time[edges[k]],0),min(time[edges[k+1]],1),color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    axes_dn[1].set_xlim(0,1)
    axes_dn[1].set_ylim(0,20)
    axes_dn[1].set_yticks([0,5,10,15,20])
    axes_dn[1].set_xticks([0,0.5,1])
    axes_dn[1].set_xticklabels(["00:00","12:00","24:00"])
    axes_dn[1].set_xlabel("Time of day",fontsize=18)
    axes_dn[1].set_ylabel("Time-dependent appetite\n(1/day)",fontsize=18)
    #  Fed state dependency 
    axes_dn[2].fill_between(x_fedstate,HealthyChow_D12450J_EIappetite_min,HealthyChow_D12450J_EIappetite_max,color="black",alpha=0.2)
    axes_dn[2].plot(x_fedstate,best_HealthyChow_D12450J_EIappetite,color="black",lw=2)
    axes_dn[2].set_xlim(5,25)
    axes_dn[2].set_ylim(1e-4,1e4)
    axes_dn[2].set_xticks([5,10,15,20,25])
    axes_dn[2].set_yscale('log')
    axes_dn[2].set_yticks([1e-4,1e-2,1e0,1e2,1e4])
    axes_dn[2].set_xlabel("Energy intake (kcal/day)",fontsize=18)
    axes_dn[2].set_ylabel("Appetite regulation\n(multiplication)",fontsize=18)
    #  Diet dependency 
    diet_x_positions = np.array([1,2])
    diet_labels = ["Non-lipids","Lipids"]
    nonlipid_min = float(np.asarray(HealthyChow_D12450J_NonLipid_satiety_min))
    nonlipid_max = float(np.asarray(HealthyChow_D12450J_NonLipid_satiety_max))
    lipid_min = float(np.asarray(HealthyChow_D12450J_Lipid_satiety_min))
    lipid_max = float(np.asarray(HealthyChow_D12450J_Lipid_satiety_max))
    best_nonlipid = float(np.asarray(best_HealthyChow_D12450J_NonLipid_satiety))
    best_lipid = float(np.asarray(best_HealthyChow_D12450J_Lipid_satiety))
    axes_dn[3].add_patch(plt.Rectangle((1-0.30,nonlipid_min),0.60,nonlipid_max-nonlipid_min,color="black",alpha=0.18,lw=0))
    axes_dn[3].add_patch(plt.Rectangle((2-0.30,lipid_min),0.60,lipid_max-lipid_min,color="black",alpha=0.18,lw=0))
    axes_dn[3].add_patch(plt.Rectangle((1-0.16,0),0.32,best_nonlipid,facecolor="grey",edgecolor="black"))
    axes_dn[3].add_patch(plt.Rectangle((2-0.16,0),0.32,best_lipid,facecolor="grey",edgecolor="black"))
    axes_dn[3].set_xlim(0.5,2.5)
    axes_dn[3].set_ylim(0,100)
    axes_dn[3].set_xticks(diet_x_positions)
    axes_dn[3].set_xticklabels(diet_labels)
    axes_dn[3].set_ylabel("Digested macroneutrient\n(Satiety/kcal)",fontsize=18)
    for ax in axes_dn:
        ax.grid(True)
        ax.tick_params(axis="both",which="major",labelsize=18)
        ax.tick_params(axis="both",which="minor",labelsize=18)
        plt.setp(ax.get_xticklabels(),fontsize=18)
        plt.setp(ax.get_yticklabels(),fontsize=18)
    fig_dn.subplots_adjust(left=0.07,right=0.99,bottom=0.18,top=0.95,wspace=0.3)
    plt.show()

   
    # ==========================================================
    # FIGURE 1: HEALTHY MICE ON DIFFERENT DIETS
    # ==========================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    colors1 = {"HealthyChow_D12450J": "black", "HealthyHFD_D12492": "red", "HealthyHFD_D12331": "orange", "HealthyControl_D124512": "blue", "HealthyControl_TekladLM_485": "green"}
    #  Daily energy intake 
    axes1[0].fill_between(time_daily_ei/7, HealthyChow_D12450J_daily_ei_min, HealthyChow_D12450J_daily_ei_max, color=colors1["HealthyChow_D12450J"], alpha=0.15, edgecolor="none")
    axes1[0].plot(time_daily_ei/7, best_HealthyChow_D12450J_daily_ei, color=colors1["HealthyChow_D12450J"], lw=2, label="Mice maintained on Chow (D12450J)")
    axes1[0].fill_between(time_daily_ei/7, HealthyHFD_D12492_daily_ei_min, HealthyHFD_D12492_daily_ei_max, color=colors1["HealthyHFD_D12492"], alpha=0.15, edgecolor="none")
    axes1[0].plot(time_daily_ei/7, best_HealthyHFD_D12492_daily_ei, color=colors1["HealthyHFD_D12492"], lw=2, label="change to high fat diet (D12492)")
    axes1[0].fill_between(time_daily_ei/7, HealthyHFD_D12331_daily_ei_min, HealthyHFD_D12331_daily_ei_max, color=colors1["HealthyHFD_D12331"], alpha=0.15, edgecolor="none")
    axes1[0].plot(time_daily_ei/7, best_HealthyHFD_D12331_daily_ei, color=colors1["HealthyHFD_D12331"], lw=2, label="change to high fat diet (D12331)")
    axes1[0].fill_between(time_daily_ei/7, HealthyControl_D124512_daily_ei_min, HealthyControl_D124512_daily_ei_max, color=colors1["HealthyControl_D124512"], alpha=0.15, edgecolor="none")
    axes1[0].plot(time_daily_ei/7, best_HealthyControl_D124512_daily_ei, color=colors1["HealthyControl_D124512"], lw=2, label="change to balanced diet (D124512)")
    axes1[0].fill_between(time_daily_ei/7, HealthyControl_TekladLM_485_daily_ei_min, HealthyControl_TekladLM_485_daily_ei_max, color=colors1["HealthyControl_TekladLM_485"], alpha=0.15, edgecolor="none")
    axes1[0].plot(time_daily_ei/7, best_HealthyControl_TekladLM_485_daily_ei, color=colors1["HealthyControl_TekladLM_485"], lw=2, label="change to high carbohydrate diet (Teklad LM-485)")
    axes1[0].set_xlabel("Time (weeks)", fontsize=18)
    axes1[0].set_ylabel("Daily energy intake\n(kcal/day)", fontsize=18)
    axes1[0].set_xlim(-0.1, 8.1)
    axes1[0].set_yticks([10,15,20,25])
    axes1[0].set_ylim(10, 30)
    axes1[0].set_xticks([0, 2, 4, 6, 8])
    axes1[0].plot([1.8,2.2],[10,10], color='black', lw=10, solid_capstyle='butt')
    axes1[0].legend(loc="best", framealpha=1, fontsize=11,ncol=2)
    #  Total body mass 
    axes1[1].fill_between(time/7, HealthyChow_D12450J_BW_min, HealthyChow_D12450J_BW_max, color=colors1["HealthyChow_D12450J"], alpha=0.15, edgecolor="none")
    axes1[1].plot(time/7, best_HealthyChow_D12450J_BW, color=colors1["HealthyChow_D12450J"], lw=2, label="D12450J")
    axes1[1].fill_between(time/7, HealthyHFD_D12492_BW_min, HealthyHFD_D12492_BW_max, color=colors1["HealthyHFD_D12492"], alpha=0.15, edgecolor="none")
    axes1[1].plot(time/7, best_HealthyHFD_D12492_BW, color=colors1["HealthyHFD_D12492"], lw=2, label="D12492")
    axes1[1].fill_between(time/7, HealthyHFD_D12331_BW_min, HealthyHFD_D12331_BW_max, color=colors1["HealthyHFD_D12331"], alpha=0.15, edgecolor="none")
    axes1[1].plot(time/7, best_HealthyHFD_D12331_BW, color=colors1["HealthyHFD_D12331"], lw=2, label="D12331")
    axes1[1].fill_between(time/7, HealthyControl_D124512_BW_min, HealthyControl_D124512_BW_max, color=colors1["HealthyControl_D124512"], alpha=0.15, edgecolor="none")
    axes1[1].plot(time/7, best_HealthyControl_D124512_BW, color=colors1["HealthyControl_D124512"], lw=2, label="D124512")
    axes1[1].fill_between(time/7, HealthyControl_TekladLM_485_BW_min, HealthyControl_TekladLM_485_BW_max, color=colors1["HealthyControl_TekladLM_485"], alpha=0.15, edgecolor="none")
    axes1[1].plot(time/7, best_HealthyControl_TekladLM_485_BW, color=colors1["HealthyControl_TekladLM_485"], lw=2, label="Teklad LM-485")
    axes1[1].set_xlabel("Time (weeks)", fontsize=18)
    axes1[1].set_ylabel("Total body mass\n(g)", fontsize=18)
    axes1[1].set_xlim(-0.1, 8.1)
    axes1[1].set_ylim(25, 38)
    axes1[1].set_xticks([0, 2, 4, 6, 8])
    axes1[1].set_yticks([25,30,35])
    axes1[1].plot([1.8,2.2],[25,25], color='black', lw=10, solid_capstyle='butt')
    for ax in axes1:
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True)
        for i in range(len(edges) - 1):
            if IsNight[edges[i]]:
                ax.axvspan(time[edges[i]]/7, time[edges[i + 1]]/7, color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    fig1.subplots_adjust(left=0.07,right=0.99,bottom=0.12,top=0.95,wspace=0.25)
    plt.show()

    # ==========================================================
    # FIGURE 2: HEALTHY AND OBESE MICE ON D12450J
    # ==========================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    colors2 = {"HealthyChow_D12450J": "black", "ObeseChow_D12450J": "purple"}
    #  Daily energy intake 
    axes2[0].fill_between(time_daily_ei/7, HealthyChow_D12450J_daily_ei_min, HealthyChow_D12450J_daily_ei_max, color=colors2["HealthyChow_D12450J"], alpha=0.15, edgecolor="none")
    axes2[0].plot(time_daily_ei/7, best_HealthyChow_D12450J_daily_ei, color=colors2["HealthyChow_D12450J"], lw=2, label="Healthy mice maintained on Chow (D12450J)")
    axes2[0].fill_between(time_daily_ei/7, ObeseChow_D12450J_daily_ei_min, ObeseChow_D12450J_daily_ei_max, color=colors2["ObeseChow_D12450J"], alpha=0.15, edgecolor="none")
    axes2[0].plot(time_daily_ei/7, best_ObeseChow_D12450J_daily_ei, color=colors2["ObeseChow_D12450J"], lw=2, label="Obese mice changed to Chow (D12450J)")
    axes2[0].set_xlabel("Time (weeks)", fontsize=18)
    axes2[0].set_ylabel("Daily energy intake\n(kcal/day)", fontsize=18)
    axes2[0].set_xlim(-0.1, 8.1)
    axes2[0].set_xticks([0, 2, 4, 6, 8])
    axes2[0].set_yticks([5,10,15,20])
    axes2[0].set_ylim(5, 20)
    axes2[0].legend(loc="best", framealpha=1, fontsize=12)
    axes2[0].plot([1.8,2.2],[5,5], color='black', lw=10, solid_capstyle='butt')
    #  Total body mass 
    axes2[1].fill_between(time/7, HealthyChow_D12450J_BW_min, HealthyChow_D12450J_BW_max, color=colors2["HealthyChow_D12450J"], alpha=0.15, edgecolor="none")
    axes2[1].plot(time/7, best_HealthyChow_D12450J_BW, color=colors2["HealthyChow_D12450J"], lw=2, label="Healthy")
    axes2[1].fill_between(time/7, ObeseChow_D12450J_BW_min, ObeseChow_D12450J_BW_max, color=colors2["ObeseChow_D12450J"], alpha=0.15, edgecolor="none")
    axes2[1].plot(time/7, best_ObeseChow_D12450J_BW, color=colors2["ObeseChow_D12450J"], lw=2, label="Obese")
    axes2[1].set_xlabel("Time (weeks)", fontsize=18)
    axes2[1].set_ylabel("Total body mass\n(g)", fontsize=18)
    axes2[1].set_xlim(-0.1, 8.1)
    axes2[1].set_ylim(25, 64)
    axes2[1].set_xticks([0, 2, 4, 6, 8])
    axes2[1].set_yticks([25,35,45,55])
    axes2[1].plot([1.8,2.2],[25,25], color='black', lw=10, solid_capstyle='butt')
    for ax in axes2:
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True)
        for i in range(len(edges) - 1):
            if IsNight[edges[i]]:
                ax.axvspan(time[edges[i]]/7, time[edges[i + 1]]/7, color="black", alpha=0.08, edgecolor="none", linewidth=0, zorder=0)
    fig2.subplots_adjust(left=0.07,right=0.99,bottom=0.12,top=0.95,wspace=0.25)
    plt.show()

    # ==========================================================
    # FIGURE 3: DIET AND SEMAGLUTIDE IN OBESE MICE
    # ==========================================================
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 6))
    colors3 = {"ObeseHFD_D12492": "red", "ObeseChow_D12450J": "purple", "ObeseHFD_Sema2": "cyan", "ObeseHFD_Sema10": "magenta"}
    #  Final day of week 8 glucose profile 
    day_start = 55
    day_end = 56
    day_start_idx = int(day_start*24*TimeRes_exposure)
    day_end_idx = int(day_end*24*TimeRes_exposure)+1
    time_day = np.linspace(0,24,day_end_idx-day_start_idx)
    #  Daily energy intake 
    axes3[0].fill_between(time_daily_ei/7, ObeseHFD_D12492_daily_ei_min, ObeseHFD_D12492_daily_ei_max, color=colors3["ObeseHFD_D12492"], alpha=0.15, edgecolor="none")
    axes3[0].plot(time_daily_ei/7, best_ObeseHFD_D12492_daily_ei, color=colors3["ObeseHFD_D12492"], lw=2, label="Mice maintained on high fat diet (D12492)")
    axes3[0].fill_between(time_daily_ei/7, ObeseChow_D12450J_daily_ei_min, ObeseChow_D12450J_daily_ei_max, color=colors3["ObeseChow_D12450J"], alpha=0.15, edgecolor="none")
    axes3[0].plot(time_daily_ei/7, best_ObeseChow_D12450J_daily_ei, color=colors3["ObeseChow_D12450J"], lw=2, label="changed to Chow (D12450J)")
    axes3[0].fill_between(time_daily_ei/7, ObeseHFD_Sema2_daily_ei_min, ObeseHFD_Sema2_daily_ei_max, color=colors3["ObeseHFD_Sema2"], alpha=0.15, edgecolor="none")
    axes3[0].plot(time_daily_ei/7, best_ObeseHFD_Sema2_daily_ei, color=colors3["ObeseHFD_Sema2"], lw=2, label="changed to Chow (D12450J)\nwith 2 nmol/kg QD semaglutide")
    axes3[0].fill_between(time_daily_ei/7, ObeseHFD_Sema10_daily_ei_min, ObeseHFD_Sema10_daily_ei_max, color=colors3["ObeseHFD_Sema10"], alpha=0.15, edgecolor="none")
    axes3[0].plot(time_daily_ei/7, best_ObeseHFD_Sema10_daily_ei, color=colors3["ObeseHFD_Sema10"], lw=2, label="changed to Chow (D12450J)\nwith 10 nmol/kg QD semaglutide")
    axes3[0].set_xlabel("Time (weeks)", fontsize=18)
    axes3[0].set_ylabel("Daily energy intake\n(kcal/day)", fontsize=18)
    axes3[0].plot([1.8,2.2],[0,0], color='black', lw=10, solid_capstyle='butt')
    axes3[0].set_xlim(-0.1, 8.1)
    axes3[0].set_xticks([0, 2, 4, 6, 8])
    axes3[0].set_ylim(0, 17)
    axes3[0].set_yticks([0, 5, 10, 15])
    axes3[0].legend(loc="best", framealpha=1, fontsize=11)
    #  Total body mass 
    axes3[1].fill_between(time/7, ObeseHFD_D12492_BW_min, ObeseHFD_D12492_BW_max, color=colors3["ObeseHFD_D12492"], alpha=0.15, edgecolor="none")
    axes3[1].plot(time/7, best_ObeseHFD_D12492_BW, color=colors3["ObeseHFD_D12492"], lw=2, label="High fat diet")
    axes3[1].fill_between(time/7, ObeseChow_D12450J_BW_min, ObeseChow_D12450J_BW_max, color=colors3["ObeseChow_D12450J"], alpha=0.15, edgecolor="none")
    axes3[1].plot(time/7, best_ObeseChow_D12450J_BW, color=colors3["ObeseChow_D12450J"], lw=2, label="Chow diet")
    axes3[1].fill_between(time/7, ObeseHFD_Sema10_BW_min, ObeseHFD_Sema10_BW_max, color=colors3["ObeseHFD_Sema10"], alpha=0.15, edgecolor="none")
    axes3[1].plot(time/7, best_ObeseHFD_Sema10_BW, color=colors3["ObeseHFD_Sema10"], lw=2, label="Chow + 10 nmol/kg QD semaglutide")
    axes3[1].fill_between(time/7, ObeseHFD_Sema2_BW_min, ObeseHFD_Sema2_BW_max, color=colors3["ObeseHFD_Sema2"], alpha=0.15, edgecolor="none")
    axes3[1].plot(time/7, best_ObeseHFD_Sema2_BW, color=colors3["ObeseHFD_Sema2"], lw=2, label="Chow + 2 nmol/kg QD semaglutide")
    axes3[1].set_xlabel("Time (weeks)", fontsize=18)
    axes3[1].set_ylabel("Total body mass\n(g)", fontsize=18)
    axes3[1].plot([1.8,2.2],[34,34], color='black', lw=10, solid_capstyle='butt')
    axes3[1].set_xlim(-0.1, 8.1)
    axes3[1].set_xticks([0, 2, 4, 6, 8])
    axes3[1].set_ylim(34, 56)
    axes3[1].set_yticks([35, 40, 45, 50, 55])
    #  Daily average glucose 
    axes3[2].fill_between(time_Glucose/7,ObeseHFD_D12492_GlucoseDaily_min,ObeseHFD_D12492_GlucoseDaily_max,color=colors3["ObeseHFD_D12492"],alpha=0.15,edgecolor="none")
    axes3[2].plot(time_Glucose/7,best_ObeseHFD_D12492_GlucoseDaily,color=colors3["ObeseHFD_D12492"],lw=2,label="HFD")
    axes3[2].fill_between(time_Glucose/7,ObeseChow_D12450J_GlucoseDaily_min,ObeseChow_D12450J_GlucoseDaily_max,color=colors3["ObeseChow_D12450J"],alpha=0.15,edgecolor="none")
    axes3[2].plot(time_Glucose/7,best_ObeseChow_D12450J_GlucoseDaily,color=colors3["ObeseChow_D12450J"],lw=2,label="Chow")
    axes3[2].fill_between(time_Glucose/7,ObeseHFD_Sema10_GlucoseDaily_min,ObeseHFD_Sema10_GlucoseDaily_max,color=colors3["ObeseHFD_Sema10"],alpha=0.15,edgecolor="none")
    axes3[2].plot(time_Glucose/7,best_ObeseHFD_Sema10_GlucoseDaily,color=colors3["ObeseHFD_Sema10"],lw=2,label="HFD + semaglutide")
    axes3[2].fill_between(time_Glucose/7,ObeseHFD_Sema2_GlucoseDaily_min,ObeseHFD_Sema2_GlucoseDaily_max,color=colors3["ObeseHFD_Sema2"],alpha=0.15,edgecolor="none")
    axes3[2].plot(time_Glucose/7,best_ObeseHFD_Sema2_GlucoseDaily,color=colors3["ObeseHFD_Sema2"],lw=2,label="Chow + semaglutide")
    axes3[2].set_xlabel("Time (weeks)",fontsize=18)
    axes3[2].set_ylabel("Daily average glucose\n(mmol/L)",fontsize=18)
    axes3[2].plot([1.8,2.2],[8,8], color='black', lw=10, solid_capstyle='butt')
    axes3[2].set_xlim(-0.1, 8.1)
    axes3[2].set_xticks([0, 2, 4, 6, 8])
    axes3[2].set_ylim(8, 12)
    axes3[2].set_yticks([8, 10, 12, 14])
    #  One day of glucose (week 11); comment: uncertainty is out commented here to increase readability of the figure. Please add if you like, we only use it to compare best fits in manuscript
    #axes3[3].fill_between(time_day,ObeseHFD_D12492_Glucose_min[day_start_idx:day_end_idx],ObeseHFD_D12492_Glucose_max[day_start_idx:day_end_idx],color=colors3["ObeseHFD_D12492"],alpha=0.15,edgecolor="none")
    axes3[3].plot(time_day,best_ObeseHFD_D12492_Glucose[day_start_idx:day_end_idx],color=colors3["ObeseHFD_D12492"],lw=2,label="HFD")
    #axes3[3].fill_between(time_day,ObeseChow_D12450J_Glucose_min[day_start_idx:day_end_idx],ObeseChow_D12450J_Glucose_max[day_start_idx:day_end_idx],color=colors3["ObeseChow_D12450J"],alpha=0.15,edgecolor="none")
    axes3[3].plot(time_day,best_ObeseChow_D12450J_Glucose[day_start_idx:day_end_idx],color=colors3["ObeseChow_D12450J"],lw=2,label="Chow")
    #axes3[3].fill_between(time_day,ObeseHFD_Sema10_Glucose_min[day_start_idx:day_end_idx],ObeseHFD_Sema10_Glucose_max[day_start_idx:day_end_idx],color=colors3["ObeseHFD_Sema10"],alpha=0.15,edgecolor="none")
    axes3[3].plot(time_day,best_ObeseHFD_Sema10_Glucose[day_start_idx:day_end_idx],color=colors3["ObeseHFD_Sema10"],lw=2,label="10 nmol/kg")
    #axes3[3].fill_between(time_day,ObeseHFD_Sema2_Glucose_min[day_start_idx:day_end_idx],ObeseHFD_Sema2_Glucose_max[day_start_idx:day_end_idx],color=colors3["ObeseHFD_Sema2"],alpha=0.15,edgecolor="none")
    axes3[3].plot(time_day,best_ObeseHFD_Sema2_Glucose[day_start_idx:day_end_idx],color=colors3["ObeseHFD_Sema2"],lw=2,label="2 nmol/kg")
    axes3[3].set_xlabel("Time of day",fontsize=18)
    axes3[3].set_ylabel("Glucose\n(mmol/L)",fontsize=18)
    axes3[3].set_xlim(0, 14)
    axes3[3].set_xticks([0, 6, 12])
    axes3[3].set_ylim(9, 15)
    axes3[3].set_yticks([10, 12, 14])
    axes3[3].set_xticklabels(["00:00","06:00","12:00"])
    #  Formating 
    for ax in axes3[:3]:
        ax.tick_params(axis="both",which="major",labelsize=16)
        ax.grid(True)
        for i in range(len(edges)-1):
            if IsNight[edges[i]]:
                ax.axvspan(time[edges[i]]/7,time[edges[i+1]]/7,color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
    axes3[3].tick_params(axis="both",which="major",labelsize=16)
    axes3[3].grid(True)
    time_zoom=time[day_start_idx:day_end_idx]
    IsNight_zoom=IsNight[day_start_idx:day_end_idx]
    night_start=None
    for i in range(len(IsNight_zoom)):
        if IsNight_zoom[i] and night_start is None:
            night_start=(time_zoom[i]-time_zoom[0])*24
        if night_start is not None and (i==len(IsNight_zoom)-1 or not IsNight_zoom[i+1]):
            night_end=(time_zoom[i]-time_zoom[0])*24
            axes3[3].axvspan(night_start,night_end,color="black",alpha=0.08,edgecolor="none",linewidth=0,zorder=0)
            night_start=None
    fig3.subplots_adjust(left=0.07,right=0.99,bottom=0.12,top=0.95,wspace=0.3)
    plt.show()

    # ---------- Fig. 2: Glucose and meals ----------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=False)
    axes = axes.flatten()
    colors_meals = ["black","red","orange","blue","green","purple","brown","magenta","cyan"]
    meal_labels = ["Healthy\nChow\nD12450J","Healthy\nHFD\nD12492","Healthy\nHDF\nD12331","Healthy\nControl\nD124512","Healthy\nControl\nTeklad\nLM-485","Obese\nChow\nD12450J","Obese\nHFD\nD12492","Obese\nHFD\n2 nmol/kg","Obese\nHFD\n10 nmol/kg"]
    x_positions = np.arange(1, 10)
    #  COLUMN 1: Final glucose 
    axes[0].add_patch(plt.Rectangle((1-0.30,HealthyChow_D12450J_Glucose_min[-1]),0.60,HealthyChow_D12450J_Glucose_max[-1]-HealthyChow_D12450J_Glucose_min[-1],color=colors_meals[0],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((2-0.30,HealthyHFD_D12492_Glucose_min[-1]),0.60,HealthyHFD_D12492_Glucose_max[-1]-HealthyHFD_D12492_Glucose_min[-1],color=colors_meals[1],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((3-0.30,HealthyHFD_D12331_Glucose_min[-1]),0.60,HealthyHFD_D12331_Glucose_max[-1]-HealthyHFD_D12331_Glucose_min[-1],color=colors_meals[2],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((4-0.30,HealthyControl_D124512_Glucose_min[-1]),0.60,HealthyControl_D124512_Glucose_max[-1]-HealthyControl_D124512_Glucose_min[-1],color=colors_meals[3],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((5-0.30,HealthyControl_TekladLM_485_Glucose_min[-1]),0.60,HealthyControl_TekladLM_485_Glucose_max[-1]-HealthyControl_TekladLM_485_Glucose_min[-1],color=colors_meals[4],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((6-0.30,ObeseChow_D12450J_Glucose_min[-1]),0.60,ObeseChow_D12450J_Glucose_max[-1]-ObeseChow_D12450J_Glucose_min[-1],color=colors_meals[5],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((7-0.30,ObeseHFD_D12492_Glucose_min[-1]),0.60,ObeseHFD_D12492_Glucose_max[-1]-ObeseHFD_D12492_Glucose_min[-1],color=colors_meals[6],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((8-0.30,ObeseHFD_Sema2_Glucose_min[-1]),0.60,ObeseHFD_Sema2_Glucose_max[-1]-ObeseHFD_Sema2_Glucose_min[-1],color=colors_meals[8],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((9-0.30,ObeseHFD_Sema10_Glucose_min[-1]),0.60,ObeseHFD_Sema10_Glucose_max[-1]-ObeseHFD_Sema10_Glucose_min[-1],color=colors_meals[7],alpha=0.18,lw=0))
    axes[0].add_patch(plt.Rectangle((1-0.16,0),0.32,best_HealthyChow_D12450J_Glucose[-1],facecolor=colors_meals[0],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((2-0.16,0),0.32,best_HealthyHFD_D12492_Glucose[-1],facecolor=colors_meals[1],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((3-0.16,0),0.32,best_HealthyHFD_D12331_Glucose[-1],facecolor=colors_meals[2],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((4-0.16,0),0.32,best_HealthyControl_D124512_Glucose[-1],facecolor=colors_meals[3],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((5-0.16,0),0.32,best_HealthyControl_TekladLM_485_Glucose[-1],facecolor=colors_meals[4],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((6-0.16,0),0.32,best_ObeseChow_D12450J_Glucose[-1],facecolor=colors_meals[5],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((7-0.16,0),0.32,best_ObeseHFD_D12492_Glucose[-1],facecolor=colors_meals[6],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((8-0.16,0),0.32,best_ObeseHFD_Sema2_Glucose[-1],facecolor=colors_meals[8],edgecolor="black"))
    axes[0].add_patch(plt.Rectangle((9-0.16,0),0.32,best_ObeseHFD_Sema10_Glucose[-1],facecolor=colors_meals[7],edgecolor="black"))
    axes[0].set_xlim(0.5,9.5)
    axes[0].set_ylim(0,20)
    axes[0].set_yticks([0,5,10,15,20])
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(meal_labels,rotation=0,ha="center")
    axes[0].set_ylabel("Average plasma glucose\n(mmol/L)",fontsize=18)
    axes[0].grid(True,axis="y")
    #  COLUMN 2: Average meals per day 
    axes[1].add_patch(plt.Rectangle((1-0.30,HealthyChow_D12450J_meals_min),0.60,HealthyChow_D12450J_meals_max-HealthyChow_D12450J_meals_min,color=colors_meals[0],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((2-0.30,HealthyHFD_D12492_meals_min),0.60,HealthyHFD_D12492_meals_max-HealthyHFD_D12492_meals_min,color=colors_meals[1],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((3-0.30,HealthyHFD_D12331_meals_min),0.60,HealthyHFD_D12331_meals_max-HealthyHFD_D12331_meals_min,color=colors_meals[2],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((4-0.30,HealthyControl_D124512_meals_min),0.60,HealthyControl_D124512_meals_max-HealthyControl_D124512_meals_min,color=colors_meals[3],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((5-0.30,HealthyControl_TekladLM_485_meals_min),0.60,HealthyControl_TekladLM_485_meals_max-HealthyControl_TekladLM_485_meals_min,color=colors_meals[4],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((6-0.30,ObeseChow_D12450J_meals_min),0.60,ObeseChow_D12450J_meals_max-ObeseChow_D12450J_meals_min,color=colors_meals[5],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((7-0.30,ObeseHFD_D12492_meals_min),0.60,ObeseHFD_D12492_meals_max-ObeseHFD_D12492_meals_min,color=colors_meals[6],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((8-0.30,ObeseHFD_Sema2_meals_min),0.60,ObeseHFD_Sema2_meals_max-ObeseHFD_Sema2_meals_min,color=colors_meals[8],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((9-0.30,ObeseHFD_Sema10_meals_min),0.60,ObeseHFD_Sema10_meals_max-ObeseHFD_Sema10_meals_min,color=colors_meals[7],alpha=0.18,lw=0))
    axes[1].add_patch(plt.Rectangle((1-0.16,0),0.32,best_HealthyChow_D12450J_meals,facecolor=colors_meals[0],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((2-0.16,0),0.32,best_HealthyHFD_D12492_meals,facecolor=colors_meals[1],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((3-0.16,0),0.32,best_HealthyHFD_D12331_meals,facecolor=colors_meals[2],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((4-0.16,0),0.32,best_HealthyControl_D124512_meals,facecolor=colors_meals[3],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((5-0.16,0),0.32,best_HealthyControl_TekladLM_485_meals,facecolor=colors_meals[4],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((6-0.16,0),0.32,best_ObeseChow_D12450J_meals,facecolor=colors_meals[5],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((7-0.16,0),0.32,best_ObeseHFD_D12492_meals,facecolor=colors_meals[6],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((8-0.16,0),0.32,best_ObeseHFD_Sema2_meals,facecolor=colors_meals[8],edgecolor="black"))
    axes[1].add_patch(plt.Rectangle((9-0.16,0),0.32,best_ObeseHFD_Sema10_meals,facecolor=colors_meals[7],edgecolor="black"))
    axes[1].set_xlim(0.5,9.5)
    axes[1].set_ylim(0,20)
    axes[1].set_yticks([0,5,10,15,20])
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(meal_labels,rotation=0,ha="center")
    axes[1].set_ylabel("Average meals per day\n(amount)",fontsize=18)
    axes[1].grid(True,axis="y")
    #  COLUMN 3: Average meal size 
    axes[2].add_patch(plt.Rectangle((1-0.30,HealthyChow_D12450J_meal_size_min),0.60,HealthyChow_D12450J_meal_size_max-HealthyChow_D12450J_meal_size_min,color=colors_meals[0],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((2-0.30,HealthyHFD_D12492_meal_size_min),0.60,HealthyHFD_D12492_meal_size_max-HealthyHFD_D12492_meal_size_min,color=colors_meals[1],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((3-0.30,HealthyHFD_D12331_meal_size_min),0.60,HealthyHFD_D12331_meal_size_max-HealthyHFD_D12331_meal_size_min,color=colors_meals[2],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((4-0.30,HealthyControl_D124512_meal_size_min),0.60,HealthyControl_D124512_meal_size_max-HealthyControl_D124512_meal_size_min,color=colors_meals[3],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((5-0.30,HealthyControl_TekladLM_485_meal_size_min),0.60,HealthyControl_TekladLM_485_meal_size_max-HealthyControl_TekladLM_485_meal_size_min,color=colors_meals[4],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((6-0.30,ObeseChow_D12450J_meal_size_min),0.60,ObeseChow_D12450J_meal_size_max-ObeseChow_D12450J_meal_size_min,color=colors_meals[5],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((7-0.30,ObeseHFD_D12492_meal_size_min),0.60,ObeseHFD_D12492_meal_size_max-ObeseHFD_D12492_meal_size_min,color=colors_meals[6],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((8-0.30,ObeseHFD_Sema2_meal_size_min),0.60,ObeseHFD_Sema2_meal_size_max-ObeseHFD_Sema2_meal_size_min,color=colors_meals[8],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((9-0.30,ObeseHFD_Sema10_meal_size_min),0.60,ObeseHFD_Sema10_meal_size_max-ObeseHFD_Sema10_meal_size_min,color=colors_meals[7],alpha=0.18,lw=0))
    axes[2].add_patch(plt.Rectangle((1-0.16,0),0.32,best_HealthyChow_D12450J_meal_size,facecolor=colors_meals[0],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((2-0.16,0),0.32,best_HealthyHFD_D12492_meal_size,facecolor=colors_meals[1],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((3-0.16,0),0.32,best_HealthyHFD_D12331_meal_size,facecolor=colors_meals[2],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((4-0.16,0),0.32,best_HealthyControl_D124512_meal_size,facecolor=colors_meals[3],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((5-0.16,0),0.32,best_HealthyControl_TekladLM_485_meal_size,facecolor=colors_meals[4],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((6-0.16,0),0.32,best_ObeseChow_D12450J_meal_size,facecolor=colors_meals[5],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((7-0.16,0),0.32,best_ObeseHFD_D12492_meal_size,facecolor=colors_meals[6],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((8-0.16,0),0.32,best_ObeseHFD_Sema2_meal_size,facecolor=colors_meals[8],edgecolor="black"))
    axes[2].add_patch(plt.Rectangle((9-0.16,0),0.32,best_ObeseHFD_Sema10_meal_size,facecolor=colors_meals[7],edgecolor="black"))
    axes[2].set_xlim(0.5,9.5)
    axes[2].set_ylim(0,4)
    axes[2].set_yticks([0,1,2,3,4])
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels(meal_labels,rotation=0,ha="center")
    axes[2].set_ylabel("Average meal size\n(kcal/meal)",fontsize=18)
    axes[2].grid(True,axis="y")
    #  COLUMN 4: Meals consumed at night 
    axes[3].add_patch(plt.Rectangle((1-0.30,HealthyChow_D12450J_night_meals_min),0.60,HealthyChow_D12450J_night_meals_max-HealthyChow_D12450J_night_meals_min,color=colors_meals[0],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((2-0.30,HealthyHFD_D12492_night_meals_min),0.60,HealthyHFD_D12492_night_meals_max-HealthyHFD_D12492_night_meals_min,color=colors_meals[1],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((3-0.30,HealthyHFD_D12331_night_meals_min),0.60,HealthyHFD_D12331_night_meals_max-HealthyHFD_D12331_night_meals_min,color=colors_meals[2],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((4-0.30,HealthyControl_D124512_night_meals_min),0.60,HealthyControl_D124512_night_meals_max-HealthyControl_D124512_night_meals_min,color=colors_meals[3],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((5-0.30,HealthyControl_TekladLM_485_night_meals_min),0.60,HealthyControl_TekladLM_485_night_meals_max-HealthyControl_TekladLM_485_night_meals_min,color=colors_meals[4],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((6-0.30,ObeseChow_D12450J_night_meals_min),0.60,ObeseChow_D12450J_night_meals_max-ObeseChow_D12450J_night_meals_min,color=colors_meals[5],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((7-0.30,ObeseHFD_D12492_night_meals_min),0.60,ObeseHFD_D12492_night_meals_max-ObeseHFD_D12492_night_meals_min,color=colors_meals[6],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((8-0.30,ObeseHFD_Sema2_night_meals_min),0.60,ObeseHFD_Sema2_night_meals_max-ObeseHFD_Sema2_night_meals_min,color=colors_meals[8],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((9-0.30,ObeseHFD_Sema10_night_meals_min),0.60,ObeseHFD_Sema10_night_meals_max-ObeseHFD_Sema10_night_meals_min,color=colors_meals[7],alpha=0.18,lw=0))
    axes[3].add_patch(plt.Rectangle((1-0.16,0),0.32,best_HealthyChow_D12450J_night_meals,facecolor=colors_meals[0],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((2-0.16,0),0.32,best_HealthyHFD_D12492_night_meals,facecolor=colors_meals[1],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((3-0.16,0),0.32,best_HealthyHFD_D12331_night_meals,facecolor=colors_meals[2],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((4-0.16,0),0.32,best_HealthyControl_D124512_night_meals,facecolor=colors_meals[3],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((5-0.16,0),0.32,best_HealthyControl_TekladLM_485_night_meals,facecolor=colors_meals[4],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((6-0.16,0),0.32,best_ObeseChow_D12450J_night_meals,facecolor=colors_meals[5],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((7-0.16,0),0.32,best_ObeseHFD_D12492_night_meals,facecolor=colors_meals[6],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((8-0.16,0),0.32,best_ObeseHFD_Sema2_night_meals,facecolor=colors_meals[8],edgecolor="black"))
    axes[3].add_patch(plt.Rectangle((9-0.16,0),0.32,best_ObeseHFD_Sema10_night_meals,facecolor=colors_meals[7],edgecolor="black"))
    axes[3].set_xlim(0.5,9.5)
    axes[3].set_ylim(0,100)
    axes[3].set_yticks([0,25,50,75,100])
    axes[3].set_xticks(x_positions)
    axes[3].set_xticklabels(meal_labels,rotation=0,ha="center")
    axes[3].set_ylabel("Meals consumed at night\n(%)",fontsize=18)
    axes[3].grid(True,axis="y")
    for j in range(4):
        axes[j].tick_params(axis="y",which="major",labelsize=18)
        axes[j].tick_params(axis="y",which="minor",labelsize=18)
        axes[j].tick_params(axis="x",which="major",labelsize=12)
        axes[j].tick_params(axis="x",which="minor",labelsize=12)
    fig.subplots_adjust(left=0.07,right=0.99,bottom=0.12,top=0.95,wspace=0.25,hspace=0.45)
    plt.show()

    

def plot_PhyslicalActivityDependency(OptimisedAppetiteparams,OptimisedBWparams, OptimisedSemaglutidePKparams, OptimisedHeparams):

    BWparams = OptimisedBWparams.to_numpy(dtype=float)
    GIparams = OptimisedHeparams.to_numpy(dtype=float)
    Best_BWParams = np.asarray(BWparams[0][:-1], dtype=float)
    Best_GIParams = np.asarray(GIparams[0][:-1], dtype=float)

    # ==========================================================
    # Study covariables
    # ==========================================================
    StudyCovariables = [
        {
            "Group": 1,
            "Diet": "Hall et al. chow",
            "DietCode": "Unspecified",
            "Study": "Hall et al.",
            "StudyGroupID": 1,
            "FatRatio": 0.12,
            "CarbRatio": 0.64,
            "Lambda": Best_BWParams[-2],
            "MidStudyAge": 25.0, # weeks
            "MiceType": "C57BL/6J",
            "MiceTypeID": 1,
            "MidStudyWeight": 30.95903439 # g

        },
        {
            "Group": 2,
            "Diet": "Hall et al. high fat diet",
            "DietCode": "Unspecified",
            "Study": "Hall et al.",
            "StudyGroupID": 1,
            "FatRatio": 0.59,
            "CarbRatio": 0.27,
            "Lambda": Best_BWParams[-1],
            "MidStudyAge": 25.0,
            "MiceType": "C57BL/6J",
            "MiceTypeID": 1,
            "MidStudyWeight": 38.16289225 # g
        },
        {
            "Group": 3,
            "Diet": "Control diet",
            "DietCode": "D12450J",
            "Study": "He et al.",
            "StudyGroupID": 2,
            "FatRatio": 0.10,
            "CarbRatio": 0.70,
            "Lambda": Best_BWParams[-2]*Best_GIParams[-3],
            "MidStudyAge": 16.5,
            "MiceType": "C57BL/6J",
            "MiceTypeID": 1,
            "MidStudyWeight": 25.04130023 # g
        },
        {
            "Group": 4,
            "Diet": "High fat diet",
            "DietCode": "D12492",
            "Study": "He et al.",
            "StudyGroupID": 2,
            "FatRatio": 0.60,
            "CarbRatio": 0.20,
            "Lambda": Best_BWParams[-1]*Best_GIParams[-2],
            "MidStudyAge": 16.5,
            "MiceType": "C57BL/6J",
            "MiceTypeID": 1,
            "MidStudyWeight": 28.73473312 # g
        },
        {
            "Group": 5,
            "Diet": "High fat diet",
            "DietCode": "D12331",
            "Study": "Mack et al.",
            "StudyGroupID": 3,
            "FatRatio": 0.58,
            "CarbRatio": 0.26,
            "Lambda": 0.195,
            "MidStudyAge": 8.0,
            "MiceType": "C57BL/6J",
            "MiceTypeID": 1,
            "MidStudyWeight": 26.30937082 # g
        },
        {
            "Group": 6,
            "Diet": "High fat diet",
            "DietCode": "D124921",
            "Study": "Internal AZ study",
            "StudyGroupID": 3,
            "FatRatio": 0.60,
            "CarbRatio": 0.20,
            "Lambda": 0.085,
            "MidStudyAge": 43.5,
            "MiceType": "DIO mice",
            "MiceTypeID": 2,
            "MidStudyWeight": 49.20091189 # g
        },
        {
            "Group": 7,
            "Diet": "Control diet",
            "DietCode": "D124512",
            "Study": "Gabery et al.",
            "StudyGroupID": 3,
            "FatRatio": 0.45,
            "CarbRatio": 0.35,
            "Lambda": 0.12,
            "MidStudyAge": 25.5,
            "MiceType": "DIO mice",
            "MiceTypeID": 2,
            "MidStudyWeight": 38.70434755 # g
        }
    ]

    StudyCovariables_df = pd.DataFrame(StudyCovariables)
    DietLabels = StudyCovariables_df["Diet"].to_list()
    DietCodes = StudyCovariables_df["DietCode"].to_list()
    StudyLabels = StudyCovariables_df["Study"].to_list()
    StudyGroupIDs = StudyCovariables_df["StudyGroupID"].to_numpy(dtype=int)
    FatRatios = StudyCovariables_df["FatRatio"].to_numpy(dtype=float)
    CarbRatios = StudyCovariables_df["CarbRatio"].to_numpy(dtype=float)
    LambdaValues = StudyCovariables_df["Lambda"].to_numpy(dtype=float)
    MidStudyAge = StudyCovariables_df["MidStudyAge"].to_numpy(dtype=float)
    MiceTypeIDs = StudyCovariables_df["MiceTypeID"].to_numpy(dtype=int)
    MidStudyWeight = StudyCovariables_df["MidStudyWeight"].to_numpy(dtype=float)
    print(StudyCovariables_df.to_string(index=False))

    # ==========================================================
    # LINEAR REGRESSION ANALYSIS
    # ==========================================================
    RegressionVariables = [
        ("StudyGroupID",StudyGroupIDs),
        ("FatRatio",FatRatios),
        ("CarbRatio",CarbRatios),
        ("MidStudyAge",MidStudyAge),
        ("MiceTypeID",MiceTypeIDs),
        ("MidStudyWeight",MidStudyWeight)
    ]
    RegressionResults = {}

    Y = np.asarray(LambdaValues,dtype=float)
    for VariableName,X in RegressionVariables:

        X = np.asarray(X,dtype=float)
        linear_slope,linear_intercept = np.polyfit(X,Y,1)
        LambdaPredicted = linear_slope*X + linear_intercept
        SS_residual = np.sum((Y - LambdaPredicted)**2)
        SS_total = np.sum((Y - np.mean(Y))**2)
        R_squared = 1 - SS_residual/SS_total
        X_regression = np.linspace(np.min(X),np.max(X),200)
        LambdaValues_regression = linear_slope*X_regression + linear_intercept
        RegressionResults[VariableName] = {
            "X": X,
            "Slope": linear_slope,
            "Intercept": linear_intercept,
            "R_squared": R_squared,
            "X_regression": X_regression,
            "LambdaValues_regression": LambdaValues_regression
        }

        print(f"{VariableName}:")
        print(f"lambda = {linear_slope:.6f}*{VariableName} {linear_intercept:+.6f}")
        print(f"R^2 = {R_squared:.6f}")
        print("")


    # ==========================================================
    # CATEGORICAL STUDY GROUP ANALYSIS
    # ==========================================================
    HallGroup = (StudyGroupIDs == 1).astype(float)
    HeGroup = (StudyGroupIDs == 2).astype(float)
    SingleGroupStudies = (StudyGroupIDs == 3).astype(float)

    X_studygroup = np.column_stack((
        HeGroup,
        SingleGroupStudies,
        np.ones(len(LambdaValues))
    ))

    beta_studygroup = np.linalg.lstsq(X_studygroup,LambdaValues,rcond=None)[0]
    Effect_He,Effect_SingleGroup,Intercept_Hall = beta_studygroup
    LambdaPredicted_studygroup = X_studygroup @ beta_studygroup
    SS_residual_studygroup = np.sum((LambdaValues-LambdaPredicted_studygroup)**2)
    SS_total_studygroup = np.sum((LambdaValues-np.mean(LambdaValues))**2)
    R_squared_studygroup = 1-SS_residual_studygroup/SS_total_studygroup
    Lambda_Hall = Intercept_Hall
    Lambda_He = Intercept_Hall + Effect_He
    Lambda_SingleGroup = Intercept_Hall + Effect_SingleGroup

    print("Categorical study-group analysis:")
    print(f"Hall lambda = {Lambda_Hall:.6f}")
    print(f"He effect = {Effect_He:+.6f}")
    print(f"Single-group effect = {Effect_SingleGroup:+.6f}")
    print(f"He lambda = {Lambda_He:.6f}")
    print(f"Single-group lambda = {Lambda_SingleGroup:.6f}")
    print(f"R^2 = {R_squared_studygroup:.6f}")
    print("")

    # ==========================================================
    # FIGURE: UNIVARIATE LINEAR REGRESSIONS
    # ==========================================================
    DietColors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
    fig4,axes4 = plt.subplots(2,3,figsize=(24,14))
    #  Study group: dummy-variable regression 
    StudyGroupPlotPositions = np.where(StudyGroupIDs==2,1,np.where(StudyGroupIDs==1,2,3))
    for i in range(len(StudyGroupIDs)):
        axes4[0,0].scatter(StudyGroupPlotPositions[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,zorder=3)
    axes4[0,0].hlines(Lambda_He,0.80,1.20,color="black",lw=3,linestyle="--")
    axes4[0,0].hlines(Lambda_Hall,1.80,2.20,color="black",lw=3,linestyle="--")
    axes4[0,0].hlines(Lambda_SingleGroup,2.80,3.20,color="black",lw=3,linestyle="--")
    axes4[0,0].text(0.03,0.03,f"$\\lambda={Intercept_Hall:.3f}{Effect_He:+.3f}\\cdot He{Effect_SingleGroup:+.3f}\\cdot SingleGroup$\n$R^2={R_squared_studygroup:.3f}$",transform=axes4[0,0].transAxes,fontsize=15,verticalalignment="bottom",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    axes4[0,0].set_xlim(0.8,3.2)
    axes4[0,0].set_xticks([1,2,3])
    axes4[0,0].set_xticklabels(["He et al.","Guo & Hall","Single-group\nstudies"])
    axes4[0,0].set_ylim(0.05,0.30)
    axes4[0,0].set_ylabel("Lambda\n(kcal/g/day)",fontsize=18)
    axes4[0,0].set_title("Study-group effects on λ",fontsize=20)
    axes4[0,0].grid(True)
    #  Fat ratio 
    for i in range(len(FatRatios)):
        axes4[0,1].scatter(FatRatios[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,zorder=3)
    axes4[0,1].plot(RegressionResults["FatRatio"]["X_regression"],RegressionResults["FatRatio"]["LambdaValues_regression"],color="black",lw=3,linestyle="--")
    axes4[0,1].text(0.03,0.03,f"$\\lambda={RegressionResults['FatRatio']['Slope']:.3f}\\cdot FatRatio{RegressionResults['FatRatio']['Intercept']:+.3f}$\n$R^2={RegressionResults['FatRatio']['R_squared']:.3f}$",transform=axes4[0,1].transAxes,fontsize=15,verticalalignment="bottom",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    axes4[0,1].set_xlim(0.05,0.65)
    axes4[0,1].set_ylim(0.05,0.30)
    axes4[0,1].set_xlabel("Dietary fat %kcal",fontsize=18)
    axes4[0,1].set_ylabel("Lambda\n(kcal/g/day)",fontsize=18)
    axes4[0,1].set_title("Fat %kcal versus λ",fontsize=20)
    axes4[0,1].grid(True)
    #  Carbohydrate ratio 
    for i in range(len(CarbRatios)):
        axes4[0,2].scatter(CarbRatios[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,zorder=3)
    axes4[0,2].plot(RegressionResults["CarbRatio"]["X_regression"],RegressionResults["CarbRatio"]["LambdaValues_regression"],color="black",lw=3,linestyle="--")
    axes4[0,2].text(0.97,0.03,f"$\\lambda={RegressionResults['CarbRatio']['Slope']:.3f}\\cdot CarbRatio{RegressionResults['CarbRatio']['Intercept']:+.3f}$\n$R^2={RegressionResults['CarbRatio']['R_squared']:.3f}$",transform=axes4[0,2].transAxes,fontsize=15,verticalalignment="bottom",horizontalalignment="right",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    axes4[0,2].set_xlim(0.15,0.75)
    axes4[0,2].set_ylim(0.05,0.30)
    axes4[0,2].set_xlabel("Dietary carbohydrate %kcal",fontsize=18)
    axes4[0,2].set_ylabel("Lambda\n(kcal/g/day)",fontsize=18)
    axes4[0,2].set_title("Carbohydrate %kcal versus λ",fontsize=20)
    axes4[0,2].grid(True)
    #  Mid-study age 
    for i in range(len(MidStudyAge)):
        axes4[1,0].scatter(MidStudyAge[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,zorder=3)
    axes4[1,0].plot(RegressionResults["MidStudyAge"]["X_regression"],RegressionResults["MidStudyAge"]["LambdaValues_regression"],color="black",lw=3,linestyle="--")
    axes4[1,0].text(0.97,0.97,f"$\\lambda={RegressionResults['MidStudyAge']['Slope']:.3f}\\cdot Age{RegressionResults['MidStudyAge']['Intercept']:+.3f}$\n$R^2={RegressionResults['MidStudyAge']['R_squared']:.3f}$",transform=axes4[1,0].transAxes,fontsize=15,verticalalignment="top",horizontalalignment="right",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    axes4[1,0].set_xlim(5,46)
    axes4[1,0].set_ylim(0.05,0.30)
    axes4[1,0].set_xlabel("Mid-study mouse age (weeks)",fontsize=18)
    axes4[1,0].set_ylabel("Lambda\n(kcal/g/day)",fontsize=18)
    axes4[1,0].set_title("Mouse age versus λ",fontsize=20)
    axes4[1,0].grid(True)
    #  Mouse type 
    for i in range(len(MiceTypeIDs)):
        axes4[1,1].scatter(MiceTypeIDs[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,zorder=3)
    axes4[1,1].plot(RegressionResults["MiceTypeID"]["X_regression"],RegressionResults["MiceTypeID"]["LambdaValues_regression"],color="black",lw=3,linestyle="--")
    axes4[1,1].text(0.97,0.97,f"$\\lambda={RegressionResults['MiceTypeID']['Slope']:.3f}\\cdot MiceTypeID{RegressionResults['MiceTypeID']['Intercept']:+.3f}$\n$R^2={RegressionResults['MiceTypeID']['R_squared']:.3f}$",transform=axes4[1,1].transAxes,fontsize=15,verticalalignment="top",horizontalalignment="right",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    axes4[1,1].set_xlim(0.8,2.2)
    axes4[1,1].set_xticks([1,2])
    axes4[1,1].set_xticklabels(["Healthy","DIO"])
    axes4[1,1].set_ylim(0.05,0.30)
    axes4[1,1].set_ylabel("Lambda\n(kcal/g/day)",fontsize=18)
    axes4[1,1].set_title("Disease status versus λ",fontsize=20)
    axes4[1,1].grid(True)
    #  Mid-study body weight 
    for i in range(len(MidStudyWeight)):
        axes4[1,2].scatter(MidStudyWeight[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,zorder=3)
    axes4[1,2].plot(RegressionResults["MidStudyWeight"]["X_regression"],RegressionResults["MidStudyWeight"]["LambdaValues_regression"],color="black",lw=3,linestyle="--")
    axes4[1,2].text(0.97,0.97,f"$\\lambda={RegressionResults['MidStudyWeight']['Slope']:.3f}\\cdot Weight{RegressionResults['MidStudyWeight']['Intercept']:+.3f}$\n$R^2={RegressionResults['MidStudyWeight']['R_squared']:.3f}$",transform=axes4[1,2].transAxes,fontsize=15,verticalalignment="top",horizontalalignment="right",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    axes4[1,2].set_xlim(20,55)
    axes4[1,2].set_ylim(0.05,0.30)
    axes4[1,2].set_xlabel("Mid-study body weight (g)",fontsize=18)
    axes4[1,2].set_ylabel("Lambda\n(kcal/g/day)",fontsize=18)
    axes4[1,2].set_title("Total body mass versus λ",fontsize=20)
    axes4[1,2].grid(True)
    #  Formatting 
    for ax in axes4.flat:
        ax.tick_params(axis="both",which="major",labelsize=16)
    legend_handles_fig4 = [plt.Line2D([0],[0],marker="o",color="none",markerfacecolor=DietColors[i],markeredgecolor="black",markeredgewidth=1.5,markersize=13,label=f"{StudyLabels[i]}: {DietCodes[i]}") for i in range(len(DietLabels))]
    legend_handles_fig4.append(plt.Line2D([0],[0],color="black",lw=3,linestyle="--",label="Function output"))
    fig4.legend(handles=legend_handles_fig4,loc="lower center",bbox_to_anchor=(0.5,0.01),ncol=4,fontsize=14,framealpha=1)
    fig4.subplots_adjust(left=0.07,right=0.98,bottom=0.18,top=0.95,wspace=0.35,hspace=0.40)
    plt.show()


    # ==========================================================
    # MULTIVARIABLE LINEAR REGRESSION ANALYSIS
    # ==========================================================
    LambdaValues = np.asarray(LambdaValues,dtype=float)
    CarbRatios = np.asarray(CarbRatios,dtype=float)
    FatRatios = np.asarray(FatRatios,dtype=float)
    MidStudyAge = np.asarray(MidStudyAge,dtype=float)
    MiceTypeIDs = np.asarray(MiceTypeIDs,dtype=float)
    MidStudyWeight = np.asarray(MidStudyWeight,dtype=float)
    SS_total = np.sum((LambdaValues-np.mean(LambdaValues))**2)
    #  Carbohydrate ratio and mid-study age 
    X_carb_age = np.column_stack((CarbRatios,MidStudyAge,np.ones(len(LambdaValues))))
    beta_carb_age = np.linalg.lstsq(X_carb_age,LambdaValues,rcond=None)[0]
    slope_carb_carb_age,slope_age_carb_age,intercept_carb_age = beta_carb_age
    LambdaPredicted_carb_age = X_carb_age@beta_carb_age
    R_squared_carb_age = 1-np.sum((LambdaValues-LambdaPredicted_carb_age)**2)/SS_total
    CarbGrid_carb_age = np.linspace(np.min(CarbRatios),np.max(CarbRatios),50)
    AgeGrid_carb_age = np.linspace(np.min(MidStudyAge),np.max(MidStudyAge),50)
    CarbMesh_carb_age,AgeMesh_carb_age = np.meshgrid(CarbGrid_carb_age,AgeGrid_carb_age)
    LambdaMesh_carb_age = slope_carb_carb_age*CarbMesh_carb_age+slope_age_carb_age*AgeMesh_carb_age+intercept_carb_age
    #  Carbohydrate ratio and mouse type 
    X_carb_type = np.column_stack((CarbRatios,MiceTypeIDs,np.ones(len(LambdaValues))))
    beta_carb_type = np.linalg.lstsq(X_carb_type,LambdaValues,rcond=None)[0]
    slope_carb_carb_type,slope_type_carb_type,intercept_carb_type = beta_carb_type
    LambdaPredicted_carb_type = X_carb_type@beta_carb_type
    R_squared_carb_type = 1-np.sum((LambdaValues-LambdaPredicted_carb_type)**2)/SS_total
    CarbGrid_carb_type = np.linspace(np.min(CarbRatios),np.max(CarbRatios),50)
    TypeGrid_carb_type = np.linspace(1,2,50)
    CarbMesh_carb_type,TypeMesh_carb_type = np.meshgrid(CarbGrid_carb_type,TypeGrid_carb_type)
    LambdaMesh_carb_type = slope_carb_carb_type*CarbMesh_carb_type+slope_type_carb_type*TypeMesh_carb_type+intercept_carb_type
    #  Mid-study age and mouse type 
    X_age_type = np.column_stack((MidStudyAge,MiceTypeIDs,np.ones(len(LambdaValues))))
    beta_age_type = np.linalg.lstsq(X_age_type,LambdaValues,rcond=None)[0]
    slope_age_age_type,slope_type_age_type,intercept_age_type = beta_age_type
    LambdaPredicted_age_type = X_age_type@beta_age_type
    R_squared_age_type = 1-np.sum((LambdaValues-LambdaPredicted_age_type)**2)/SS_total
    AgeGrid_age_type = np.linspace(np.min(MidStudyAge),np.max(MidStudyAge),50)
    TypeGrid_age_type = np.linspace(1,2,50)
    AgeMesh_age_type,TypeMesh_age_type = np.meshgrid(AgeGrid_age_type,TypeGrid_age_type)
    LambdaMesh_age_type = slope_age_age_type*AgeMesh_age_type+slope_type_age_type*TypeMesh_age_type+intercept_age_type
    #  Fat ratio and mid-study age 
    X_fat_age = np.column_stack((FatRatios,MidStudyAge,np.ones(len(LambdaValues))))
    beta_fat_age = np.linalg.lstsq(X_fat_age,LambdaValues,rcond=None)[0]
    slope_fat_fat_age,slope_age_fat_age,intercept_fat_age = beta_fat_age
    LambdaPredicted_fat_age = X_fat_age@beta_fat_age
    R_squared_fat_age = 1-np.sum((LambdaValues-LambdaPredicted_fat_age)**2)/SS_total
    FatGrid_fat_age = np.linspace(np.min(FatRatios),np.max(FatRatios),50)
    AgeGrid_fat_age = np.linspace(np.min(MidStudyAge),np.max(MidStudyAge),50)
    FatMesh_fat_age,AgeMesh_fat_age = np.meshgrid(FatGrid_fat_age,AgeGrid_fat_age)
    LambdaMesh_fat_age = slope_fat_fat_age*FatMesh_fat_age+slope_age_fat_age*AgeMesh_fat_age+intercept_fat_age
    #  Fat ratio and mouse type 
    X_fat_type = np.column_stack((FatRatios,MiceTypeIDs,np.ones(len(LambdaValues))))
    beta_fat_type = np.linalg.lstsq(X_fat_type,LambdaValues,rcond=None)[0]
    slope_fat_fat_type,slope_type_fat_type,intercept_fat_type = beta_fat_type
    LambdaPredicted_fat_type = X_fat_type@beta_fat_type
    R_squared_fat_type = 1-np.sum((LambdaValues-LambdaPredicted_fat_type)**2)/SS_total
    FatGrid_fat_type = np.linspace(np.min(FatRatios),np.max(FatRatios),50)
    TypeGrid_fat_type = np.linspace(1,2,50)
    FatMesh_fat_type,TypeMesh_fat_type = np.meshgrid(FatGrid_fat_type,TypeGrid_fat_type)
    LambdaMesh_fat_type = slope_fat_fat_type*FatMesh_fat_type+slope_type_fat_type*TypeMesh_fat_type+intercept_fat_type
    #  Mid-study body weight and mid-study age 
    X_weight_age = np.column_stack((MidStudyWeight,MidStudyAge,np.ones(len(LambdaValues))))
    beta_weight_age = np.linalg.lstsq(X_weight_age,LambdaValues,rcond=None)[0]
    slope_weight_weight_age,slope_age_weight_age,intercept_weight_age = beta_weight_age
    LambdaPredicted_weight_age = X_weight_age@beta_weight_age
    R_squared_weight_age = 1-np.sum((LambdaValues-LambdaPredicted_weight_age)**2)/SS_total
    WeightGrid_weight_age = np.linspace(np.min(MidStudyWeight),np.max(MidStudyWeight),50)
    AgeGrid_weight_age = np.linspace(np.min(MidStudyAge),np.max(MidStudyAge),50)
    WeightMesh_weight_age,AgeMesh_weight_age = np.meshgrid(WeightGrid_weight_age,AgeGrid_weight_age)
    LambdaMesh_weight_age = slope_weight_weight_age*WeightMesh_weight_age+slope_age_weight_age*AgeMesh_weight_age+intercept_weight_age
    #  Carbohydrate ratio and mid-study body weight 
    X_carb_weight = np.column_stack((CarbRatios,MidStudyWeight,np.ones(len(LambdaValues))))
    beta_carb_weight = np.linalg.lstsq(X_carb_weight,LambdaValues,rcond=None)[0]
    slope_carb_carb_weight,slope_weight_carb_weight,intercept_carb_weight = beta_carb_weight
    LambdaPredicted_carb_weight = X_carb_weight@beta_carb_weight
    R_squared_carb_weight = 1-np.sum((LambdaValues-LambdaPredicted_carb_weight)**2)/SS_total
    CarbGrid_carb_weight = np.linspace(np.min(CarbRatios),np.max(CarbRatios),50)
    WeightGrid_carb_weight = np.linspace(np.min(MidStudyWeight),np.max(MidStudyWeight),50)
    CarbMesh_carb_weight,WeightMesh_carb_weight = np.meshgrid(CarbGrid_carb_weight,WeightGrid_carb_weight)
    LambdaMesh_carb_weight = slope_carb_carb_weight*CarbMesh_carb_weight+slope_weight_carb_weight*WeightMesh_carb_weight+intercept_carb_weight
    #  Fat ratio and mid-study body weight 
    X_fat_weight = np.column_stack((FatRatios,MidStudyWeight,np.ones(len(LambdaValues))))
    beta_fat_weight = np.linalg.lstsq(X_fat_weight,LambdaValues,rcond=None)[0]
    slope_fat_fat_weight,slope_weight_fat_weight,intercept_fat_weight = beta_fat_weight
    LambdaPredicted_fat_weight = X_fat_weight@beta_fat_weight
    R_squared_fat_weight = 1-np.sum((LambdaValues-LambdaPredicted_fat_weight)**2)/SS_total
    FatGrid_fat_weight = np.linspace(np.min(FatRatios),np.max(FatRatios),50)
    WeightGrid_fat_weight = np.linspace(np.min(MidStudyWeight),np.max(MidStudyWeight),50)
    FatMesh_fat_weight,WeightMesh_fat_weight = np.meshgrid(FatGrid_fat_weight,WeightGrid_fat_weight)
    LambdaMesh_fat_weight = slope_fat_fat_weight*FatMesh_fat_weight+slope_weight_fat_weight*WeightMesh_fat_weight+intercept_fat_weight

    print("Carbohydrate ratio and mid-study age:")
    print(f"lambda = {slope_carb_carb_age:.6f}*CarbRatio {slope_age_carb_age:+.6f}*MidStudyAge {intercept_carb_age:+.6f}")
    print(f"R^2 = {R_squared_carb_age:.6f}")
    print("")
    print("Carbohydrate ratio and mouse type:")
    print(f"lambda = {slope_carb_carb_type:.6f}*CarbRatio {slope_type_carb_type:+.6f}*MiceTypeID {intercept_carb_type:+.6f}")
    print(f"R^2 = {R_squared_carb_type:.6f}")
    print("")
    print("Mid-study age and mouse type:")
    print(f"lambda = {slope_age_age_type:.6f}*MidStudyAge {slope_type_age_type:+.6f}*MiceTypeID {intercept_age_type:+.6f}")
    print(f"R^2 = {R_squared_age_type:.6f}")
    print("")
    print("Fat ratio and mid-study age:")
    print(f"lambda = {slope_fat_fat_age:.6f}*FatRatio {slope_age_fat_age:+.6f}*MidStudyAge {intercept_fat_age:+.6f}")
    print(f"R^2 = {R_squared_fat_age:.6f}")
    print("")
    print("Fat ratio and mouse type:")
    print(f"lambda = {slope_fat_fat_type:.6f}*FatRatio {slope_type_fat_type:+.6f}*MiceTypeID {intercept_fat_type:+.6f}")
    print(f"R^2 = {R_squared_fat_type:.6f}")
    print("")
    print("Mid-study body weight and mid-study age:")
    print(f"lambda = {slope_weight_weight_age:.6f}*MidStudyWeight {slope_age_weight_age:+.6f}*MidStudyAge {intercept_weight_age:+.6f}")
    print(f"R^2 = {R_squared_weight_age:.6f}")
    print("")
    print("Carbohydrate ratio and mid-study body weight:")
    print(f"lambda = {slope_carb_carb_weight:.6f}*CarbRatio {slope_weight_carb_weight:+.6f}*MidStudyWeight {intercept_carb_weight:+.6f}")
    print(f"R^2 = {R_squared_carb_weight:.6f}")
    print("")
    print("Fat ratio and mid-study body weight:")
    print(f"lambda = {slope_fat_fat_weight:.6f}*FatRatio {slope_weight_fat_weight:+.6f}*MidStudyWeight {intercept_fat_weight:+.6f}")
    print(f"R^2 = {R_squared_fat_weight:.6f}")
    print("")

    # ==========================================================
    # FIGURE: MULTIVARIABLE LINEAR REGRESSIONS
    # ==========================================================
    fig5 = plt.figure(figsize=(27,18))
    ax5_carb_age = fig5.add_subplot(2,3,1,projection="3d")
    ax5_carb_type = fig5.add_subplot(2,3,2,projection="3d")
    ax5_age_type = fig5.add_subplot(2,3,3,projection="3d")
    ax5_fat_age = fig5.add_subplot(2,3,4,projection="3d")
    ax5_fat_type = fig5.add_subplot(2,3,5,projection="3d")
    ax5_legend = fig5.add_subplot(2,3,6)
    #  Carbohydrate ratio and mid-study age 
    ax5_carb_age.plot_surface(CarbMesh_carb_age,AgeMesh_carb_age,LambdaMesh_carb_age,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax5_carb_age.scatter(CarbRatios[i],MidStudyAge[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax5_carb_age.text2D(0.02,0.97,f"$\\lambda={slope_carb_carb_age:.3f}\\cdot CarbRatio{slope_age_carb_age:+.3f}\\cdot Age{intercept_carb_age:+.3f}$\n$R^2={R_squared_carb_age:.3f}$",transform=ax5_carb_age.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax5_carb_age.set_xlim(np.min(CarbRatios)-0.05,np.max(CarbRatios)+0.05)
    ax5_carb_age.set_ylim(np.min(MidStudyAge)-2,np.max(MidStudyAge)+2)
    ax5_carb_age.set_zlim(0.05,0.30)
    ax5_carb_age.set_zticks([0.05,0.15,0.25])
    ax5_carb_age.set_xticks([0.2,0.4,0.6])
    ax5_carb_age.set_yticks([10,25,40])
    ax5_carb_age.set_xlabel("Dietary carbohydrates\n(%kcal)",fontsize=16,labelpad=14)
    ax5_carb_age.set_ylabel("Mid-study age\n(weeks)",fontsize=16,labelpad=14)
    ax5_carb_age.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax5_carb_age.tick_params(axis="both",which="major",labelsize=14)
    ax5_carb_age.view_init(elev=25,azim=-55)
    #  Carbohydrate ratio and mouse type 
    ax5_carb_type.plot_surface(CarbMesh_carb_type,TypeMesh_carb_type,LambdaMesh_carb_type,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax5_carb_type.scatter(CarbRatios[i],MiceTypeIDs[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax5_carb_type.text2D(0.02,0.97,f"$\\lambda={slope_carb_carb_type:.3f}\\cdot CarbRatio{slope_type_carb_type:+.3f}\\cdot MiceTypeID{intercept_carb_type:+.3f}$\n$R^2={R_squared_carb_type:.3f}$",transform=ax5_carb_type.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax5_carb_type.set_xlim(np.min(CarbRatios)-0.05,np.max(CarbRatios)+0.05)
    ax5_carb_type.set_ylim(0.8,2.2)
    ax5_carb_type.set_zlim(0.05,0.30)
    ax5_carb_type.set_zticks([0.05,0.15,0.25])
    ax5_carb_type.set_xticks([0.2,0.4,0.6])
    ax5_carb_type.set_yticks([1,2])
    ax5_carb_type.set_yticklabels(["Healthy","DIO"])
    ax5_carb_type.set_xlabel("Dietary carbohydrates\n(%kcal)",fontsize=16,labelpad=14)
    ax5_carb_type.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax5_carb_type.tick_params(axis="both",which="major",labelsize=14)
    ax5_carb_type.view_init(elev=25,azim=-55)
    #  Mid-study age and mouse type 
    ax5_age_type.plot_surface(AgeMesh_age_type,TypeMesh_age_type,LambdaMesh_age_type,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax5_age_type.scatter(MidStudyAge[i],MiceTypeIDs[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax5_age_type.text2D(0.02,0.97,f"$\\lambda={slope_age_age_type:.3f}\\cdot Age{slope_type_age_type:+.3f}\\cdot MiceTypeID{intercept_age_type:+.3f}$\n$R^2={R_squared_age_type:.3f}$",transform=ax5_age_type.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax5_age_type.set_xlim(np.min(MidStudyAge)-2,np.max(MidStudyAge)+2)
    ax5_age_type.set_ylim(0.8,2.2)
    ax5_age_type.set_zlim(0.05,0.30)
    ax5_age_type.set_yticks([1,2])
    ax5_age_type.set_zticks([0.05,0.15,0.25])
    ax5_age_type.set_xticks([10,25,40])
    ax5_age_type.set_yticklabels(["Healthy","DIO"])
    ax5_age_type.set_xlabel("Mid-study age\n(weeks)",fontsize=16,labelpad=14)
    ax5_age_type.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax5_age_type.tick_params(axis="both",which="major",labelsize=14)
    ax5_age_type.view_init(elev=25,azim=-55)
    #  Fat ratio and mid-study age 
    ax5_fat_age.plot_surface(FatMesh_fat_age,AgeMesh_fat_age,LambdaMesh_fat_age,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax5_fat_age.scatter(FatRatios[i],MidStudyAge[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax5_fat_age.text2D(0.02,0.97,f"$\\lambda={slope_fat_fat_age:.3f}\\cdot FatRatio{slope_age_fat_age:+.3f}\\cdot Age{intercept_fat_age:+.3f}$\n$R^2={R_squared_fat_age:.3f}$",transform=ax5_fat_age.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax5_fat_age.set_xlim(np.min(FatRatios)-0.05,np.max(FatRatios)+0.05)
    ax5_fat_age.set_ylim(np.min(MidStudyAge)-2,np.max(MidStudyAge)+2)
    ax5_fat_age.set_zlim(0.05,0.30)
    ax5_fat_age.set_zticks([0.05,0.15,0.25])
    ax5_fat_age.set_xticks([0.1,0.4,0.6])
    ax5_fat_age.set_yticks([10,25,40])
    ax5_fat_age.set_xlabel("Dietary fats\n(%kcal)",fontsize=16,labelpad=14)
    ax5_fat_age.set_ylabel("Mid-study age\n(weeks)",fontsize=16,labelpad=14)
    ax5_fat_age.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax5_fat_age.tick_params(axis="both",which="major",labelsize=14)
    ax5_fat_age.view_init(elev=25,azim=-55)
    #  Fat ratio and mouse type 
    ax5_fat_type.plot_surface(FatMesh_fat_type,TypeMesh_fat_type,LambdaMesh_fat_type,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax5_fat_type.scatter(FatRatios[i],MiceTypeIDs[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax5_fat_type.text2D(0.02,0.97,f"$\\lambda={slope_fat_fat_type:.3f}\\cdot FatRatio{slope_type_fat_type:+.3f}\\cdot MiceTypeID{intercept_fat_type:+.3f}$\n$R^2={R_squared_fat_type:.3f}$",transform=ax5_fat_type.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax5_fat_type.set_xlim(np.min(FatRatios)-0.05,np.max(FatRatios)+0.05)
    ax5_fat_type.set_ylim(0.8,2.2)
    ax5_fat_type.set_zlim(0.05,0.30)
    ax5_fat_type.set_zticks([0.05,0.15,0.25])
    ax5_fat_type.set_xticks([0.2,0.4,0.6])
    ax5_fat_type.set_yticks([1,2])
    ax5_fat_type.set_yticklabels(["Healthy","DIO"])
    ax5_fat_type.set_xlabel("Dietary fats\n(%kcal)",fontsize=16,labelpad=14)
    ax5_fat_type.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax5_fat_type.tick_params(axis="both",which="major",labelsize=14)
    ax5_fat_type.view_init(elev=25,azim=-55)
    #  Legend 
    ax5_legend.axis("off")
    legend_handles_fig5 = [plt.Line2D([0],[0],marker="o",color="none",markerfacecolor=DietColors[i],markeredgecolor="black",markeredgewidth=1.5,markersize=13,label=f"{StudyLabels[i]}: {DietCodes[i]}") for i in range(len(DietLabels))]
    legend_handles_fig5.append(plt.Line2D([0],[0],color="blue",lw=8,alpha=0.25,label="Multivariable regression surface"))
    ax5_legend.legend(handles=legend_handles_fig5,loc="center",fontsize=15,framealpha=1)
    fig5.subplots_adjust(left=0.02,right=0.98,bottom=0.05,top=0.94,wspace=0.08,hspace=0.12)
    plt.show()



    # ==========================================================
    # FIGURE: MULTIVARIABLE LINEAR REGRESSIONS
    # ==========================================================
    fig6 = plt.figure(figsize=(27,18))
    ax6_weight_age = fig6.add_subplot(2,3,1,projection="3d")
    ax6_carb_weight = fig6.add_subplot(2,3,2,projection="3d")
    ax6_fat_weight = fig6.add_subplot(2,3,3,projection="3d")
    ax6_empty1 = fig6.add_subplot(2,3,4)
    ax6_empty2 = fig6.add_subplot(2,3,5)
    ax6_empty3 = fig6.add_subplot(2,3,6)
    #  Mid-study body weight and mid-study age 
    ax6_weight_age.plot_surface(WeightMesh_weight_age,AgeMesh_weight_age,LambdaMesh_weight_age,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax6_weight_age.scatter(MidStudyWeight[i],MidStudyAge[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax6_weight_age.text2D(0.02,0.97,f"$\\lambda={slope_weight_weight_age:.3f}\\cdot Weight{slope_age_weight_age:+.3f}\\cdot Age{intercept_weight_age:+.3f}$\n$R^2={R_squared_weight_age:.3f}$",transform=ax6_weight_age.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax6_weight_age.set_xlim(np.min(MidStudyWeight)-2,np.max(MidStudyWeight)+2)
    ax6_weight_age.set_ylim(np.min(MidStudyAge)-2,np.max(MidStudyAge)+2)
    ax6_weight_age.set_zlim(0.05,0.30)
    ax6_weight_age.set_zticks([0.05,0.15,0.25])
    ax6_weight_age.set_xticks([25,35,45])
    ax6_weight_age.set_yticks([10,25,40])
    ax6_weight_age.set_xlabel("Mid-study body weight\n(g)",fontsize=16,labelpad=14)
    ax6_weight_age.set_ylabel("Mid-study age\n(weeks)",fontsize=16,labelpad=14)
    ax6_weight_age.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax6_weight_age.tick_params(axis="both",which="major",labelsize=14)
    ax6_weight_age.view_init(elev=25,azim=-55)
    #  Carbohydrate ratio and mid-study body weight 
    ax6_carb_weight.plot_surface(CarbMesh_carb_weight,WeightMesh_carb_weight,LambdaMesh_carb_weight,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax6_carb_weight.scatter(CarbRatios[i],MidStudyWeight[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax6_carb_weight.text2D(0.02,0.97,f"$\\lambda={slope_carb_carb_weight:.3f}\\cdot CarbRatio{slope_weight_carb_weight:+.3f}\\cdot Weight{intercept_carb_weight:+.3f}$\n$R^2={R_squared_carb_weight:.3f}$",transform=ax6_carb_weight.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax6_carb_weight.set_xlim(np.min(CarbRatios)-0.05,np.max(CarbRatios)+0.05)
    ax6_carb_weight.set_ylim(np.min(MidStudyWeight)-2,np.max(MidStudyWeight)+2)
    ax6_carb_weight.set_zlim(0.05,0.30)
    ax6_carb_weight.set_zticks([0.05,0.15,0.25])
    ax6_carb_weight.set_xticks([0.2,0.4,0.6])
    ax6_carb_weight.set_yticks([25,35,45])
    ax6_carb_weight.set_xlabel("Dietary carbohydrates\n(%kcal)",fontsize=16,labelpad=14)
    ax6_carb_weight.set_ylabel("Mid-study body weight\n(g)",fontsize=16,labelpad=14)
    ax6_carb_weight.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax6_carb_weight.tick_params(axis="both",which="major",labelsize=14)
    ax6_carb_weight.view_init(elev=25,azim=-55)
    #  Fat ratio and mid-study body weight 
    ax6_fat_weight.plot_surface(FatMesh_fat_weight,WeightMesh_fat_weight,LambdaMesh_fat_weight,alpha=0.25,color="blue",edgecolor="none")
    for i in range(len(LambdaValues)):
        ax6_fat_weight.scatter(FatRatios[i],MidStudyWeight[i],LambdaValues[i],s=180,color=DietColors[i],edgecolor="black",linewidth=1.5,depthshade=False)
    ax6_fat_weight.text2D(0.02,0.97,f"$\\lambda={slope_fat_fat_weight:.3f}\\cdot FatRatio{slope_weight_fat_weight:+.3f}\\cdot Weight{intercept_fat_weight:+.3f}$\n$R^2={R_squared_fat_weight:.3f}$",transform=ax6_fat_weight.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax6_fat_weight.set_xlim(np.min(FatRatios)-0.05,np.max(FatRatios)+0.05)
    ax6_fat_weight.set_ylim(np.min(MidStudyWeight)-2,np.max(MidStudyWeight)+2)
    ax6_fat_weight.set_zlim(0.05,0.30)
    ax6_fat_weight.set_zticks([0.05,0.15,0.25])
    ax6_fat_weight.set_xticks([0.1,0.4,0.6])
    ax6_fat_weight.set_yticks([25,35,45])
    ax6_fat_weight.set_xlabel("Dietary fats\n(%)",fontsize=16,labelpad=14)
    ax6_fat_weight.set_ylabel("Mid-study body weight\n(g)",fontsize=16,labelpad=14)
    ax6_fat_weight.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax6_fat_weight.tick_params(axis="both",which="major",labelsize=14)
    ax6_fat_weight.view_init(elev=25,azim=-55)
    #  Empty second row 
    ax6_empty1.axis("off")
    ax6_empty2.axis("off")
    ax6_empty3.axis("off")
    #  Legend 
    legend_handles_fig6 = [plt.Line2D([0],[0],marker="o",color="none",markerfacecolor=DietColors[i],markeredgecolor="black",markeredgewidth=1.5,markersize=13,label=f"{StudyLabels[i]}: {DietCodes[i]}") for i in range(len(DietLabels))]
    legend_handles_fig6.append(plt.Line2D([0],[0],color="blue",lw=8,alpha=0.25,label="Multivariable regression surface"))
    fig6.legend(handles=legend_handles_fig6,loc="lower center",bbox_to_anchor=(0.5,0.04),ncol=4,fontsize=15,framealpha=1)
    fig6.subplots_adjust(left=0.02,right=0.98,bottom=0.12,top=0.96,wspace=0.08,hspace=0.12)
    plt.show()


    # ==========================================================
    # TRIPLE MULTIVARIABLE LINEAR REGRESSION ANALYSIS
    # ==========================================================
    LambdaValues = np.asarray(LambdaValues,dtype=float)
    CarbRatios = np.asarray(CarbRatios,dtype=float)
    FatRatios = np.asarray(FatRatios,dtype=float)
    MidStudyAge = np.asarray(MidStudyAge,dtype=float)
    MidStudyWeight = np.asarray(MidStudyWeight,dtype=float)
    SS_total = np.sum((LambdaValues-np.mean(LambdaValues))**2)
    #  Carbohydrate ratio, mouse age and mid-study body weight 
    X_carb_age_weight = np.column_stack((CarbRatios,MidStudyAge,MidStudyWeight,np.ones(len(LambdaValues))))
    beta_carb_age_weight = np.linalg.lstsq(X_carb_age_weight,LambdaValues,rcond=None)[0]
    slope_carb_carb_age_weight,slope_age_carb_age_weight,slope_weight_carb_age_weight,intercept_carb_age_weight = beta_carb_age_weight
    LambdaPredicted_carb_age_weight = X_carb_age_weight@beta_carb_age_weight
    SS_residual_carb_age_weight = np.sum((LambdaValues-LambdaPredicted_carb_age_weight)**2)
    R_squared_carb_age_weight = 1-SS_residual_carb_age_weight/SS_total
    CarbGrid_carb_age_weight = np.linspace(np.min(CarbRatios),np.max(CarbRatios),50)
    AgeGrid_carb_age_weight = np.linspace(np.min(MidStudyAge),np.max(MidStudyAge),50)
    CarbMesh_carb_age_weight,AgeMesh_carb_age_weight = np.meshgrid(CarbGrid_carb_age_weight,AgeGrid_carb_age_weight)
    LambdaMesh_carb_age_weight = slope_carb_carb_age_weight*CarbMesh_carb_age_weight+slope_age_carb_age_weight*AgeMesh_carb_age_weight+slope_weight_carb_age_weight*np.mean(MidStudyWeight)+intercept_carb_age_weight
    #  Fat ratio, mouse age and mid-study body weight 
    X_fat_age_weight = np.column_stack((FatRatios,MidStudyAge,MidStudyWeight,np.ones(len(LambdaValues))))
    beta_fat_age_weight = np.linalg.lstsq(X_fat_age_weight,LambdaValues,rcond=None)[0]
    slope_fat_fat_age_weight,slope_age_fat_age_weight,slope_weight_fat_age_weight,intercept_fat_age_weight = beta_fat_age_weight
    LambdaPredicted_fat_age_weight = X_fat_age_weight@beta_fat_age_weight
    SS_residual_fat_age_weight = np.sum((LambdaValues-LambdaPredicted_fat_age_weight)**2)
    R_squared_fat_age_weight = 1-SS_residual_fat_age_weight/SS_total
    FatGrid_fat_age_weight = np.linspace(np.min(FatRatios),np.max(FatRatios),50)
    AgeGrid_fat_age_weight = np.linspace(np.min(MidStudyAge),np.max(MidStudyAge),50)
    FatMesh_fat_age_weight,AgeMesh_fat_age_weight = np.meshgrid(FatGrid_fat_age_weight,AgeGrid_fat_age_weight)
    LambdaMesh_fat_age_weight = slope_fat_fat_age_weight*FatMesh_fat_age_weight+slope_age_fat_age_weight*AgeMesh_fat_age_weight+slope_weight_fat_age_weight*np.mean(MidStudyWeight)+intercept_fat_age_weight
    print("Carbohydrate ratio, mouse age and mid-study body weight:")
    print(f"lambda = {slope_carb_carb_age_weight:.6f}*CarbRatio {slope_age_carb_age_weight:+.6f}*MidStudyAge {slope_weight_carb_age_weight:+.6f}*MidStudyWeight {intercept_carb_age_weight:+.6f}")
    print(f"R^2 = {R_squared_carb_age_weight:.6f}")
    print("")
    print("Fat ratio, mouse age and mid-study body weight:")
    print(f"lambda = {slope_fat_fat_age_weight:.6f}*FatRatio {slope_age_fat_age_weight:+.6f}*MidStudyAge {slope_weight_fat_age_weight:+.6f}*MidStudyWeight {intercept_fat_age_weight:+.6f}")
    print(f"R^2 = {R_squared_fat_age_weight:.6f}")
    print("")

    # ==========================================================
    # FIGURE: TRIPLE MULTIVARIABLE LINEAR REGRESSIONS
    # ==========================================================
    fig6 = plt.figure(figsize=(24,10))
    ax6_carb_age_weight = fig6.add_subplot(2,3,1,projection="3d")
    ax6_fat_age_weight = fig6.add_subplot(2,3,2,projection="3d")
    ax6_legend = fig6.add_subplot(2,3,3)
    ax6_empty1 = fig6.add_subplot(2,3,4)
    ax6_empty2 = fig6.add_subplot(2,3,5)
    ax6_empty3 = fig6.add_subplot(2,3,6)
    #  Carbohydrate ratio, mouse age and mid-study body weight 
    #surface_carb_age_weight = ax6_carb_age_weight.plot_surface(CarbMesh_carb_age_weight,AgeMesh_carb_age_weight,LambdaMesh_carb_age_weight,facecolors=plt.cm.viridis((slope_weight_carb_age_weight*(np.mean(MidStudyWeight)-np.min(MidStudyWeight)))/(np.max(LambdaValues)-np.min(LambdaValues)+1e-12)*np.ones_like(LambdaMesh_carb_age_weight)),alpha=0.25,edgecolor="none")
    scatter_carb_age_weight = ax6_carb_age_weight.scatter(CarbRatios,MidStudyAge,LambdaValues,c=MidStudyWeight,cmap="viridis",vmin=np.min(MidStudyWeight),vmax=np.max(MidStudyWeight),s=180,edgecolor="black",linewidth=1.5,depthshade=False,zorder=5)
    ax6_carb_age_weight.text2D(0.02,0.97,f"$\\lambda={slope_carb_carb_age_weight:.3f}\\cdot CarbRatio{slope_age_carb_age_weight:+.3f}\\cdot Age{slope_weight_carb_age_weight:+.3f}\\cdot Weight{intercept_carb_age_weight:+.3f}$\n$R^2={R_squared_carb_age_weight:.3f}$",transform=ax6_carb_age_weight.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax6_carb_age_weight.set_xlim(np.min(CarbRatios)-0.05,np.max(CarbRatios)+0.05)
    ax6_carb_age_weight.set_ylim(np.min(MidStudyAge)-2,np.max(MidStudyAge)+2)
    ax6_carb_age_weight.set_zlim(0.05,0.30)
    ax6_carb_age_weight.set_xticks([0.2,0.4,0.6])
    ax6_carb_age_weight.set_yticks([10,25,40])
    ax6_carb_age_weight.set_zticks([0.05,0.15,0.25])
    ax6_carb_age_weight.set_xlabel("Dietary carbohydrates\n(ratio)",fontsize=16,labelpad=14)
    ax6_carb_age_weight.set_ylabel("Mid-study age\n(weeks)",fontsize=16,labelpad=14)
    ax6_carb_age_weight.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax6_carb_age_weight.set_title("Carbohydrate ratio, mouse age and body weight",fontsize=19,pad=22)
    ax6_carb_age_weight.tick_params(axis="both",which="major",labelsize=14)
    ax6_carb_age_weight.view_init(elev=25,azim=-55)
    #  Fat ratio, mouse age and mid-study body weight 
    #surface_fat_age_weight = ax6_fat_age_weight.plot_surface(FatMesh_fat_age_weight,AgeMesh_fat_age_weight,LambdaMesh_fat_age_weight,facecolors=plt.cm.viridis((slope_weight_fat_age_weight*(np.mean(MidStudyWeight)-np.min(MidStudyWeight)))/(np.max(LambdaValues)-np.min(LambdaValues)+1e-12)*np.ones_like(LambdaMesh_fat_age_weight)),alpha=0.25,edgecolor="none")
    scatter_fat_age_weight = ax6_fat_age_weight.scatter(FatRatios,MidStudyAge,LambdaValues,c=MidStudyWeight,cmap="viridis",vmin=np.min(MidStudyWeight),vmax=np.max(MidStudyWeight),s=180,edgecolor="black",linewidth=1.5,depthshade=False,zorder=5)
    ax6_fat_age_weight.text2D(0.02,0.97,f"$\\lambda={slope_fat_fat_age_weight:.3f}\\cdot FatRatio{slope_age_fat_age_weight:+.3f}\\cdot Age{slope_weight_fat_age_weight:+.3f}\\cdot Weight{intercept_fat_age_weight:+.3f}$\n$R^2={R_squared_fat_age_weight:.3f}$",transform=ax6_fat_age_weight.transAxes,fontsize=15,verticalalignment="top",horizontalalignment="left",bbox=dict(facecolor="white",edgecolor="black",alpha=0.9))
    ax6_fat_age_weight.set_xlim(np.min(FatRatios)-0.05,np.max(FatRatios)+0.05)
    ax6_fat_age_weight.set_ylim(np.min(MidStudyAge)-2,np.max(MidStudyAge)+2)
    ax6_fat_age_weight.set_zlim(0.05,0.30)
    ax6_fat_age_weight.set_xticks([0.1,0.4,0.6])
    ax6_fat_age_weight.set_yticks([10,25,40])
    ax6_fat_age_weight.set_zticks([0.05,0.15,0.25])
    ax6_fat_age_weight.set_xlabel("Dietary fats\n(ratio)",fontsize=16,labelpad=14)
    ax6_fat_age_weight.set_ylabel("Mid-study age\n(weeks)",fontsize=16,labelpad=14)
    ax6_fat_age_weight.set_zlabel("Lambda\n(kcal/g/day)",fontsize=16,labelpad=12)
    ax6_fat_age_weight.set_title("Fat ratio, mouse age and body weight",fontsize=19,pad=22)
    ax6_fat_age_weight.tick_params(axis="both",which="major",labelsize=14)
    ax6_fat_age_weight.view_init(elev=25,azim=-55)
    #  Legend 
    ax6_legend.axis("off")
    legend_handles_fig6 = [plt.Line2D([0],[0],marker="o",color="none",markerfacecolor=DietColors[i],markeredgecolor="black",markeredgewidth=1.5,markersize=13,label=f"{StudyLabels[i]}: {DietCodes[i]}") for i in range(len(DietLabels))]
    legend_handles_fig6.append(plt.Line2D([0],[0],color="blue",lw=8,alpha=0.25,label="Regression surface at mean body weight"))
    cbar = fig6.colorbar(scatter_carb_age_weight,ax=ax6_legend,orientation="vertical",fraction=0.12,pad=0.08)
    cbar.set_label("Mid-study body weight (g)",fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    # ---------- Empty second row ----------
    ax6_empty1.axis("off")
    ax6_empty2.axis("off")
    ax6_empty3.axis("off")
    fig6.subplots_adjust(left=0.02,right=0.98,bottom=0.05,top=0.92,wspace=0.08,hspace=0.08)
    plt.show()