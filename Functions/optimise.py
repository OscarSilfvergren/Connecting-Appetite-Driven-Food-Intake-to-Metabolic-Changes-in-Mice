import json
import csv
import os
import multiprocessing
import numpy as np
import sund

from scipy.stats import chi2
from pyswarm import pso
from pyswarms.single import GlobalBestPSO
from concurrent.futures import ProcessPoolExecutor, as_completed
from Functions.cost import Costfunction_PKstudy, Costfunction_PDstudy, Costfunction_AppetiteBW, Costfunction_SemaglutidePKstudy


# -------------------------- Optimise PK data -------------------------- #

# Exenatide
def Optimise_ExenatidePKstudy_PSO(model, data, x0):

    PKparamsFull = []

    def CostFunction(param_log, model, data, chi2_limit):
        param = np.exp(param_log)
        full_param = np.array(model.parameter_values, dtype=float, copy=True)
        full_param[0:7] = param
        cost = Costfunction_PKstudy(model, full_param, data)
        if cost < chi2_limit:
            full_param_with_cost = np.append(full_param, cost)
            PKparamsFull.append(full_param_with_cost.tolist())
        return cost

    x0 = x0[0:7]
    lb = np.log((1/5) * x0)
    ub = np.log(5 * x0)

    dgf = len(data['RatPK']['all_time'])
    chi2_limit = chi2.ppf(0.95, dgf)

    args_M2_uncertainty = (model, data, chi2_limit)
    result = pso(CostFunction, lb=lb, ub=ub, args=args_M2_uncertainty, swarmsize=400, maxiter=100, omega=0.8, phip=1.5, phig=1.5, seed=1, debug=True)
    best_param_log = result.x
    best_cost = result.fun
    
    # Convert best result back from log-space and save it sorted with the other accepted parameter vectors
    best_param = np.exp(best_param_log)
    full_best_param = np.array(model.parameter_values, dtype=float, copy=True)
    full_best_param[0:7] = best_param
    PKparamsFull.sort(key=lambda x: x[-1])
    project_dir = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(project_dir, 'Parameters')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path_with_cost = os.path.join(csv_dir, 'FullParams_ExenatidePK_PSO.csv')
    with open(csv_path_with_cost, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if len(PKparamsFull) > 0:
            writer.writerows(PKparamsFull)
            
    print(f"Optimise_ExenatidePKstudy: saved {len(PKparamsFull)} rows to {csv_path_with_cost}")
    print(f"Best cost: {best_cost}")  
def Optimise_ExenatidePKstudy_PSO_locked_scan(model, data):

    # 1) Initial parameter vector and names
    x0 = np.asarray(model.parameter_values[0:7], dtype=float)
    param_names = list(model.parameter_names[0:7])
    # 2) Degrees of freedom and chi2 limit
    dgf = len(data['RatPK']['all_time'])
    chi2_limit = chi2.ppf(0.95, dgf)
    # 3) Values to test for each locked parameter
    scale_factors = np.logspace(-2, 2, 20)
    # 4) Master results dictionary
    results_dict = {"chi2_limit": float(chi2_limit), "dgf": int(dgf), "n_locked_parameters": 7, "n_test_values_per_parameter": 20, "total_optimisations": 7 * 20, "runs": []}
    # 5) Loop over each parameter to lock
    for locked_idx in range(7):

        locked_name        = param_names[locked_idx]
        locked_test_values = x0[locked_idx] * scale_factors

        # Indices of the remaining free parameters
        free_indices = [i for i in range(7) if i != locked_idx]

        # Initial guess for the free parameters
        x0_free = x0[free_indices]

        # Bounds in log-space for the free parameters
        lb_free = np.log((1 / 10) * x0_free)
        ub_free = np.log(10 * x0_free)

        # 6) Loop over the 20 fixed values for this locked parameter
        for test_value in locked_test_values:

            def CostFunction_freeparam(free_param_log, model, data, chi2_limit, locked_idx, locked_value, free_indices):
                free_param = np.exp(free_param_log)
                # Build full parameter vector
                full_param = np.array(model.parameter_values, dtype=float, copy=True)
                # Set the 7 optimised parameters
                full_param[locked_idx] = locked_value
                full_param[free_indices] = free_param
                # Compute cost
                cost = Costfunction_PKstudy(model, full_param, data)
                return cost

            args_this_run = (model, data, chi2_limit, locked_idx, test_value, free_indices)

            result = pso(CostFunction_freeparam, lb=lb_free, ub=ub_free, args=args_this_run, swarmsize=200, maxiter=300,  omega=0.5, phip=0.5, phig=0.5, seed=1)
            best_free_param_log = result.x
            best_cost           = result.fun
            best_free_param = np.exp(best_free_param_log)
            # Reconstruct full best parameter vector
            full_best_param = np.array(model.parameter_values, dtype=float, copy=True)
            full_best_param[locked_idx]   = test_value
            full_best_param[free_indices] = best_free_param
            # Save one entry for this optimisation
            run_result = {
                "locked_parameter_index": int(locked_idx),
                "locked_parameter_name": str(locked_name),
                "locked_parameter_value": float(test_value),
                "free_parameter_indices": [int(i) for i in free_indices],
                "best_cost": float(best_cost),
                "best_full_parameter_vector": full_best_param.tolist()
            }
            results_dict["runs"].append(run_result)
            print(
                f"Locked {locked_name} (index {locked_idx}) "
                f"at {test_value:.6g} -> best cost = {best_cost:.6g}"
            )
    # 7) Sort all runs by best cost
    results_dict["runs"].sort(key=lambda x: x["best_cost"])
    # 8) Save results
    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, 'Parameters')
    os.makedirs(save_dir, exist_ok=True)
    # Save JSON with all run information
    json_path = os.path.join(save_dir, 'PKPD_PSO_locked_scan_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    # Save compact CSV summary
    csv_path = os.path.join(save_dir, 'PKPD_PSO_locked_scan_summary.csv')
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "locked_parameter_index",
            "locked_parameter_name",
            "locked_parameter_value",
            "best_cost",
            "best_full_parameter_vector"
        ])
        for run in results_dict["runs"]:
            writer.writerow([
                run["locked_parameter_index"],
                run["locked_parameter_name"],
                run["locked_parameter_value"],
                run["best_cost"],
                run["best_full_parameter_vector"]
            ])
    print(f"Saved JSON results to: {json_path}")
    print(f"Saved CSV summary to: {csv_path}")
    print(f"Total runs completed: {len(results_dict['runs'])}")

    return results_dict

# Semaglutide
def Optimise_SemaglutidePKstudy_PSO(model, data):

    SemaglutidePKparamsFull = []

    x0 = np.array([
        0.01,                       # Ka
        0.009537336906550556,       # V1_ref
        0.00001,                    # CL_ref
    ], dtype=float)

    # 2) Cost function
    def CostFunction(param_log, model, data, chi2_limit):

        param = np.exp(param_log)

        # Mutiply parameters with 1000000000 so we have a linnear relation for the optimised parameters (dont need saturaiton for the datasets) see Note S1 for details
        full_param = np.array(model.parameter_values, dtype=float, copy=True)
        full_param[0] = 1000000000*param[0]   # Ka
        full_param[1] = 1000000000            # Ka50
        full_param[2] = 0                   # Q_ref
        full_param[3] = param[1]            # V1_ref
        full_param[4] = 0.00832             # V2_ref
        full_param[5] = 1000000000*param[2]   # CL_ref
        full_param[6] = 1000000000            # CL50

        cost = Costfunction_SemaglutidePKstudy(model, full_param, data)
        if cost < chi2_limit:
            full_param_with_cost = np.append(full_param, cost)
            SemaglutidePKparamsFull.append(full_param_with_cost.tolist())

        return cost

    lb = np.log((1 / 3) * x0)
    ub = np.log(3 * x0)

    dgf = (
        1 +    # Sema2Exposure
        1 +    # Sema4_Q2DExposure
        1 +    # Sema10Exposure
        16     # Sema10PK_prestudy
    )
    chi2_limit = chi2.ppf(0.95, dgf)

    args_uncertainty = (model, data, chi2_limit)
    result = pso(CostFunction, lb=lb, ub=ub, args=args_uncertainty, swarmsize=300,  maxiter=100, omega=0.8,  phip=1.5, phig=1.5, seed=1, debug=True)
    best_param_log = result.x
    best_param = np.exp(best_param_log)
    best_cost = result.fun

    # Convert best result back from log-space and save it sorted with the other accepted parameter vectors
    full_best_param = np.array(model.parameter_values, dtype=float, copy=True)
    full_best_param[0] = 1000000000*best_param[0]    # Ka
    full_best_param[1] = 1000000000                  # Ka50
    full_best_param[2] = 0                           # Q_ref
    full_best_param[3] = best_param[1]               # V1_ref
    full_best_param[4] = 0.00832                     # V2_ref
    full_best_param[5] = 1000000000*best_param[2]    # CL_ref
    full_best_param[6] = 1000000000                  # CL50

    full_best_param_with_cost = np.append(full_best_param, best_cost)
    SemaglutidePKparamsFull.append(full_best_param_with_cost.tolist())
    SemaglutidePKparamsFull.sort(key=lambda x: x[-1])
    project_dir = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(project_dir, "Parameters")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path_with_cost = os.path.join(csv_dir, "FullParams_SemaglutidePK_PSO.csv")
    with open(csv_path_with_cost, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if len(SemaglutidePKparamsFull) > 0:
            writer.writerows(SemaglutidePKparamsFull)

    print(f"Optimise_SemaglutidePKstudy: saved {len(SemaglutidePKparamsFull)} rows to {csv_path_with_cost}")
    print(f"Best cost: {best_cost}")

    return full_best_param, best_cost
def _run_single_locked_pso_SemaglutidePK(model_name, data, x0, locked_idx, locked_name, test_value, free_indices, lb_free, ub_free, chi2_limit):

    model = sund.load_model(model_name)

    def CostFunction_freeparam(free_param_log, model, x0, data, chi2_limit, locked_idx, locked_value, free_indices):
        free_param = np.exp(free_param_log)
        fitted_param = np.array(x0, dtype=float, copy=True)
        fitted_param[locked_idx] = locked_value
        fitted_param[free_indices] = free_param
        full_param = np.array(model.parameter_values, dtype=float, copy=True)
        # Mutiply parameters with 1000000000 so we have a linnear relation for the optimised parameters (dont need saturaiton for the datasets) see Note S1 for details
        full_param[0] = 1000000000*fitted_param[0]    # Ka
        full_param[1] = 1000000000                    # Ka50
        full_param[2] = 0                             # Q_ref
        full_param[3] = fitted_param[1]               # V1_ref
        full_param[4] = 0.00832                       # V2_ref
        full_param[5] = 1000000000*fitted_param[2]    # CL_ref
        full_param[6] = 1000000000                    # CL50

        model.reset_states()
        try:
            cost = Costfunction_SemaglutidePKstudy(model, full_param, data)
        except Exception:
            return np.inf
        return cost
    args_this_run = (model, x0, data, chi2_limit, locked_idx, test_value, free_indices)

    try:
        result = pso(CostFunction_freeparam, lb=lb_free, ub=ub_free, args=args_this_run, swarmsize=300, maxiter=300, omega=0.8, phip=1.5, phig=1.5, seed=1, debug=False)

        best_free_param_log = result.x
        best_cost = result.fun
        best_free_param = np.exp(best_free_param_log)

        fitted_best_param = np.array(x0, dtype=float, copy=True)
        fitted_best_param[locked_idx] = test_value
        fitted_best_param[free_indices] = best_free_param

        # Mutiply parameters with 1000000000 so we have a linnear relation for the optimised parameters (dont need saturaiton for the datasets) see Note S1 for details
        full_best_param = np.array(model.parameter_values, dtype=float, copy=True)
        full_best_param[0] = 1000000000*fitted_best_param[0]    # Ka
        full_best_param[1] = 1000000000                         # Ka50
        full_best_param[2] = 0                                  # Q_ref
        full_best_param[3] = fitted_best_param[1]               # V1_ref
        full_best_param[4] = 0.00832                            # V2_ref
        full_best_param[5] = 1000000000*fitted_best_param[2]    # CL_ref
        full_best_param[6] = 1000000000                         # CL50

        run_result = {
            "locked_parameter_index": int(locked_idx),
            "locked_parameter_name": str(locked_name),
            "locked_parameter_value": float(test_value),
            "free_parameter_indices": [int(i) for i in free_indices],
            "best_cost": float(best_cost),
            "best_full_parameter_vector": full_best_param.tolist()
        }
        return run_result, None
    except Exception as e:
        return None, f"Failed run for {locked_name} at {test_value:.6g}: {e}"
def Optimise_SemaglutidePKstudy_PSO_locked_scan(model, data):

    model_name = model.__class__.__name__ if not hasattr(model, "name") else model.name

    x0 = np.array([
        11.3/1440,                       # Ka
        0.008,       # V1_ref
        0.0165/1440,                    # CL_ref
    ], dtype=float)

    param_names = ["Ka", "V1_ref", "CL_ref"]
    n_params = len(x0)

    dgf = (
        1 +     # Sema2Exposure
        1 +     # Sema4_Q2DExposure
        1 +     # Sema10Exposure
        16      # Sema10PK_prestudy
    )

    chi2_limit = chi2.ppf(0.95, dgf)

    lb_full = np.log((1 / 2) * x0)
    ub_full = np.log(2 * x0)

    n_test_values = 20
    scale_factors = np.logspace(-2, 2, n_test_values)

    results_dict = {
        "chi2_limit": float(chi2_limit),
        "dgf": int(dgf),
        "n_locked_parameters": int(n_params),
        "n_test_values_per_parameter": int(n_test_values),
        "total_optimisations": int(n_params * n_test_values),
        "runs": []
    }

    jobs = []

    for locked_idx in range(n_params):
        locked_name = param_names[locked_idx]
        locked_test_values = x0[locked_idx] * scale_factors
        free_indices = np.array([i for i in range(n_params) if i != locked_idx], dtype=int)
        lb_free = lb_full[free_indices]
        ub_free = ub_full[free_indices]

        for test_value in locked_test_values:
            jobs.append((model_name, data, x0, locked_idx, locked_name, test_value, free_indices, lb_free, ub_free, chi2_limit))

    n_workers = 5
    batch_size = 40

    for i in range(0, len(jobs), batch_size):
        batch = jobs[i:i + batch_size]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_run_single_locked_pso_SemaglutidePK, *job) for job in batch]
            for future in as_completed(futures):
                run_result, error_msg = future.result()
                if error_msg is not None:
                    print(error_msg)
                    continue
                if run_result is not None:
                    results_dict["runs"].append(run_result)
                    print(
                        f"Locked {run_result['locked_parameter_name']} "
                        f"(index {run_result['locked_parameter_index']}) "
                        f"at {run_result['locked_parameter_value']:.6g} "
                        f"-> best cost = {run_result['best_cost']:.6g}"
                    )
    results_dict["runs"].sort(key=lambda x: x["best_cost"])

    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, "Parameters")
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "LockedScan_SemaglutidePK.json")

    with open(json_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=2)

    print(f"Saved JSON results to: {json_path}")
    print(f"Total runs completed: {len(results_dict['runs'])}")

    return results_dict, None


# -------------------------- Optimise GI data -------------------------- #
def Optimise_PDstudy_PSO(model_PKPD, model_GI, data,x0_full,OptimisedPKparams):
   
    PDparamsFull = []
    PKParams                    = OptimisedPKparams.to_numpy(dtype=float)
    Best_PKParams               = np.asarray(PKParams[0][:-1], dtype=float)

    def CostFunction(param_log, model_PKPD, model_GI, x0_full, data, chi2_limit,Best_PKParams):
        param = np.exp(param_log)
        cost = Costfunction_PDstudy(model_PKPD, model_GI, param, data,Best_PKParams)
        if cost < chi2_limit:   # or use chi2_limit if you prefer
            full_param_with_cost = np.append(param, cost)
            PDparamsFull.append(full_param_with_cost.tolist())
        return cost

    x0_opt     = x0_full
    x0_names   = model_GI.parameter_names
    lb = np.log((1/2) * x0_opt)
    ub = np.log(2     * x0_opt)

    # Restrict bounds for specific parameters based on prior knowledge of constraints 
    idx = x0_names.index("GPRG")
    if lb[idx] < np.log(1):
        lb[idx] = np.log(1)
    if ub[idx] > np.log(4):
        ub[idx] = np.log(4)
    idx = x0_names.index("U_50")
    if lb[idx] < np.log(2):
        lb[idx] = np.log(2)
    if ub[idx] > np.log(20):
        ub[idx] = np.log(20)
    idx = x0_names.index("I_G0")
    if lb[idx] < np.log(9):
        lb[idx] = np.log(9)
    if ub[idx] > np.log(15):
        ub[idx] = np.log(15)
    idx = x0_names.index("ExenatideInsulinMax")
    if lb[idx] < np.log(4):
        lb[idx] = np.log(4)
    if ub[idx] > np.log(13):
        ub[idx] = np.log(13)
    idx = x0_names.index("ExenatideInsulin50")
    if lb[idx] < np.log(0.05):
        lb[idx] = np.log(0.05)
    if ub[idx] > np.log(10):
        ub[idx] = np.log(10)
    idx = x0_names.index("ExenatideInsulin_hill")
    if lb[idx] < np.log(1):
        lb[idx] = np.log(1)
    if ub[idx] > np.log(4):
        ub[idx] = np.log(4)
    idx = x0_names.index("ExenatideInsulinDecayMax")
    if lb[idx] < np.log(2):
        lb[idx] = np.log(2)
    if ub[idx] > np.log(8):
        ub[idx] = np.log(8)
    idx = x0_names.index("ExenatideInsulinDecay50")
    if lb[idx] < np.log(2):
        lb[idx] = np.log(2)
    if ub[idx] > np.log(100):
        ub[idx] = np.log(100)

    dgf = len(data['RatPD']['all_mean'])
    chi2_limit = chi2.ppf(0.95, dgf)


    args_PD_uncertainty = (model_PKPD, model_GI, x0_full, data, chi2_limit,Best_PKParams)
    result = pso(CostFunction, lb=lb, ub=ub, args=args_PD_uncertainty, swarmsize=300, maxiter=300, omega=0.8, phip=1.5, phig=1.5, seed=1, debug=True)
    best_param_log  = result.x
    best_cost       = result.fun
    full_best_param = np.exp(best_param_log)

    # 8) Sort and save accepted parameter vectors
    PDparamsFull.sort(key=lambda x: x[-1])
    project_dir = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(project_dir, 'Parameters')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path_with_cost = os.path.join(csv_dir, 'all_params_sorted_PD_PSO.csv')
    with open(csv_path_with_cost, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if len(PDparamsFull) > 0:
            writer.writerows(PDparamsFull)

    print(f"Optimise_PDstudy: saved {len(PDparamsFull)} rows to {csv_path_with_cost}")
    print(f"Best cost: {best_cost}")

    return full_best_param, best_cost, PDparamsFull
def _run_single_locked_pso_PD(model_pkpd_name, model_gi_name, data, x0_full, Best_PKParams, locked_idx, locked_name, test_value, free_indices, lb_free, ub_free, chi2_limit):

    model_PKPD = sund.load_model(model_pkpd_name)
    model_GI = sund.load_model(model_gi_name)


    def CostFunction_freeparam(free_param_log, model_PKPD, model_GI, x0_full, data, Best_PKParams, chi2_limit, locked_idx, locked_value, free_indices):

        free_param = np.exp(free_param_log)
        full_param = np.array(x0_full, dtype=float, copy=True)
        full_param[locked_idx] = locked_value
        full_param[free_indices] = free_param
        model_PKPD.reset_states()
        model_GI.reset_states()
        try:
            cost = Costfunction_PDstudy(model_PKPD, model_GI, full_param, data, Best_PKParams)
        except Exception:
            return np.inf
        return cost
    args_this_run = (model_PKPD, model_GI, x0_full, data, Best_PKParams, chi2_limit, locked_idx, test_value, free_indices)

    try:
        result = pso(CostFunction_freeparam, lb=lb_free, ub=ub_free, args=args_this_run, swarmsize=300, maxiter=300, omega=0.8, phip=1.5, phig=1.5, seed=1, debug=False)

        best_free_param_log = result.x
        best_cost = result.fun
        best_free_param = np.exp(best_free_param_log)

        full_best_param = np.array(x0_full, dtype=float, copy=True)
        full_best_param[locked_idx] = test_value
        full_best_param[free_indices] = best_free_param

        run_result = {
            "locked_parameter_index": int(locked_idx),
            "locked_parameter_name": str(locked_name),
            "locked_parameter_value": float(test_value),
            "free_parameter_indices": [int(i) for i in free_indices],
            "best_cost": float(best_cost),
            "best_full_parameter_vector": full_best_param.tolist()
        }

        return run_result, None

    except Exception as e:
        return None, f"Failed run for {locked_name} at {test_value:.6g}: {e}"
    
def Optimise_PDstudy_PSO_locked_scan(model_PKPD, model_GI, data, x0_full, OptimisedPKparams):

    PKParams = OptimisedPKparams.to_numpy(dtype=float)
    Best_PKParams = np.asarray(PKParams[0][:-1], dtype=float)

    model_pkpd_name = model_PKPD.__class__.__name__ if not hasattr(model_PKPD, "name") else model_PKPD.name
    model_gi_name = model_GI.__class__.__name__ if not hasattr(model_GI, "name") else model_GI.name

    x0 = np.asarray(x0_full, dtype=float)
    param_names = list(model_GI.parameter_names)
    n_params = len(x0)

    if len(param_names) != n_params:
        param_names = param_names[:n_params]

    dgf = len(data['RatPD']['all_mean'])
    chi2_limit = chi2.ppf(0.95, dgf)

    lb_full = np.log((1 / 2) * x0)
    ub_full = np.log(2 * x0)

    n_test_values = 20
    scale_factors = np.logspace(-2, 2, n_test_values)
    
    results_dict = {
        "chi2_limit": float(chi2_limit),
        "dgf": int(dgf),
        "n_locked_parameters": int(n_params),
        "n_test_values_per_parameter": int(n_test_values),
        "total_optimisations": int(n_params * n_test_values),
        "runs": []
    }

    jobs = []

    for locked_idx in range(n_params):

        locked_name = param_names[locked_idx]

        locked_test_values = x0[locked_idx] * scale_factors
        free_indices = np.array([i for i in range(n_params) if i != locked_idx], dtype=int)

        lb_free = lb_full[free_indices]
        ub_free = ub_full[free_indices]

        for test_value in locked_test_values:
            jobs.append((model_pkpd_name, model_gi_name, data, x0, Best_PKParams, locked_idx, locked_name, test_value, free_indices, lb_free, ub_free, chi2_limit))

    n_workers = 5
    batch_size = 40

    for i in range(0, len(jobs), batch_size):

        batch = jobs[i:i+batch_size]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:

            futures = [executor.submit(_run_single_locked_pso_PD, *job) for job in batch]

            for future in as_completed(futures):

                run_result, error_msg = future.result()

                if error_msg is not None:
                    print(error_msg)
                    continue

                if run_result is not None:
                    results_dict["runs"].append(run_result)

                    print(
                        f"Locked {run_result['locked_parameter_name']} "
                        f"(index {run_result['locked_parameter_index']}) "
                        f"at {run_result['locked_parameter_value']:.6g} "
                        f"-> best cost = {run_result['best_cost']:.6g}"
                    )

    results_dict["runs"].sort(key=lambda x: x["best_cost"])

    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, 'Parameters')
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, 'PD_PSO_locked_scan_results.json')

    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Saved JSON results to: {json_path}")
    print(f"Total runs completed: {len(results_dict['runs'])}")

    return results_dict, None


# -------- Appetite and BW module --------
def _Swarm_AppetiteBW(X, x0_full, data, SimulationRes, chi2_limit, accepted_params):

    X = np.atleast_2d(X)
    costs = np.zeros(X.shape[0], dtype=float)

    #  -- tested to cache models in global scope to avoid reloading them for each particle. THis should make it faster, but caused issues with multiprocessing, so now we load them for each particle instead. I keep these lines here if someone wants to continue our work in optimising the process
    #global _bw_model_cache
    #if "_bw_model_cache" not in globals():
    #    _bw_model_cache = (
    #        sund.load_model("Model_BW"),
    #        sund.load_model("Model_Appetite"),
    #        sund.load_model("Model_GI")
    #    )
    #model_BW, Model_Appetite, Model_GI = _bw_model_cache

    for i, param_log in enumerate(X):

        try:
            model_BW       = sund.load_model("Model_BW")
            Model_Appetite = sund.load_model("Model_Appetite")
            Model_GI       = sund.load_model("Model_GI")

            param = np.exp(param_log)
            full_appetite_param = np.array(x0_full, dtype=float, copy=True)
            full_appetite_param[:-2] = param[:-7]
            full_bw_param = np.asarray(model_BW.parameter_values, dtype=float).copy()
            full_bw_param[0] = param[-7]
            lambda_chow = param[-6]
            lambda_HF = param[-5]
            full_bw_param_with_lambdas = np.concatenate([full_bw_param, [lambda_chow], [lambda_HF]])

            full_GI_param = np.asarray(Model_GI.parameter_values, dtype=float).copy()
            Glucose_baselineScaling = param[-4]
            ChowPhysicalActivity_Scaling = param[-3]
            HFDPhysicalActivity_Scaling = param[-2]
            DiseaseProgression_Scaling = param[-1]
            full_GI_param_with_extra = np.concatenate([full_GI_param, [Glucose_baselineScaling], [ChowPhysicalActivity_Scaling], [HFDPhysicalActivity_Scaling], [DiseaseProgression_Scaling]])

            cost, CostComponents = Costfunction_AppetiteBW(Model_Appetite, Model_GI, model_BW, [], full_appetite_param, full_bw_param_with_lambdas, full_GI_param_with_extra, [], [], data, SimulationRes, "Estimation", "All")
            costs[i] = cost

            if CostComponents["HallBW"] < chi2_limit["HallBW"] and CostComponents["HallEI"] < chi2_limit["HallEI"] and CostComponents["MackStudy1"] < chi2_limit["MackStudy1"] and CostComponents["HeStudy1"] < chi2_limit["HeStudy1"] and CostComponents["Cost_QualitativeAssessments"] < 150:
                accepted_params.append(np.concatenate([param, [cost]]).tolist())

        except Exception as e:
            print("Particle failed:", i)
            print("param:", param if "param" in locals() else "param not created")
            print("error:", repr(e))
            costs[i] = 1e40
    return costs
def Optimise_AppetiteBW__Parallel(data,):

    # Install models
    sund.install_model("Model_Appetite.txt")
    sund.install_model("Model_BW.txt")
    sund.install_model("Model_GI.txt")

    # Load models to inspect parameter names / bounds logic
    Model_Appetite = sund.load_model("Model_Appetite")
    Model_BW = sund.load_model("Model_BW")
    Model_GI = sund.load_model("Model_GI")

    #  Initial values
    x0_full = np.asarray(Model_Appetite.parameter_values, dtype=float)
    x0_opt = np.concatenate([np.array(x0_full[:-2], dtype=float), [0.3869799013749835 ,0.1967580784058201,0.13126533663847198] , [0.4804911458633526,1.3373442739191184,1.657492958687241,0.03535096004258263]])
    opt_param_names = list(Model_Appetite.parameter_names[:-2]) + ["lambda_delayK", "lambda_chow", "lambda_HF"] + ["Glucose_baselineScaling", "ChowPhysicalActivity_Scaling", "HFDPhysicalActivity_Scaling", "DiseaseProgression_Scaling"]
    n_dim = len(x0_opt)
    lb = np.log((1 / 4) * x0_opt)
    ub = np.log(4 * x0_opt)

    # Restrict bounds for specific parameters based on prior knowledge of constraints 
    if "MealIntakeRate" in opt_param_names:
        idx = opt_param_names.index("MealIntakeRate")
        if lb[idx] < np.log(7):
            lb[idx] = np.log(7)
        if ub[idx] > np.log(20):
            ub[idx] = np.log(20)
    if "AppetiteStabiliser" in opt_param_names:
        idx = opt_param_names.index("AppetiteStabiliser")
        if lb[idx] < np.log(1.5):
            lb[idx] = np.log(1.5)
        if ub[idx] > np.log(2.5):
            ub[idx] = np.log(2.5)
    if "EI_HungerRate" in opt_param_names:
        idx = opt_param_names.index("EI_HungerRate")
        if lb[idx] < np.log(0.1):
            lb[idx] = np.log(0.1)
        if ub[idx] > np.log(2):
            ub[idx] = np.log(2)
    if "Time_BaselineShift" in opt_param_names:
        idx = opt_param_names.index("Time_BaselineShift")
        lb[idx] = np.log(0.05)
        if ub[idx] > np.log(1):
            ub[idx] = np.log(1)
    if "lambda_delayK" in opt_param_names:
        idx = opt_param_names.index("lambda_delayK")
        if lb[idx] < np.log(0.3):
            lb[idx] = np.log(0.3)
        if ub[idx] > np.log(1):
            ub[idx] = np.log(1)
    bounds = (lb, ub)

    # ---------- Degrees of freedom / chi2 threshold ----------
    dgf_HallBW = np.sum(np.array(data["Hall_BW"]["EstimationOrValidation"]) == "Estimation")
    dgf_HallEI = np.sum(np.array(data["Hall_EI"]["EstimationOrValidation"]) == "Estimation")
    dgf_MackStudy1 = np.sum(data["MackStudy1"]["all_group"] == "vehicle")
    dgf_HeStudy1 = np.sum(data["HeStudy"]["EstimationOrValidation"] == "Estimation")
    chi2_limit_HallBW = chi2.ppf(0.95, dgf_HallBW)
    chi2_limit_HallEI = chi2.ppf(0.95, dgf_HallEI)
    chi2_limit_MackStudy1 = chi2.ppf(0.95, dgf_MackStudy1)
    chi2_limit_HeStudy1 = chi2.ppf(0.95, dgf_HeStudy1)
    chi2_limits = {"HallBW": chi2_limit_HallBW, "HallEI": chi2_limit_HallEI, "MackStudy1": chi2_limit_MackStudy1, "HeStudy1": chi2_limit_HeStudy1}

    # ---------- Shared storage for accepted parameter vectors ----------
    manager = multiprocessing.Manager()
    accepted_params = manager.list()

    # ---------- Optimiser ----------
    np.random.seed(1)
    n_particles = 120
    x0_log = np.log(x0_opt)
    init_pos = np.zeros((n_particles, n_dim), dtype=float)
    n_local = int(0.3 * n_particles)
    n_medium = int(0.35 * n_particles)
    n_global = n_particles - n_local - n_medium
    local_lb = np.maximum(np.log((1 / 1.2) * x0_opt), lb)
    local_ub = np.minimum(np.log(1.2 * x0_opt), ub)
    medium_lb = np.maximum(np.log((1 / 2) * x0_opt), lb)
    medium_ub = np.minimum(np.log(2 * x0_opt), ub)
    init_pos[:n_local, :] = np.random.uniform(low=local_lb, high=local_ub, size=(n_local, n_dim))
    init_pos[n_local:n_local + n_medium, :] = np.random.uniform(low=medium_lb, high=medium_ub, size=(n_medium, n_dim))
    init_pos[n_local + n_medium:, :] = np.random.uniform(low=lb, high=ub, size=(n_global, n_dim))
    init_pos[0, :] = x0_log # force one particle to start at the initial guess

    Debug = False # True or False to enable/disable debug output
    if Debug:
        debug_accepted_params = []
        initial_costs = _Swarm_AppetiteBW(init_pos, x0_full=x0_full, data=data, SimulationRes=96, chi2_limit=chi2_limits, accepted_params=debug_accepted_params)
        x0_param = np.asarray(x0_opt, dtype=float)
        full_appetite_param_debug = np.array(x0_full, dtype=float, copy=True)
        full_appetite_param_debug[:-2] = x0_param[:-7]
        full_bw_param_debug = np.asarray(Model_BW.parameter_values, dtype=float).copy()
        full_bw_param_debug[0] = x0_param[-7]
        full_bw_param_with_lambdas_debug = np.concatenate([full_bw_param_debug, [x0_param[-6]], [x0_param[-5]]])
        full_GI_param_debug = np.asarray(Model_GI.parameter_values, dtype=float).copy()
        full_GI_param_with_extra_debug = np.concatenate([full_GI_param_debug, [x0_param[-4]], [x0_param[-3]], [x0_param[-2]], [x0_param[-1]]])
        x0_cost_debug, x0_components_debug = Costfunction_AppetiteBW(Model_Appetite, Model_GI, Model_BW, [], full_appetite_param_debug, full_bw_param_with_lambdas_debug, full_GI_param_with_extra_debug, [], [], data, 96, "Estimation", "All")

        print("X0 cost:", x0_cost_debug)
        print("X0 CostComponents:", x0_components_debug)
        print("Initial swarm successful costs:", np.sum(initial_costs < 1e40))
        print("Initial swarm failed costs:", np.sum(initial_costs >= 1e40))
        print("Initial swarm best cost:", np.min(initial_costs))
        print("Initial swarm median cost:", np.median(initial_costs[np.isfinite(initial_costs)]))
        print("Initial swarm accepted vectors:", len(debug_accepted_params))
        print("Number of particles better than X0:", np.sum(initial_costs < x0_cost_debug))
        print("Cost of particle 0:", initial_costs[0])
        print("Difference particle0-X0:", initial_costs[0] - x0_cost_debug)

    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=n_dim, options={"c1": 1.5, "c2": 1.5, "w": 0.8}, bounds=bounds, init_pos=init_pos)
    best_cost, best_pos = optimizer.optimize(_Swarm_AppetiteBW, iters=200, n_processes=8, verbose=True, x0_full=x0_full, data=data, SimulationRes=96, chi2_limit=chi2_limits, accepted_params=accepted_params)
    best_pos = np.exp(best_pos)
    accepted_params.append(np.append(np.asarray(best_pos, dtype=float), float(best_cost)).tolist())

    # ---------- Rebuild accepted parameter vectors ----------
    AcceptedAppetiteParams = []
    AcceptedHallBWParams   = []
    AcceptedHeParams       = []
    for row in accepted_params:
        row = np.asarray(row, dtype=float)
        param = row[:-1]
        cost = row[-1]

        full_appetite_param = np.array(x0_full, dtype=float, copy=True)
        full_appetite_param[:-2] = param[:-7]
        AcceptedAppetiteParams.append(np.append(full_appetite_param, cost).tolist())
        HallBWparams_tmp = np.asarray(Model_BW.parameter_values, dtype=float).copy()
        HallBWparams_tmp[0] = param[-7]      # lambda_delayK

        AcceptedHallBWParams.append(
            np.concatenate([
                HallBWparams_tmp,
                [param[-6]],     # lambda_chow
                [param[-5]],     # lambda_HF
                [cost]
            ]).tolist()
        )

        # ----- He-study parameters -----
        GIparams_tmp = np.asarray(Model_GI.parameter_values, dtype=float).copy()
        AcceptedHeParams.append(
            np.concatenate([
                GIparams_tmp,
                [param[-4]],     # Glucose_baselineScaling
                [param[-3]],     # ChowPhysicalActivity_Scaling
                [param[-2]],     # HFDPhysicalActivity_Scaling
                [param[-1]],     # DiseaseProgression_Scaling
                [cost]
            ]).tolist()
        )

    AcceptedAppetiteParams.sort(key=lambda x: x[-1])
    AcceptedHallBWParams.sort(key=lambda x: x[-1])
    AcceptedHeParams.sort(key=lambda x: x[-1])

    # ---------- Save results ----------
    project_dir = os.path.dirname(os.path.dirname(__file__))
    csv_dir = os.path.join(project_dir, "Parameters")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path_appetite = os.path.join(csv_dir, "Fullparams_Appetite.csv")
    with open(csv_path_appetite, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if len(AcceptedAppetiteParams) > 0:
            writer.writerows(AcceptedAppetiteParams)
    csv_path_bw = os.path.join(csv_dir, "Fullparams_BW.csv")
    with open(csv_path_bw, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if len(AcceptedHallBWParams) > 0:
            writer.writerows(AcceptedHallBWParams)
    csv_path_he = os.path.join(csv_dir, "Fullparams_He.csv")
    with open(csv_path_he, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if len(AcceptedHeParams) > 0:
            writer.writerows(AcceptedHeParams)

    print(f"Saved appetite parameters to {csv_path_appetite}")
    print(f"Saved Hall BW parameters to {csv_path_bw}")
    print(f"Saved He parameters to {csv_path_he}")
    print(f"Accepted appetite parameter vectors: {len(AcceptedAppetiteParams)}")
    print(f"Accepted Hall BW parameter vectors: {len(AcceptedHallBWParams)}")
    print(f"Accepted He parameter vectors: {len(AcceptedHeParams)}")
    print(f"Best cost: {best_cost}")

    return AcceptedAppetiteParams, AcceptedHallBWParams, AcceptedHeParams, best_cost



    sund.install_model("Model_Appetite.txt")
    sund.install_model("Model_BW.txt")
    sund.install_model("Model_GI.txt")

    Model_Appetite = sund.load_model("Model_Appetite")
    model_BW = sund.load_model("Model_BW")
    Model_GI = sund.load_model("Model_GI")

    model_appetite_name = Model_Appetite.__class__.__name__ if not hasattr(Model_Appetite, "name") else Model_Appetite.name
    model_bw_name = model_BW.__class__.__name__ if not hasattr(model_BW, "name") else model_BW.name
    model_gi_name = Model_GI.__class__.__name__ if not hasattr(Model_GI, "name") else Model_GI.name

    # ---------- Initial values ----------
    x0_opt = np.concatenate([np.array(x0_full[:-2], dtype=float), [0.3869799013749835 ,0.1933318776726268,0.13823518979548527] , [0.1616130774098934,1.2624611451215446,1.6355892241774026,0.045910219063971025]])

    param_names = list(Model_Appetite.parameter_names[:-2]) + ["lambda_delayK","lambda_chow","lambda_HF"] + ["Glucose_baselineScaling","ChowPhysicalActivity_Scaling","HFDPhysicalActivity_Scaling","DiseaseProgression_Scaling"]

    n_params = len(x0_opt)

    if len(param_names) != n_params:
        raise ValueError(f"Length mismatch: len(param_names)={len(param_names)}, len(x0_opt)={n_params}")

    if np.any(x0_opt <= 0):
        raise ValueError("All x0_opt values must be positive because optimisation is performed in log-space.")

    # ---------- Locked scan values ----------
    scale_factors = np.logspace(-2, 2, 20)

    results_dict = {
        "n_locked_parameters": int(n_params),
        "n_test_values_per_parameter": int(len(scale_factors)),
        "total_optimisations": int(n_params * len(scale_factors)),
        "parameter_names": param_names,
        "runs": []
    }

    jobs = []

    for locked_idx in range(n_params):
        locked_name = param_names[locked_idx]
        locked_test_values = x0_opt[locked_idx] * scale_factors
        free_indices = np.array([i for i in range(n_params) if i != locked_idx], dtype=int)

        lb_full = np.log((1 / 1.2) * x0_opt)
        ub_full = np.log(1.2 * x0_opt)

        if "MealIntakeRate" in param_names:
            idx = param_names.index("MealIntakeRate")
            if lb_full[idx] < np.log(7):
                lb_full[idx] = np.log(7)
            if ub_full[idx] > np.log(20):
                ub_full[idx] = np.log(20)

        if "AppetiteStabiliser" in param_names:
            idx = param_names.index("AppetiteStabiliser")
            if lb_full[idx] < np.log(1.5):
                lb_full[idx] = np.log(1.5)
            if ub_full[idx] > np.log(3):
                ub_full[idx] = np.log(3)

        lb_free = lb_full[free_indices]
        ub_free = ub_full[free_indices]

        for test_value in locked_test_values:
            jobs.append((model_appetite_name, model_bw_name, model_gi_name, data, x0_full, x0_opt, locked_idx, locked_name, test_value, free_indices, lb_free, ub_free, 96))

    n_workers = 16
    batch_size = 40

    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]

            futures = [executor.submit(_run_single_locked_pso_AppetiteBW, *job) for job in batch]

            for future in as_completed(futures):
                run_result, error_msg = future.result()

                if error_msg is not None:
                    print(error_msg)
                    continue

                results_dict["runs"].append(run_result)
                print(f"Locked {run_result['locked_parameter_name']} at {run_result['locked_parameter_value']:.6g} -> best cost = {run_result['best_cost']:.6g}")

    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, "Parameters")
    os.makedirs(save_dir, exist_ok=True)

    json_path = os.path.join(save_dir, "LockedScan_AppetiteBW.json")

    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Saved JSON results to: {json_path}")
    print(f"Total runs completed: {len(results_dict['runs'])}")

    return results_dict