import os
import json
import pandas as pd

import numpy as np

# -------------------- General functions -------------------- #
def Data_formating():


    file = "MiceBW_data.xlsx"

    def _load_sheet(sheet_name):
        df = pd.read_excel(file, sheet_name=sheet_name)
        df = df.dropna(how='all').copy()
        df.columns = df.columns.astype(str).str.strip()
        df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
        return df.to_dict(orient='list')

    # ---------------- LOAD SHEETS ---------------- #
    data = {
        "RatPD": _load_sheet("GaoExenatidePD"),
        "RatPK": _load_sheet("GaoExenatidePK"),
        "Hall_BW": _load_sheet("HallBW"),
        "Hall_EI": _load_sheet("HallEI"),
        "MackStudy1": _load_sheet("MackStudy1"),
        "MackStudy2": _load_sheet("MackStudy2"),
        "HeStudy": _load_sheet("HeStudy"),
        "GaberyStudy": _load_sheet("GaberyStudy"),
        "AZstudy": _load_sheet("AZstudy"),
    }

    # ---------------- RAT PK ---------------- #
    PK_specs = [
        ("timeExenatideIV", "ExenatideIV_50nM", "ExenatideIV_50nMSEM", "IV_50"),
        ("timeExenatideIV", "ExenatideIV_5nM", "ExenatideIV_5nMSEM", "IV_5"),
        ("timeExenatideIV", "ExenatideIV_0p5nM", "ExenatideIV_0p5nMSEM", "IV_0p5"),

        ("timeExenatideIVtransfusion", "ExenatideIVinfusion_50nM", "ExenatideIVinfusion_50nMSEM", "IVinf_50"),
        ("timeExenatideIVtransfusion", "ExenatideIVinfusion_5nM", "ExenatideIVinfusion_5nMSEM", "IVinf_5"),
        ("timeExenatideIVtransfusion", "ExenatideIVinfusion_0p5nM", "ExenatideIVinfusion_0p5nMSEM", "IVinf_0p5"),

        ("Exenatidestimesd", "Exenatidesd_50nM", "Exenatidesd_50nMSEM", "SC_50"),
        ("Exenatidestimesd", "Exenatidesd_5nM", "Exenatidesd_5nMSEM", "SC_5"),
        ("Exenatidestimesd", "Exenatidesd_0p5nM", "Exenatidesd_0p5nMSEM", "SC_0p5")
    ]

    PK_time = np.concatenate([np.array(data['RatPK'][time_col], dtype=float)[~np.isnan(np.array(data['RatPK'][time_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][mean_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][sem_col], dtype=float))] / 60 for time_col, mean_col, sem_col, group_name in PK_specs])
    PK_mean = np.concatenate([np.array(data['RatPK'][mean_col], dtype=float)[~np.isnan(np.array(data['RatPK'][time_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][mean_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][sem_col], dtype=float))] / 1000 for time_col, mean_col, sem_col, group_name in PK_specs])
    PK_sd = np.concatenate([np.array(data['RatPK'][sem_col], dtype=float)[~np.isnan(np.array(data['RatPK'][time_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][mean_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][sem_col], dtype=float))] / 1000 for time_col, mean_col, sem_col, group_name in PK_specs])
    PK_group = np.concatenate([np.repeat(group_name, np.sum(~np.isnan(np.array(data['RatPK'][time_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][mean_col], dtype=float)) & ~np.isnan(np.array(data['RatPK'][sem_col], dtype=float)))) for time_col, mean_col, sem_col, group_name in PK_specs])

    PK_weighted_sd = np.maximum(PK_sd, 0.4 * PK_mean)

    assert len(PK_time) == len(PK_mean) == len(PK_sd) == len(PK_weighted_sd) == len(PK_group), f"RatPK vector length mismatch: time={len(PK_time)}, mean={len(PK_mean)}, sd={len(PK_sd)}, weighted_sd={len(PK_weighted_sd)}, group={len(PK_group)}"

    data['RatPK']['all_time'] = PK_time
    data['RatPK']['all_mean'] = PK_mean
    data['RatPK']['all_sd'] = PK_sd
    data['RatPK']['all_weighted_sd'] = PK_weighted_sd
    data['RatPK']['all_group'] = PK_group


    # ---------------- RAT PD ---------------- #
    PD_time = np.concatenate([
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),

        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float),
        np.array(data['RatPD']['Time'][0:12], dtype=float)
    ])
    PD_mean = np.concatenate([
        np.array(data['RatPD']['glucoseMean_0'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseMean_3'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseMean_30'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseMean_300'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseMean_3000'][0:12], dtype=float),

        np.array(data['RatPD']['insulinMean_0'][0:12], dtype=float),
        np.array(data['RatPD']['insulinMean_3'][0:12], dtype=float),
        np.array(data['RatPD']['insulinMean_30'][0:12], dtype=float),
        np.array(data['RatPD']['insulinMean_300'][0:12], dtype=float),
        np.array(data['RatPD']['insulinMean_3000'][0:12], dtype=float)
    ])
    PD_sd = np.concatenate([
        np.array(data['RatPD']['glucoseSEM_0'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseSEM_3'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseSEM_30'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseSEM_300'][0:12], dtype=float),
        np.array(data['RatPD']['glucoseSEM_3000'][0:12], dtype=float),

        np.array(data['RatPD']['insulinSEM_0'][0:12], dtype=float),
        np.array(data['RatPD']['insulinSEM_3'][0:12], dtype=float),
        np.array(data['RatPD']['insulinSEM_30'][0:12], dtype=float),
        np.array(data['RatPD']['insulinSEM_300'][0:12], dtype=float),
        np.array(data['RatPD']['insulinSEM_3000'][0:12], dtype=float)
    ])
    PD_weighted_sd = np.maximum(PD_sd, 0.15 * PD_mean)
    PD_group = np.concatenate([
        np.repeat("glucose_0", len(np.array(data['RatPD']['glucoseMean_0'][0:12], dtype=float))),
        np.repeat("glucose_3", len(np.array(data['RatPD']['glucoseMean_3'][0:12], dtype=float))),
        np.repeat("glucose_30", len(np.array(data['RatPD']['glucoseMean_30'][0:12], dtype=float))),
        np.repeat("glucose_300", len(np.array(data['RatPD']['glucoseMean_300'][0:12], dtype=float))),
        np.repeat("glucose_3000", len(np.array(data['RatPD']['glucoseMean_3000'][0:12], dtype=float))),

        np.repeat("insulin_0", len(np.array(data['RatPD']['insulinMean_0'][0:12], dtype=float))),
        np.repeat("insulin_3", len(np.array(data['RatPD']['insulinMean_3'][0:12], dtype=float))),
        np.repeat("insulin_30", len(np.array(data['RatPD']['insulinMean_30'][0:12], dtype=float))),
        np.repeat("insulin_300", len(np.array(data['RatPD']['insulinMean_300'][0:12], dtype=float))),
        np.repeat("insulin_3000", len(np.array(data['RatPD']['insulinMean_3000'][0:12], dtype=float)))
    ])
    data['RatPD']['all_time'] = PD_time
    data['RatPD']['all_mean'] = PD_mean
    data['RatPD']['all_sd'] = PD_sd
    data['RatPD']['all_weighted_sd'] = PD_weighted_sd
    data['RatPD']['all_group'] = PD_group

    # ---------------- Hall BW ---------------- #

    BW_time = np.concatenate([
        np.array(data["Hall_BW"]["Time"], dtype=float),
        np.array(data["Hall_BW"]["Time"], dtype=float),

        np.array(data["Hall_BW"]["Time"], dtype=float),
        np.array(data["Hall_BW"]["Time"], dtype=float),

        np.array(data["Hall_BW"]["Time"], dtype=float),
        np.array(data["Hall_BW"]["Time"], dtype=float),

        np.array(data["Hall_BW"]["Time"], dtype=float),
        np.array(data["Hall_BW"]["Time"], dtype=float),

        np.array(data["Hall_BW"]["Time"], dtype=float),
        np.array(data["Hall_BW"]["Time"], dtype=float)
    ])
    BW_mean = np.concatenate([
        np.array(data["Hall_BW"]["ControlBW_mean"], dtype=float),
        np.array(data["Hall_BW"]["ControlFM_mean"], dtype=float),

        np.array(data["Hall_BW"]["HF7_CBW_mean"], dtype=float),
        np.array(data["Hall_BW"]["HF7_CFM_mean"], dtype=float),

        np.array(data["Hall_BW"]["HF_C_HF_CBW_mean"], dtype=float),
        np.array(data["Hall_BW"]["HF_C_HF_CFM_mean"], dtype=float),

        np.array(data["Hall_BW"]["HF20_CBW_mean"], dtype=float),
        np.array(data["Hall_BW"]["HF20_CFM_mean"], dtype=float),

        np.array(data["Hall_BW"]["HF4_CBW_mean"], dtype=float),
        np.array(data["Hall_BW"]["HF4_CFM_mean"], dtype=float)
    ])
    BW_sd = np.concatenate([
        np.array(data["Hall_BW"]["ControlBW_sd"], dtype=float),
        np.array(data["Hall_BW"]["ControlFM_sd"], dtype=float),

        np.array(data["Hall_BW"]["HF7_CBW_sd"], dtype=float),
        np.array(data["Hall_BW"]["HF7_CFM_sd"], dtype=float),

        np.array(data["Hall_BW"]["HF_C_HF_CBW_sd"], dtype=float),
        np.array(data["Hall_BW"]["HF_C_HF_CFM_sd"], dtype=float),

        np.array(data["Hall_BW"]["HF20_CBW_sd"], dtype=float),
        np.array(data["Hall_BW"]["HF20_CFM_sd"], dtype=float),

        np.array(data["Hall_BW"]["HF4_CBW_sd"], dtype=float),
        np.array(data["Hall_BW"]["HF4_CFM_sd"], dtype=float)
    ])
    BW_group = np.concatenate([
        np.repeat("BW_control", len(np.array(data["Hall_BW"]["ControlBW_mean"], dtype=float))),
        np.repeat("FM_control", len(np.array(data["Hall_BW"]["ControlFM_mean"], dtype=float))),

        np.repeat("BW_7HF_C", len(np.array(data["Hall_BW"]["HF7_CBW_mean"], dtype=float))),
        np.repeat("FM_7HF_C", len(np.array(data["Hall_BW"]["HF7_CFM_mean"], dtype=float))),

        np.repeat("BW_HF_C_HF_C", len(np.array(data["Hall_BW"]["HF_C_HF_CBW_mean"], dtype=float))),
        np.repeat("FM_HF_C_HF_C", len(np.array(data["Hall_BW"]["HF_C_HF_CFM_mean"], dtype=float))),

        np.repeat("BW_20HF_C", len(np.array(data["Hall_BW"]["HF20_CBW_mean"], dtype=float))),
        np.repeat("FM_20HF_C", len(np.array(data["Hall_BW"]["HF20_CFM_mean"], dtype=float))),

        np.repeat("BW_4HF_C", len(np.array(data["Hall_BW"]["HF4_CBW_mean"], dtype=float))),
        np.repeat("FM_4HF_C", len(np.array(data["Hall_BW"]["HF4_CFM_mean"], dtype=float)))
    ])
    EstimationOrValidation = np.concatenate([
        np.repeat("Estimation", len(np.array(data["Hall_BW"]["ControlBW_mean"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_BW"]["ControlFM_mean"], dtype=float))),

        np.repeat("Estimation", len(np.array(data["Hall_BW"]["HF7_CBW_mean"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_BW"]["HF7_CFM_mean"], dtype=float))),

        np.repeat("Validation", len(np.array(data["Hall_BW"]["HF_C_HF_CBW_mean"], dtype=float))),
        np.repeat("Validation", len(np.array(data["Hall_BW"]["HF_C_HF_CFM_mean"], dtype=float))),

        np.repeat("Estimation", len(np.array(data["Hall_BW"]["HF20_CBW_mean"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_BW"]["HF20_CFM_mean"], dtype=float))),

        np.repeat("Estimation", len(np.array(data["Hall_BW"]["HF4_CBW_mean"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_BW"]["HF4_CFM_mean"], dtype=float)))
    ])

    # Store in data structure
    data["Hall_BW"]["all_time"]   = BW_time
    data["Hall_BW"]["all_mean"]   = BW_mean
    data["Hall_BW"]["all_sd"]     = BW_sd
    data["Hall_BW"]["all_group"]  = BW_group
    data["Hall_BW"]["EstimationOrValidation"] = EstimationOrValidation

    # ---------- Load EI ----------
    EI_time = np.concatenate([
        np.array(data["Hall_EI"]["Time"], dtype=float),
        np.array(data["Hall_EI"]["Time"], dtype=float),
        np.array(data["Hall_EI"]["Time"], dtype=float),
        np.array(data["Hall_EI"]["Time"], dtype=float),
        np.array(data["Hall_EI"]["Time"], dtype=float)
    ])
    EI_mean = np.concatenate([
        np.array(data["Hall_EI"]["Control"], dtype=float),
        np.array(data["Hall_EI"]["7HF-C"], dtype=float),
        np.array(data["Hall_EI"]["HF-C-HF-C"], dtype=float),
        np.array(data["Hall_EI"]["20HF-C"], dtype=float),
        np.array(data["Hall_EI"]["4HF-C"], dtype=float)
    ])
    EI_weighted_sd = 0.35 * EI_mean
    EI_group = np.concatenate([
        np.repeat("Control", len(np.array(data["Hall_EI"]["Control"], dtype=float))),
        np.repeat("7HF_C", len(np.array(data["Hall_EI"]["7HF-C"], dtype=float))),
        np.repeat("HF_C_HF_C", len(np.array(data["Hall_EI"]["HF-C-HF-C"], dtype=float))),
        np.repeat("20HF_C", len(np.array(data["Hall_EI"]["20HF-C"], dtype=float))),
        np.repeat("4HF_C", len(np.array(data["Hall_EI"]["4HF-C"], dtype=float)))
    ])
    EstimationOrValidation = np.concatenate([
        np.repeat("Estimation", len(np.array(data["Hall_EI"]["Control"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_EI"]["7HF-C"], dtype=float))),
        np.repeat("Validation", len(np.array(data["Hall_EI"]["HF-C-HF-C"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_EI"]["20HF-C"], dtype=float))),
        np.repeat("Estimation", len(np.array(data["Hall_EI"]["4HF-C"], dtype=float)))
    ])

    data["Hall_EI"]["all_time"] = EI_time
    data["Hall_EI"]["all_mean"] = EI_mean
    data["Hall_EI"]["all_weighted_sd"] = EI_weighted_sd
    data["Hall_EI"]["all_group"] = EI_group
    data["Hall_EI"]["EstimationOrValidation"] = EstimationOrValidation


    # ---------------- Mack study 1 ---------------- #
    PD_time = np.concatenate([
        np.array(data['MackStudy1']['Time'][0:7], dtype=float),
        np.array(data['MackStudy1']['Time'][0:7], dtype=float)
    ])
    PD_mean = np.concatenate([
        np.array(data['MackStudy1']['VehicleEI'][0:7], dtype=float),
        np.array(data['MackStudy1']['ExenatideEI'][0:7], dtype=float)
    ])
    PD_sd = np.concatenate([
        np.array(data['MackStudy1']['VehicleEI_stdev'][0:7], dtype=float),
        np.array(data['MackStudy1']['ExenatideEI_stdev'][0:7], dtype=float)
    ])
    PD_weighted_sd = np.maximum(PD_sd, 0.15 * PD_mean)
    PD_group = np.concatenate([
        np.repeat("vehicle", len(np.array(data['MackStudy1']['VehicleEI'][0:7], dtype=float))),
        np.repeat("exenatide", len(np.array(data['MackStudy1']['ExenatideEI'][0:7], dtype=float))),
    ])
    data['MackStudy1']['all_time'] = PD_time
    data['MackStudy1']['all_mean'] = PD_mean
    data['MackStudy1']['all_sd']   = PD_sd
    data['MackStudy1']['all_weighted_sd'] = PD_weighted_sd
    data['MackStudy1']['all_group'] = PD_group

    # ---------------- Mack study 2 ---------------- #
    PD_time = np.concatenate([
        # BW
        np.array(data['MackStudy2']['BW_Time'][0:5], dtype=float),
        np.array(data['MackStudy2']['BW_Time'][0:5], dtype=float),
        np.array(data['MackStudy2']['BW_Time'][0:5], dtype=float),
        np.array(data['MackStudy2']['BW_Time'][0:5], dtype=float),
        # EI cumulative
        np.array(data['MackStudy2']['EI_Time'][0:4], dtype=float),
        np.array(data['MackStudy2']['EI_Time'][0:4], dtype=float),
        np.array(data['MackStudy2']['EI_Time'][0:4], dtype=float),
        np.array(data['MackStudy2']['EI_Time'][0:4], dtype=float),
    ])
    PD_mean = np.concatenate([
        # BW
        np.array(data['MackStudy2']['VehicleBW'][0:5], dtype=float),
        np.array(data['MackStudy2']['E3BW'][0:5], dtype=float),
        np.array(data['MackStudy2']['E10BW'][0:5], dtype=float),
        np.array(data['MackStudy2']['E30BW'][0:5], dtype=float),
        # EI cumulative
        np.array(data['MackStudy2']['VehicleEI'][0:4], dtype=float),
        np.array(data['MackStudy2']['E3EI'][0:4], dtype=float),
        np.array(data['MackStudy2']['E10EI'][0:4], dtype=float),
        np.array(data['MackStudy2']['E30EI'][0:4], dtype=float),
    ])
    PD_sd = np.concatenate([
        # BW
        np.array(data['MackStudy2']['VehicleBW_stdev'][0:5], dtype=float),
        np.array(data['MackStudy2']['E3BW_stdev'][0:5], dtype=float),
        np.array(data['MackStudy2']['E10BW_stdev'][0:5], dtype=float),
        np.array(data['MackStudy2']['E30BW_stdev'][0:5], dtype=float),
        # EI cumulative
        np.array(data['MackStudy2']['VehicleEI_stdev'][0:4], dtype=float),
        np.array(data['MackStudy2']['E3EI_stdev'][0:4], dtype=float),
        np.array(data['MackStudy2']['E10EI_stdev'][0:4], dtype=float),
        np.array(data['MackStudy2']['E30EI_stdev'][0:4], dtype=float),
    ])

    PD_weighted_sd = np.maximum(PD_sd, 0.15 * PD_mean)


    data['MackStudy2']['all_time'] = PD_time
    data['MackStudy2']['all_mean'] = PD_mean
    data['MackStudy2']['all_sd'] = PD_sd
    data['MackStudy2']['all_weighted_sd'] = PD_weighted_sd


    # ---------------- He Study ---------------- #
    HE_time = np.concatenate([
        np.array(data["HeStudy"]["time_BW"][0:15], dtype=float),
        np.array(data["HeStudy"]["time_BW"][0:15], dtype=float),

        np.array(data["HeStudy"]["time_Glucose"][0:16], dtype=float),
        np.array(data["HeStudy"]["time_Glucose"][0:16], dtype=float)
    ])
    HE_mean = np.concatenate([
        np.array(data["HeStudy"]["Control_BW"][0:15], dtype=float),
        np.array(data["HeStudy"]["HFD_BW"][0:15], dtype=float),

        np.array(data["HeStudy"]["Control_Glucose"][0:16], dtype=float),
        np.array(data["HeStudy"]["HFD_Glucose"][0:16], dtype=float)
    ])
    HE_sd = np.concatenate([
        np.array(data["HeStudy"]["Control_BW_stdev"][0:15], dtype=float),
        np.array(data["HeStudy"]["HFD_BW_stdev"][0:15], dtype=float),

        np.array(data["HeStudy"]["Control_Glucose_stdev"][0:16], dtype=float),
        np.array(data["HeStudy"]["HFD_Glucose_stdev"][0:16], dtype=float)
    ])
    HE_weighted_sd = np.maximum(HE_sd, 0.15 * HE_mean)
    HE_group = np.concatenate([
        np.repeat("BW_control", len(np.array(data["HeStudy"]["Control_BW"][0:15], dtype=float))),
        np.repeat("BW_HFD", len(np.array(data["HeStudy"]["HFD_BW"][0:15], dtype=float))),

        np.repeat("Glucose_control", len(np.array(data["HeStudy"]["Control_Glucose"][0:16], dtype=float))),
        np.repeat("Glucose_HFD", len(np.array(data["HeStudy"]["HFD_Glucose"][0:16], dtype=float)))
    ])
    EstimationOrValidation = np.concatenate([
        np.repeat("Estimation", len(np.array(data["HeStudy"]["Control_BW"][0:15], dtype=float))),
        np.repeat("Estimation", len(np.array(data["HeStudy"]["HFD_BW"][0:15], dtype=float))),

        np.repeat("Estimation", len(np.array(data["HeStudy"]["Control_Glucose"][0:16], dtype=float))),
        np.repeat("Estimation", len(np.array(data["HeStudy"]["HFD_Glucose"][0:16], dtype=float)))
    ])

    # Store in data structure
    data["HeStudy"]["all_time"]   = HE_time
    data["HeStudy"]["all_mean"]   = HE_mean
    data["HeStudy"]["all_sd"]     = HE_sd
    data['HeStudy']['all_weighted_sd'] = HE_weighted_sd
    data["HeStudy"]["all_group"]  = HE_group
    data["HeStudy"]["EstimationOrValidation"] = EstimationOrValidation


    # ---------------- GaberyStudy ---------------- #
    Gabery_time = np.concatenate([
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),

        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["time"][0:24], dtype=float)
    ])

    Gabery_mean = np.concatenate([
        np.array(data["GaberyStudy"]["VehicleBW"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema1BW"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema3BW"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema10BW"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema30BW"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema100BW"][0:24], dtype=float),

        np.array(data["GaberyStudy"]["VehicleFI"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema1FI"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema3FI"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema10FI"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema30FI"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema100FI"][0:24], dtype=float)
    ])

    Gabery_sd = np.concatenate([
        np.array(data["GaberyStudy"]["VehicleBW_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema1BW_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema3BW_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema10BW_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema30BW_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema100BW_stdev"][0:24], dtype=float),

        np.array(data["GaberyStudy"]["VehicleFI_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema1FI_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema3FI_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema10FI_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema30FI_stdev"][0:24], dtype=float),
        np.array(data["GaberyStudy"]["Sema100FI_stdev"][0:24], dtype=float)
    ])

    Gabery_weighted_sd = np.maximum(Gabery_sd, 0.15 * Gabery_mean)

    Gabery_group = np.concatenate([
        np.repeat("VehicleBW", len(np.array(data["GaberyStudy"]["VehicleBW"][0:24], dtype=float))),
        np.repeat("Sema1BW", len(np.array(data["GaberyStudy"]["Sema1BW"][0:24], dtype=float))),
        np.repeat("Sema3BW", len(np.array(data["GaberyStudy"]["Sema3BW"][0:24], dtype=float))),
        np.repeat("Sema10BW", len(np.array(data["GaberyStudy"]["Sema10BW"][0:24], dtype=float))),
        np.repeat("Sema30BW", len(np.array(data["GaberyStudy"]["Sema30BW"][0:24], dtype=float))),
        np.repeat("Sema100BW", len(np.array(data["GaberyStudy"]["Sema100BW"][0:24], dtype=float))),

        np.repeat("VehicleFI", len(np.array(data["GaberyStudy"]["VehicleFI"][0:24], dtype=float))),
        np.repeat("Sema1FI", len(np.array(data["GaberyStudy"]["Sema1FI"][0:24], dtype=float))),
        np.repeat("Sema3FI", len(np.array(data["GaberyStudy"]["Sema3FI"][0:24], dtype=float))),
        np.repeat("Sema10FI", len(np.array(data["GaberyStudy"]["Sema10FI"][0:24], dtype=float))),
        np.repeat("Sema30FI", len(np.array(data["GaberyStudy"]["Sema30FI"][0:24], dtype=float))),
        np.repeat("Sema100FI", len(np.array(data["GaberyStudy"]["Sema100FI"][0:24], dtype=float)))
    ])

    data["GaberyStudy"]["all_time"] = Gabery_time
    data["GaberyStudy"]["all_mean"] = Gabery_mean
    data["GaberyStudy"]["all_sd"] = Gabery_sd
    data["GaberyStudy"]["all_weighted_sd"] = Gabery_weighted_sd
    data["GaberyStudy"]["all_group"] = Gabery_group



    # ---------------- AZ Study ---------------- #
    AZ_time = np.concatenate([
        np.array(data["AZstudy"]["timeBW"][0:22], dtype=float),
        np.array(data["AZstudy"]["timeBW"][0:22], dtype=float),
        np.array(data["AZstudy"]["timeBW"][0:22], dtype=float),
        np.array(data["AZstudy"]["timeBW"][0:22], dtype=float),
        np.array(data["AZstudy"]["timeBW"][0:34], dtype=float),

        np.array(data["AZstudy"]["timeFI"][0:3], dtype=float),
        np.array(data["AZstudy"]["timeFI"][0:3], dtype=float),
        np.array(data["AZstudy"]["timeFI"][0:3], dtype=float),
        np.array(data["AZstudy"]["timeFI"][0:3], dtype=float),
        np.array(data["AZstudy"]["timeFI"][0:4], dtype=float),

        np.array(data["AZstudy"]["timeInsulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeInsulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeInsulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeInsulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeInsulin"][0:1], dtype=float),

        np.array(data["AZstudy"]["timeGlucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeGlucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeGlucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeGlucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeGlucose"][0:1], dtype=float),

        np.array(data["AZstudy"]["timeExposure"][0:1], dtype=float),
        np.array(data["AZstudy"]["timeExposure"][0:1], dtype=float),

        np.array(data["AZstudy"]["timeExposureLateEnd"][0:1], dtype=float),

        np.array(data["AZstudy"]["time_prestudy"][0:16], dtype=float)
    ])

    AZ_mean = np.concatenate([
        np.array(data["AZstudy"]["VehicleBW"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema0p3BW"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema2BW"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DBW"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema10BW"][0:34], dtype=float),

        np.array(data["AZstudy"]["VehicleFI"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema0p3FI"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema2FI"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DFI"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema10FI"][0:4], dtype=float),

        np.array(data["AZstudy"]["VehicleInsulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema0p3Insulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema2Insulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DInsulin"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema10Insulin"][0:1], dtype=float),

        np.array(data["AZstudy"]["VehicleGlucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema0p3Glucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema2Glucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DGlucose"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema10Glucose"][0:1], dtype=float),

        np.array(data["AZstudy"]["Sema2Exposure"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DExposure"][0:1], dtype=float),

        np.array(data["AZstudy"]["Sema10Exposure"][0:1], dtype=float),

        np.array(data["AZstudy"]["Sema10PK_prestudy"][0:16], dtype=float)
    ])

    AZ_sd = np.concatenate([
        np.array(data["AZstudy"]["VehicleBW_stdev"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema0p3BW_stdev"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema2BW_stdev"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DBW_stdev"][0:22], dtype=float),
        np.array(data["AZstudy"]["Sema10BW_stdev"][0:34], dtype=float),

        np.array(data["AZstudy"]["VehicleFI_stdev"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema0p3FI_stdev"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema2FI_stdev"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DFI_stdev"][0:3], dtype=float),
        np.array(data["AZstudy"]["Sema10FI_stdev"][0:4], dtype=float),

        np.array(data["AZstudy"]["VehicleInsulin_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema0p3Insulin_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema2Insulin_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DInsulin_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema10Insulin_stdev"][0:1], dtype=float),

        np.array(data["AZstudy"]["VehicleGlucose_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema0p3Glucose_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema2Glucose_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DGlucose_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema10Glucose_stdev"][0:1], dtype=float),

        np.array(data["AZstudy"]["Sema2Exposure_stdev"][0:1], dtype=float),
        np.array(data["AZstudy"]["Sema4_Q2DExposure_stdev"][0:1], dtype=float),

        np.array(data["AZstudy"]["Sema10Exposure_stdev"][0:1], dtype=float),
    ])

    AZ_weighted_sd = np.concatenate([np.maximum(AZ_sd[0:151], 0.15 * AZ_mean[0:151]), 0.35 * np.array(data["AZstudy"]["Sema10PK_prestudy"][0:16], dtype=float)])
    
    AZ_group = np.concatenate([
        np.repeat("VehicleBW", len(np.array(data["AZstudy"]["VehicleBW"][0:22], dtype=float))),
        np.repeat("Sema0p3BW", len(np.array(data["AZstudy"]["Sema0p3BW"][0:22], dtype=float))),
        np.repeat("Sema2BW", len(np.array(data["AZstudy"]["Sema2BW"][0:22], dtype=float))),
        np.repeat("Sema4_Q2DBW", len(np.array(data["AZstudy"]["Sema4_Q2DBW"][0:22], dtype=float))),
        np.repeat("Sema10BW", len(np.array(data["AZstudy"]["Sema10BW"][0:34], dtype=float))),

        np.repeat("VehicleFI", len(np.array(data["AZstudy"]["VehicleFI"][0:3], dtype=float))),
        np.repeat("Sema0p3FI", len(np.array(data["AZstudy"]["Sema0p3FI"][0:3], dtype=float))),
        np.repeat("Sema2FI", len(np.array(data["AZstudy"]["Sema2FI"][0:3], dtype=float))),
        np.repeat("Sema4_Q2DFI", len(np.array(data["AZstudy"]["Sema4_Q2DFI"][0:3], dtype=float))),
        np.repeat("Sema10FI", len(np.array(data["AZstudy"]["Sema10FI"][0:4], dtype=float))),

        np.repeat("VehicleInsulin", 1),
        np.repeat("Sema0p3Insulin", 1),
        np.repeat("Sema2Insulin", 1),
        np.repeat("Sema4_Q2DInsulin", 1),
        np.repeat("Sema10Insulin", 1),

        np.repeat("VehicleGlucose", 1),
        np.repeat("Sema0p3Glucose", 1),
        np.repeat("Sema2Glucose", 1),
        np.repeat("Sema4_Q2DGlucose", 1),
        np.repeat("Sema10Glucose", 1),

        np.repeat("Sema2Exposure", 1),
        np.repeat("Sema4_Q2DExposure", 1),

        np.repeat("Sema10Exposure", 1),

        np.repeat("Sema10PK_prestudy", len(np.array(data["AZstudy"]["Sema10PK_prestudy"][0:16], dtype=float)))
    ])

    data["AZstudy"]["all_time"] = AZ_time
    data["AZstudy"]["all_mean"] = AZ_mean
    data["AZstudy"]["all_sd"] = AZ_sd
    data["AZstudy"]["all_weighted_sd"] = AZ_weighted_sd
    data["AZstudy"]["all_group"] = AZ_group


    return [data]

def assign_state_values_from_simulation(model, simulation):
    sim_state_names = list(simulation.state_names)
    sim_state_values = np.asarray(simulation.state_values)
    if sim_state_values.ndim > 1:
        sim_state_values = sim_state_values.ravel()

    model_state_names = list(model.state_names)
    model_state_values = np.asarray(model.state_values)
    if model_state_values.ndim > 1:
        model_state_values = model_state_values.ravel()

    if model_state_values.shape[0] != len(model_state_names):
        model_state_values = np.zeros(len(model_state_names), dtype=sim_state_values.dtype)
    else:
        model_state_values = model_state_values.copy()

    sim_indices = {}
    for sim_idx, sim_name in enumerate(sim_state_names):
        sim_indices.setdefault(sim_name, []).append(sim_idx)

    used_counts = {}
    for model_idx, model_name in enumerate(model_state_names):
        if model_name not in sim_indices:
            continue
        count = used_counts.get(model_name, 0)
        if count < len(sim_indices[model_name]):
            sim_idx = sim_indices[model_name][count]
            model_state_values[model_idx] = sim_state_values[sim_idx]
            used_counts[model_name] = count + 1

    model.state_values = model_state_values
    return model

def load_locked_scan_results(filename):
    project_dir = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_dir, 'Parameters')

    json_path = os.path.join(save_dir, filename)

    with open(json_path, 'r') as f:
        results_dict = json.load(f)

    print(f"Loaded results from: {json_path}")
    print(f"Number of runs: {len(results_dict['runs'])}")

    return results_dict


# -------------------- Meal calculation functions -------------------- #
def average_meals_per_day_and_meal_size(ei_ac, eating_array, time,TimeRes):

    ei_ac = np.asarray(ei_ac, dtype=float)
    eating_array = np.asarray(eating_array, dtype=float)
    time = np.asarray(time, dtype=float)

    daily_ei = daily_energy_intake_from_ei_ac(ei_ac, time)

    meal_starts = (eating_array[1:] > 0.5) & (eating_array[:-1] <= 0.5)
    total_meals = np.sum(meal_starts) + int(eating_array[0] > 0.5)

    meal_start_idx = np.where(meal_starts)[0] + 1

    if eating_array[0] > 0.5:
        meal_start_idx = np.insert(meal_start_idx, 0, 0)

    meal_start_hours = (time[meal_start_idx] % 1.0) * 24
    night_meals = np.sum((meal_start_hours < 9.0) | (meal_start_hours >= 21.0))
    n_days = len(daily_ei)

    meals_per_day = total_meals / n_days if n_days > 0 else 0.0
    mean_meal_size = np.sum(daily_ei) / total_meals if total_meals > 0 else 0.0
    night_meal_percent = 100 * night_meals / total_meals if total_meals > 0 else 0.0

    return meals_per_day, mean_meal_size, daily_ei, night_meal_percent

def daily_energy_intake_from_ei_ac(ei_ac, time):
    ei_ac = np.asarray(ei_ac, dtype=float)
    time = np.asarray(time, dtype=float)
    first_day = int(np.floor(np.nanmin(time)))
    last_day  = int(np.floor(np.nanmax(time)))
    day_edges = np.arange(first_day, last_day + 1, dtype=float)
    ei_edges = np.interp(day_edges, time, ei_ac)
    daily_ei = np.diff(ei_edges)
    return daily_ei

