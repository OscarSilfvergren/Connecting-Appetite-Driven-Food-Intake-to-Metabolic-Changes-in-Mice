# Python package imports (these needs to be installed)
import numpy as np
from Functions.simulate import CalcSimulatedValues_PKstudy, CalcSimulatedValues_PDstudy, CalcSimulatedValues_HallEI_ACstudy, CalcSimulatedValues_MackStudy1, CalcSimulatedValues_HEStudy, CalcSimulatedValues_SemaglutidePKstudy

# -------------------- PK -------------------- #
def Costfunction_PKstudy(model, parameter_values, data):

    try:
        SimulatedValues = CalcSimulatedValues_PKstudy(model, data, parameter_values)
    except Exception:
        return 1e40

    if np.array_equal(data['RatPK']['all_group'], SimulatedValues['groups']):
        CostModel   = ((data['RatPK']['all_mean']  - SimulatedValues['values'])  ** 2) / (data['RatPK']['all_weighted_sd']  ** 2)
    else:
        print('Groups are not the same between data and simulation. Check your code!')
        return 1e40

    # Extra weighting for specific data points to emphasize qualitatively important dynamics
    Index = np.where(SimulatedValues['groups'] == "IV_50")[0][11]
    CostModel[Index] = CostModel[Index] * 2
    Index = np.where(SimulatedValues['groups'] == "IV_5")[0][10]
    CostModel[Index] = CostModel[Index] * 2
    Index = np.where(SimulatedValues['groups'] == "IV_0p5")[0][5]
    CostModel[Index] = CostModel[Index] * 2

    Index = np.where(SimulatedValues['groups'] == "IVinf_50")[0][6]
    CostModel[Index]  = CostModel[Index]  * 3
    Index = np.where(SimulatedValues['groups'] == "IVinf_5")[0][6]
    CostModel[Index]  = CostModel[Index]   * 3
    Index = np.where(SimulatedValues['groups'] == "IVinf_0p5")[0][6]
    CostModel[Index]  = CostModel[Index] * 3

    Index = np.where(SimulatedValues['groups'] == "SC_50")[0][11]
    CostModel[Index]  = CostModel[Index]  * 3
    Index = np.where(SimulatedValues['groups'] == "SC_5")[0][11]
    CostModel[Index]  = CostModel[Index]   * 3
    Index = np.where(SimulatedValues['groups'] == "SC_0p5")[0][11]
    CostModel[Index]  = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "SC_50")[0][4]
    CostModel[Index]  = CostModel[Index]  * 5
    Index = np.where(SimulatedValues['groups'] == "SC_5")[0][4]
    CostModel[Index]  = CostModel[Index]   * 3
    Index = np.where(SimulatedValues['groups'] == "SC_0p5")[0][4]
    CostModel[Index]  = CostModel[Index] * 3

    CostModel = np.nansum(CostModel)

    return CostModel
def Costfunction_SemaglutidePKstudy(model, parameter_values, data):

    try:
        SimulatedValues = CalcSimulatedValues_SemaglutidePKstudy(model, data, parameter_values)
    except Exception:
        return 1e40

    PK_mask = np.isin(np.array(data["AZstudy"]["all_group"]), [ "Sema2Exposure", "Sema4_Q2DExposure", "Sema10Exposure", "Sema10PK_prestudy"])
    if np.array_equal( np.array(data["AZstudy"]["all_group"])[PK_mask], SimulatedValues["groups"]):
        CostModel = (( np.array(data["AZstudy"]["all_mean"])[PK_mask] - SimulatedValues["values"]) ** 2) / (np.array(data["AZstudy"]["all_weighted_sd"])[PK_mask] ** 2)
    else:
        print("Groups are not the same between data and simulation. Check your code!")
        return 1e40

    CostModel = np.nansum(CostModel)

    return CostModel

# -------------------- GI -------------------- #
def Costfunction_PDstudy(model_PKPD, model_GI, parameter_values, data,Best_PKParams):

    try:
        SimulatedValues = CalcSimulatedValues_PDstudy(model_PKPD, model_GI, data, parameter_values,Best_PKParams)
    except Exception:
        return 1e40

    if np.array_equal(data['RatPD']['all_group'], SimulatedValues['groups']):
        CostModel   = ((data['RatPD']['all_mean']  - SimulatedValues['values'])  ** 2) / (data['RatPD']['all_weighted_sd']  ** 2)
    else:
        print('Groups are not the same between data and simulation. Check the code!')
        return 1e40

    # Extra weighting for specific data points to emphasize qualitatively important dynamics
    Index = np.where(SimulatedValues['groups'] == "insulin_0")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "insulin_3")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "insulin_30")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "insulin_300")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "insulin_3000")[0][3]

    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "glucose_0")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "glucose_3")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "glucose_30")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "glucose_300")[0][3]
    CostModel[Index] = CostModel[Index] * 3
    Index = np.where(SimulatedValues['groups'] == "glucose_3000")[0][3]
    CostModel[Index] = CostModel[Index] * 3

    Index = np.where(SimulatedValues['groups'] == "insulin_3")[0][5:11]
    CostModel[Index] = CostModel[Index] * 0.5
    
    CostModel = np.nansum(CostModel)

    return CostModel

# -------------------- BW and appetite -------------------- #
def Cost_HallStudy(Model_Appetite, model_BW, data, Appetite_params, BW_params, SimulationRes, ValidationOrEstimation):

    Hall_SimulatedValues = CalcSimulatedValues_HallEI_ACstudy(Model_Appetite, model_BW, data, Appetite_params, BW_params, SimulationRes)

    Cost_EI = 0
    Cost_EI_extra = 0
    Cost_BW = 0

    # Hall energy intake
    n_EI = len(data["Hall_EI"]["all_group"])
    Sim_EI_values = np.array(Hall_SimulatedValues["values"][:n_EI])
    Sim_EI_groups = np.array(Hall_SimulatedValues["groups"][:n_EI])
    if not np.array_equal(data["Hall_EI"]["all_group"], Sim_EI_groups):
        print("EI groups are not the same between data and simulation.")
        return None
    if ValidationOrEstimation == "Estimation":
        EI_mask = (np.array(data["Hall_EI"]["EstimationOrValidation"]) == "Estimation")
    elif ValidationOrEstimation == "Validation":
        EI_mask = (np.array(data["Hall_EI"]["EstimationOrValidation"]) == "Validation")
    else:
        print("Input 'ValidationOrEstimation' must be 'Estimation' or 'Validation'")
        return None
    Cost_EI = ((np.array(data["Hall_EI"]["all_mean"])[EI_mask] - Sim_EI_values[EI_mask]) ** 2) / (np.array(data["Hall_EI"]["all_weighted_sd"])[EI_mask] ** 2)

    # Extra weighting for specific data points to emphasize qualitatively important dynamics
    Cost_EI_extra = Cost_EI.copy()
    if ValidationOrEstimation == "Estimation":
        Groups_EI = Sim_EI_groups[EI_mask]
        Times_EI = np.array(data["Hall_EI"]["all_time"])[EI_mask]
        Index = np.where(Groups_EI == "Control")[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where(Groups_EI == "4HF_C")[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where((Groups_EI == "4HF_C") & (Times_EI >= 28))[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where(Groups_EI == "7HF_C")[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where((Groups_EI == "7HF_C") & (Times_EI >= 49))[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where(Groups_EI == "20HF_C")[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where((Groups_EI == "20HF_C") & (Times_EI >= 140))[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where(Groups_EI == "HF_C_HF_C")[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where((Groups_EI == "HF_C_HF_C") & (Times_EI >= 49))[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where((Groups_EI == "HF_C_HF_C") & (Times_EI >= 140))[0][0:5]
        Cost_EI[Index] *= 2
        Index = np.where((Groups_EI == "HF_C_HF_C") & (Times_EI >= 70))[0][0:5]
        Cost_EI[Index] *= 2
        for Group in np.unique(Groups_EI):
            Index = np.where(Groups_EI == Group)[0][-20:]
            Cost_EI[Index] *= 2
    Cost_EI = np.nansum(Cost_EI)

    # Extra weighting for specific data points to emphasize qualitatively important dynamics, repeated for Cost_EI_extra
    if ValidationOrEstimation == "Estimation":
        Groups_EI = Sim_EI_groups[EI_mask]
        Times_EI = np.array(data["Hall_EI"]["all_time"])[EI_mask]
        Index = np.where(Groups_EI == "Control")[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where(Groups_EI == "4HF_C")[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where((Groups_EI == "4HF_C") & (Times_EI >= 28))[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where(Groups_EI == "7HF_C")[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where((Groups_EI == "7HF_C") & (Times_EI >= 49))[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where(Groups_EI == "20HF_C")[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where((Groups_EI == "20HF_C") & (Times_EI >= 140))[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where(Groups_EI == "HF_C_HF_C")[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where((Groups_EI == "HF_C_HF_C") & (Times_EI >= 49))[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where((Groups_EI == "HF_C_HF_C") & (Times_EI >= 140))[0][0:5]
        Cost_EI_extra[Index] *= 2
        Index = np.where((Groups_EI == "HF_C_HF_C") & (Times_EI >= 70))[0][0:5]
        Cost_EI_extra[Index] *= 2
        for Group in np.unique(Groups_EI):
            Index = np.where(Groups_EI == Group)[0][-20:]
            Cost_EI_extra[Index] *= 2
    Cost_EI_extra = np.nansum(Cost_EI_extra)


    #  ---------- Hall BW cost ----------
    Sim_BW_values = np.array(Hall_SimulatedValues["values"][n_EI:])
    Sim_BW_groups = np.array(Hall_SimulatedValues["groups"][n_EI:])
    if not np.array_equal(data["Hall_BW"]["all_group"], Sim_BW_groups):
        print("BW groups are not the same between data and simulation.")
        return None
    if ValidationOrEstimation == "Estimation":
        BW_mask = (np.array(data["Hall_BW"]["EstimationOrValidation"]) == "Estimation")
    elif ValidationOrEstimation == "Validation":
        BW_mask = (np.array(data["Hall_BW"]["EstimationOrValidation"]) == "Validation")
    else:
        print("Input 'ValidationOrEstimation' must be 'Estimation' or 'Validation'")
        return None
    Cost_BW = ((np.array(data["Hall_BW"]["all_mean"])[BW_mask] - Sim_BW_values[BW_mask]) ** 2) / (np.array(data["Hall_BW"]["all_sd"])[BW_mask] ** 2)
    
    # Extra weighting for specific data points to emphasize qualitatively important dynamics
    if ValidationOrEstimation == "Estimation":
        Groups_BW = Sim_BW_groups[BW_mask]
        for Group in np.unique(Groups_BW):
            Index = np.where(Groups_BW == Group)[0][-2:]
            Cost_BW[Index] *= 2
    Cost_BW = np.nansum(Cost_BW)

    return {
        "Cost": Cost_BW + Cost_EI,
        "CostExtra": Cost_BW + Cost_EI_extra,
        "HallBW": Cost_BW,
        "HallEI": Cost_EI,
        "HallEIExtra": Cost_EI_extra,
        "AverageAmountOfMealsPerDay": Hall_SimulatedValues["AverageAmountOfMealsPerDay"],
        "MealsAtNight": Hall_SimulatedValues["MealsAtNight"]
    }
def Cost_MackStudy(Model_Appetite, model_PK, data, Appetite_params, PK_Params, IC50, SimulateExenatideGroup, ValidationOrEstimation):

    Cost_MackStudy1 = 0

    Mack_SimulatedValues = CalcSimulatedValues_MackStudy1(Model_Appetite, model_PK, data, Appetite_params, PK_Params, IC50, SimulateExenatideGroup)
    if ValidationOrEstimation == "Estimation":
        Mack_mask = (np.array(data["MackStudy1"]["all_group"]) == "vehicle")
        Data = np.array(data["MackStudy1"]["all_mean"])[Mack_mask]/3
        Simulation = np.array(Mack_SimulatedValues["values"])/3.4
        Weight = np.array(data["MackStudy1"]["all_weighted_sd"])[Mack_mask]/3
        Cost_MackStudy1 = ((Data - Simulation) ** 2) / (Weight ** 2)
    elif ValidationOrEstimation == "Validation":
        Mack_mask = (np.array(data["MackStudy1"]["all_group"]) == "exenatide")
        Data = np.array(data["MackStudy1"]["all_mean"])[Mack_mask]/3
        Simulation = np.array(Mack_SimulatedValues["values"])[Mack_mask]/3.4
        Weight = np.array(data["MackStudy1"]["all_weighted_sd"])[Mack_mask]/3
        Cost_MackStudy1 = ((Data - Simulation) ** 2) / (Weight ** 2)
    else:
        print("Input 'ValidationOrEstimation' must be 'Estimation' or 'Validation'")
        return None
    Cost_MackStudy1_extra = Cost_MackStudy1.copy()
    Cost_MackStudy1 = np.nansum(Cost_MackStudy1)

    # Extra weighting for specific data points to emphasize qualitatively important dynamics
    if ValidationOrEstimation == "Estimation":
        Group = np.array(data["MackStudy1"]["all_group"])[Mack_mask]
        Index = np.where(Group == "vehicle")[0][-1]
        Cost_MackStudy1_extra[Index] *= 10
    Cost_MackStudy1_extra = np.nansum(Cost_MackStudy1_extra) * 2

    return {
        "Cost": Cost_MackStudy1,
        "CostExtra": Cost_MackStudy1_extra,
        "MackStudy1": Cost_MackStudy1,
        "MackStudy1Extra": Cost_MackStudy1_extra
    }
def Cost_HeStudy(Model_Appetite, Model_GI, model_BW, data, Appetite_params, full_GI_param_with_extra, BW_params, ValidationOrEstimation):

    Cost_HeStudy1 = 0

    He_SimulatedValues = CalcSimulatedValues_HEStudy(Model_Appetite, Model_GI, model_BW, data, Appetite_params, full_GI_param_with_extra, BW_params)
    HE_Data_all = np.array(data["HeStudy"]["all_mean"], dtype=float)
    HE_Weight_all = np.array(data["HeStudy"]["all_weighted_sd"], dtype=float)
    HE_Group_all = np.array(data["HeStudy"]["all_group"])
    HE_EstVal_all = np.array(data["HeStudy"]["EstimationOrValidation"])
    HE_Simulation_all = np.array(He_SimulatedValues["values"], dtype=float)
    if ValidationOrEstimation == "Estimation":
        HE_mask = HE_EstVal_all == "Estimation"
        Data = HE_Data_all[HE_mask]
        Simulation = HE_Simulation_all[HE_mask]
        Weight = HE_Weight_all[HE_mask]
        Group = HE_Group_all[HE_mask]
        Cost_HeStudy1 = ((Data - Simulation) ** 2) / (Weight ** 2)
    elif ValidationOrEstimation == "Validation":
        HE_mask = HE_EstVal_all == "Validation"
        Data = HE_Data_all[HE_mask]
        Simulation = HE_Simulation_all[HE_mask]
        Weight = HE_Weight_all[HE_mask]
        Group = HE_Group_all[HE_mask]
        Cost_HeStudy1 = ((Data - Simulation) ** 2) / (Weight ** 2)
    else:
        print("Input 'ValidationOrEstimation' must be 'Estimation' or 'Validation'")
        return None
    Cost_HeStudy1_extra = Cost_HeStudy1.copy()
    Cost_HeStudy1 = np.nansum(Cost_HeStudy1)

    # Extra weighting for specific data points to emphasize qualitatively important dynamics
    if ValidationOrEstimation == "Estimation":
        if np.any(Group == "BW_control"):
            Index = np.where(Group == "BW_control")[0][-1]
            Cost_HeStudy1_extra[Index] *= 10
        if np.any(Group == "BW_HFD"):
            Index = np.where(Group == "BW_HFD")[0][-1]
            Cost_HeStudy1_extra[Index] *= 10
        if np.any(Group == "Glucose_control"):
            Index = np.where(Group == "Glucose_control")[0][-1]
            Cost_HeStudy1_extra[Index] *= 10
        if np.any(Group == "Glucose_HFD"):
            Index = np.where(Group == "Glucose_HFD")[0][-1]
            Cost_HeStudy1_extra[Index] *= 10
    Cost_HeStudy1_extra = np.nansum(Cost_HeStudy1_extra)

    return {
        "Cost": Cost_HeStudy1,
        "CostExtra": Cost_HeStudy1_extra,
        "HeStudy1": Cost_HeStudy1,
        "HeStudy1Extra": Cost_HeStudy1_extra
    }

def Costfunction_AppetiteBW(Model_Appetite, Model_GI, model_BW, model_PK, Appetite_params, BW_params, full_GI_param_with_extra, PK_Params, IC50, data, SimulationRes, ValidationOrEstimation, StudiesToUse):

    FailedCostComponents = {
        "HallBW": 1e40,
        "HallEI": 1e40,
        "MackStudy1": 1e40,
        "HeStudy1": 1e40,
        "Cost_QualitativeAssessments": 1e40
    }

    if StudiesToUse not in ["Hall", "Mack", "He", "All"]:
        print("Input 'StudiesToUse' must be 'Hall', 'Mack', 'He', or 'All'")
        return 1e40, FailedCostComponents

    if ValidationOrEstimation == "Validation":
        SimulateExenatideGroup = True
    elif ValidationOrEstimation == "Estimation":
        SimulateExenatideGroup = False
    else:
        print("Input 'ValidationOrEstimation' must be 'Estimation' or 'Validation'")
        return 1e40, FailedCostComponents

    # set all to zero to avoid errors if a study is not used
    HallResult = {"Cost": 0, "CostExtra": 0, "HallBW": 0, "HallEI": 0, "HallEIExtra": 0, "AverageAmountOfMealsPerDay": np.array([np.nan, np.nan]), "MealsAtNight": np.array([np.nan, np.nan])}
    MackResult = {"Cost": 0, "CostExtra": 0, "MackStudy1": 0, "MackStudy1Extra": 0}
    HeResult   = {"Cost": 0, "CostExtra": 0, "HeStudy1": 0, "HeStudy1Extra": 0}
    Cost_QualitativeAssessments = 0

    try:
        if StudiesToUse in ["Hall", "All"]:
            HallResult = Cost_HallStudy(Model_Appetite, model_BW, data, Appetite_params, BW_params, SimulationRes, ValidationOrEstimation)
            if HallResult is None:
                return 1e40, FailedCostComponents
        if StudiesToUse in ["Mack", "All"]:
            MackResult = Cost_MackStudy(Model_Appetite, model_PK, data, Appetite_params, PK_Params, IC50, SimulateExenatideGroup, ValidationOrEstimation)
            if MackResult is None:
                return 1e40, FailedCostComponents
        if StudiesToUse in ["He", "All"]:
            HeResult = Cost_HeStudy(Model_Appetite, Model_GI, model_BW, data, Appetite_params, full_GI_param_with_extra, BW_params, ValidationOrEstimation)
            if HeResult is None:
                return 1e40, FailedCostComponents
    except Exception as e:
        return 1e40, FailedCostComponents

    if StudiesToUse in ["Hall", "All"] and ValidationOrEstimation == "Estimation":
        target_meals = np.array([15, 10], dtype=float)
        Cost_AverageAmountOfMealsPerDay = np.nansum((target_meals - HallResult["AverageAmountOfMealsPerDay"]) ** 2 / (0.05 * target_meals) ** 2)
        target_night = np.array([70, 60], dtype=float)
        Cost_MealsAtNight = np.nansum((target_night - HallResult["MealsAtNight"]) ** 2 / (0.1 * target_night) ** 2)
        MatrixVariability = np.array([np.log(full_GI_param_with_extra[-3]), np.log(full_GI_param_with_extra[-2])], dtype=float)
        Cost_ConditionAlterations = np.sum(5 * MatrixVariability**2)
        Cost_QualitativeAssessments = Cost_AverageAmountOfMealsPerDay + Cost_MealsAtNight + Cost_ConditionAlterations

    CostModel = HallResult["CostExtra"] + MackResult["CostExtra"] + HeResult["CostExtra"] + Cost_QualitativeAssessments

    CostComponents = {
        "HallBW": HallResult["HallBW"],
        "HallEI": HallResult["HallEI"],
        "MackStudy1": MackResult["MackStudy1"],
        "HeStudy1": HeResult["HeStudy1"],
        "Cost_QualitativeAssessments": Cost_QualitativeAssessments
    }

    return CostModel, CostComponents