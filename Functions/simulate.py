import numpy as np
import sund
from Functions.functions import assign_state_values_from_simulation, average_meals_per_day_and_meal_size


# - ------- Simulate PK studies and export the simulated values at the datapoints -----

# Exenatide
def Simulate_PKstudy(PKmodel, PKparams):

    PKmodel.parameter_values = np.asarray(PKparams, dtype=float)
    PKmodel_state_values_reset = np.asarray(PKmodel.state_values, dtype=float).copy()
    V1ref = PKparams[PKmodel.parameter_names.index("V1_ref")]

    # IV_low
    activity_IV_low = sund.Activity(time_unit="d")
    activity_IV_low.add_state_manipulation(mode="add", name="C1", t=[0], f=[0.5/V1ref])
    activity_IV_low.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_IV_low.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationIV_low = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_IV_low], time_vector=np.linspace(0, 420, 421)/1440)
    simulationIV_low.simulate()
    # IV_mid
    activity_IV_mid = sund.Activity(time_unit="d")
    activity_IV_mid.add_state_manipulation(mode="add", name="C1", t=[0], f=[5/V1ref])
    activity_IV_mid.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_IV_mid.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationIV_mid = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_IV_mid], time_vector=np.linspace(0, 420, 421)/1440)
    simulationIV_mid.simulate()
    # IV_high
    activity_IV_high = sund.Activity(time_unit="d")
    activity_IV_high.add_state_manipulation(mode="add", name="C1", t=[0], f=[50/V1ref])
    activity_IV_high.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_IV_high.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationIV_high = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_IV_high], time_vector=np.linspace(0, 420, 421)/1440)
    simulationIV_high.simulate()
    # IV_infusion_low
    activity_IV_infusion_low = sund.Activity(time_unit="d")
    activity_IV_infusion_low.add_output(type="piecewise_constant", name="IVTransfusion", t=[180/1440], f=[0.0083,0,])
    activity_IV_infusion_low.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_IV_infusion_low.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationIV_infusion_low = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_IV_infusion_low], time_vector=np.linspace(0, 420, 421)/1440)
    simulationIV_infusion_low.simulate()
    # IV_infusion_mid
    activity_IV_infusion_mid = sund.Activity(time_unit="d")
    activity_IV_infusion_mid.add_output(type="piecewise_constant", name="IVTransfusion", t=[180/1440], f=[0.0833,0,])
    activity_IV_infusion_mid.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_IV_infusion_mid.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationIV_infusion_mid = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_IV_infusion_mid], time_vector=np.linspace(0, 420, 421)/1440)
    simulationIV_infusion_mid.simulate()
    # IV_infusion_high
    activity_IV_infusion_high = sund.Activity(time_unit="d")
    activity_IV_infusion_high.add_output(type="piecewise_constant", name="IVTransfusion", t=[180/1440], f=[0.8333,0,])
    activity_IV_infusion_high.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_IV_infusion_high.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationIV_infusion_high = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_IV_infusion_high], time_vector=np.linspace(0, 420, 421)/1440)
    simulationIV_infusion_high.simulate()
    # SC_low
    activity_SC_low = sund.Activity(time_unit="d")
    activity_SC_low.add_state_manipulation(mode="add", name="A", t=[0], f=[0.5])
    activity_SC_low.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_SC_low.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationSC_low = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_SC_low], time_vector=np.linspace(0, 420, 421)/1440)
    simulationSC_low.simulate()
    # SC_mid
    activity_SC_mid = sund.Activity(time_unit="d")
    activity_SC_mid.add_state_manipulation(mode="add", name="A", t=[0], f=[5])
    activity_SC_mid.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_SC_mid.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationSC_mid = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_SC_mid], time_vector=np.linspace(0, 420, 421)/1440)
    simulationSC_mid.simulate()
    # SC_high
    activity_SC_high = sund.Activity(time_unit="d")
    activity_SC_high.add_state_manipulation(mode="add", name="A", t=[0], f=[50])
    activity_SC_high.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[0.34])
    activity_SC_high.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    simulationSC_high = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_SC_high], time_vector=np.linspace(0, 420, 421)/1440)
    simulationSC_high.simulate()

    PKsimulations = {
        'IV_low': simulationIV_low,
        'IV_mid': simulationIV_mid,
        'IV_high': simulationIV_high,
        'IV_infusion_low': simulationIV_infusion_low,
        'IV_infusion_mid': simulationIV_infusion_mid,
        'IV_infusion_high': simulationIV_infusion_high,
        'SC_low': simulationSC_low,
        'SC_mid': simulationSC_mid,
        'SC_high': simulationSC_high
    }

    return PKsimulations
def CalcSimulatedValues_PKstudy(PKmodel, data, parameter_values=None):

    PKsimulations = Simulate_PKstudy(PKmodel, parameter_values)

    # feature index
    idx_C1 = PKsimulations['IV_high'].feature_names.index("Plasma_Exenatide")

    # IV bolus
    t_iv_50  = np.array(data['RatPK']['timeExenatideIV'][0:12], dtype=int) # Only works when the time-vector is one step every minute.
    t_iv_5   = np.array(data['RatPK']['timeExenatideIV'][0:11], dtype=int)
    t_iv_0p5 = np.array(data['RatPK']['timeExenatideIV'][0:6], dtype=int)
    E_IV50  = PKsimulations['IV_high'].feature_values[t_iv_50, idx_C1]
    E_IV5   = PKsimulations['IV_mid'].feature_values[t_iv_5, idx_C1]
    E_IV0p5 = PKsimulations['IV_low'].feature_values[t_iv_0p5, idx_C1]

    # IV infusion
    t_ivinf_50  = np.array(data['RatPK']['timeExenatideIVtransfusion'][0:17], dtype=int)
    t_ivinf_5   = np.array(data['RatPK']['timeExenatideIVtransfusion'][0:17], dtype=int)
    t_ivinf_0p5 = np.array(data['RatPK']['timeExenatideIVtransfusion'][0:14], dtype=int)
    E_IVinfusion50  = PKsimulations['IV_infusion_high'].feature_values[t_ivinf_50, idx_C1]
    E_IVinfusion5   = PKsimulations['IV_infusion_mid'].feature_values[t_ivinf_5, idx_C1]
    E_IVinfusion0p5 = PKsimulations['IV_infusion_low'].feature_values[t_ivinf_0p5, idx_C1]

    # SC
    t_sc_50  = np.array(data['RatPK']['Exenatidestimesd'][0:12], dtype=int)
    t_sc_5   = np.array(data['RatPK']['Exenatidestimesd'][0:12], dtype=int)
    t_sc_0p5 = np.array(data['RatPK']['Exenatidestimesd'][0:12], dtype=int)
    E_SC50  = PKsimulations['SC_high'].feature_values[t_sc_50, idx_C1]
    E_SC5   = PKsimulations['SC_mid'].feature_values[t_sc_5, idx_C1]
    E_SC0p5 = PKsimulations['SC_low'].feature_values[t_sc_0p5, idx_C1]
    
    SimulatedValues = {}
    
    SimulatedValues['values'] = np.concatenate([
        E_IV50,
        E_IV5,
        E_IV0p5,
        E_IVinfusion50,
        E_IVinfusion5,
        E_IVinfusion0p5,
        E_SC50,
        E_SC5,
        E_SC0p5
    ])

    SimulatedValues['groups']  = np.concatenate([
        np.repeat("IV_50", len(np.array(data['RatPK']['ExenatideIV_50nM'][0:12], dtype=float))),
        np.repeat("IV_5", len(np.array(data['RatPK']['ExenatideIV_5nM'][0:11], dtype=float))),
        np.repeat("IV_0p5", len(np.array(data['RatPK']['ExenatideIV_0p5nM'][0:6], dtype=float))),

        np.repeat("IVinf_50", len(np.array(data['RatPK']['ExenatideIVinfusion_50nM'][0:17], dtype=float))),
        np.repeat("IVinf_5", len(np.array(data['RatPK']['ExenatideIVinfusion_5nM'][0:17], dtype=float))),
        np.repeat("IVinf_0p5", len(np.array(data['RatPK']['ExenatideIVinfusion_0p5nM'][0:14], dtype=float))),

        np.repeat("SC_50", len(np.array(data['RatPK']['Exenatidesd_50nM'][0:12], dtype=float))),
        np.repeat("SC_5", len(np.array(data['RatPK']['Exenatidesd_5nM'][0:12], dtype=float))),
        np.repeat("SC_0p5", len(np.array(data['RatPK']['Exenatidesd_0p5nM'][0:12], dtype=float)))
    ])

    return SimulatedValues

# Semaglutide
def Simulate_SemaglutidePKstudy(PKmodel, data, PKparams, TimeRes_prestudy, TimeRes_exposure):

    PKmodel.parameter_values = np.asarray(PKparams, dtype=float)
    PKmodel_state_values_reset = np.asarray(PKmodel.state_values, dtype=float).copy()

    BW_kg = 0.05                  # kg
    dose_Sema0p3   = 0.3 * BW_kg  # nmol/kg to nmol
    dose_Sema2     = 2 * BW_kg    # nmol/kg to nmol
    dose_Sema4_Q2D = 4 * BW_kg    # nmol/kg to nmol
    dose_Sema10    = 10 * BW_kg   # nmol/kg to nmol

    # ---------- Simulation time vectors ----------
    sim_t_end_regular = 22
    sim_t_end_Sema10 = 33
    sim_t_end_prestudy = 1.3

    sim_t_regular = np.linspace(0, sim_t_end_regular, int(sim_t_end_regular * 24 * TimeRes_exposure) + 1)
    sim_t_Sema10 = np.linspace(0, sim_t_end_Sema10, int(sim_t_end_Sema10 * 24 * TimeRes_exposure) + 1)
    sim_t_prestudy = np.linspace(0, sim_t_end_prestudy, int(sim_t_end_prestudy * 24 * TimeRes_prestudy) + 1)

    # ---------- Dosing times ----------
    t_Sema0p3_QD = np.arange(0, 22, 1, dtype=float)
    t_Sema2_QD   = np.arange(0, 22, 1, dtype=float)
    t_Sema4_Q2D  = np.append(np.arange(0, 22, 2, dtype=float), 21.0)
    t_Sema10_QD  = np.arange(0, 22, 1, dtype=float)

    # ---------- 0.3 nmol/kg QD for 22 days ----------
    PKmodel.state_values = PKmodel_state_values_reset.copy()
    activity_Sema0p3 = sund.Activity(time_unit="d")
    activity_Sema0p3.add_state_manipulation(mode="add", name="A", t=t_Sema0p3_QD.tolist(), f=np.repeat(dose_Sema0p3, len(t_Sema0p3_QD)).tolist())
    activity_Sema0p3.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW_kg])
    activity_Sema0p3.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[BW_kg])
    simulation_Sema0p3 = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_Sema0p3], time_vector=sim_t_regular)
    simulation_Sema0p3.simulate()

    # ---------- 2 nmol/kg QD for 22 days ----------
    PKmodel.state_values = PKmodel_state_values_reset.copy()
    activity_Sema2 = sund.Activity(time_unit="d")
    activity_Sema2.add_state_manipulation(mode="add", name="A", t=t_Sema2_QD.tolist(), f=np.repeat(dose_Sema2, len(t_Sema2_QD)).tolist())
    activity_Sema2.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW_kg])
    activity_Sema2.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[BW_kg])
    simulation_Sema2 = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_Sema2], time_vector=sim_t_regular)
    simulation_Sema2.simulate()

    # ---------- 4 nmol/kg Q2D for 22 days ----------
    PKmodel.state_values = PKmodel_state_values_reset.copy()
    activity_Sema4_Q2D = sund.Activity(time_unit="d")
    activity_Sema4_Q2D.add_state_manipulation(mode="add", name="A", t=t_Sema4_Q2D.tolist(), f=np.repeat(dose_Sema4_Q2D, len(t_Sema4_Q2D)).tolist())
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW_kg])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[BW_kg])
    simulation_Sema4_Q2D = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_Sema4_Q2D], time_vector=sim_t_regular)
    simulation_Sema4_Q2D.simulate()

    # ---------- 10 nmol/kg QD for 33 days ----------
    PKmodel.state_values = PKmodel_state_values_reset.copy()
    activity_Sema10 = sund.Activity(time_unit="d")
    activity_Sema10.add_state_manipulation(mode="add", name="A", t=t_Sema10_QD.tolist(), f=np.repeat(dose_Sema10, len(t_Sema10_QD)).tolist())
    activity_Sema10.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW_kg])
    activity_Sema10.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[BW_kg])
    simulation_Sema10 = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_Sema10], time_vector=sim_t_Sema10)
    simulation_Sema10.simulate()

    # ---------- Prestudy 10 nmol/kg single dose ----------
    PKmodel.state_values = PKmodel_state_values_reset.copy()
    activity_Sema10_prestudy = sund.Activity(time_unit="d")
    activity_Sema10_prestudy.add_state_manipulation(mode="add", name="A", t=[0], f=[dose_Sema10])
    simulation_Sema10_prestudy = sund.Simulation(time_unit="d", models=[PKmodel], activities=[activity_Sema10_prestudy], time_vector=sim_t_prestudy)
    simulation_Sema10_prestudy.simulate()

    PKmodel.state_values = PKmodel_state_values_reset.copy()

    PKsimulations = {
        "Sema0p3": simulation_Sema0p3,
        "Sema2": simulation_Sema2,
        "Sema4_Q2D": simulation_Sema4_Q2D,
        "Sema10": simulation_Sema10,
        "Sema10_prestudy": simulation_Sema10_prestudy
    }

    return PKsimulations
def CalcSimulatedValues_SemaglutidePKstudy(model, data, parameter_values=None):

    TimeRes_prestudy = 60 # 60 points per hour
    TimeRes_exposure = 2 # 2 points per hour

    PKsimulations = Simulate_SemaglutidePKstudy(model, data, parameter_values, TimeRes_prestudy, TimeRes_exposure)

    idx_C1 = PKsimulations["Sema2"].feature_names.index("Plasma_Semaglutide")

    # ---------- Sema2 exposure: one simulation point every hour ----------
    t_Sema2 = np.array(data["AZstudy"]["timeExposure"][0:1], dtype=float)
    t_Sema2_idx = np.round(t_Sema2 * 24 * TimeRes_exposure).astype(int)
    Sema2Exposure = PKsimulations["Sema2"].feature_values[t_Sema2_idx, idx_C1]
    # ---------- Sema4 Q2D exposure: one simulation point every hour ----------
    t_Sema4_Q2D = np.array(data["AZstudy"]["timeExposure"][0:1], dtype=float)
    t_Sema4_Q2D_idx = np.round(t_Sema4_Q2D * 24 * TimeRes_exposure).astype(int)
    Sema4_Q2DExposure = PKsimulations["Sema4_Q2D"].feature_values[t_Sema4_Q2D_idx, idx_C1]
    # ---------- Sema10 exposure: one simulation point every hour ----------
    t_Sema10 = np.array(data["AZstudy"]["timeExposureLateEnd"][0:1], dtype=float)
    t_Sema10_idx = np.round(t_Sema10 * 24 * TimeRes_exposure).astype(int)
    Sema10Exposure = PKsimulations["Sema10"].feature_values[t_Sema10_idx, idx_C1]

    # ---------- Sema10 prestudy PK: one simulation point every 5 minutes ----------
    t_Sema10_prestudy = np.array(data["AZstudy"]["time_prestudy"][0:16], dtype=float)
    t_Sema10_prestudy_idx = np.round(t_Sema10_prestudy * 24 * TimeRes_prestudy).astype(int)
    Sema10PK_prestudy = PKsimulations["Sema10_prestudy"].feature_values[t_Sema10_prestudy_idx, idx_C1]

    SimulatedValues = {}

    SimulatedValues["values"] = np.concatenate([
        Sema2Exposure,
        Sema4_Q2DExposure,
        Sema10Exposure,
        Sema10PK_prestudy
    ])
    SimulatedValues["groups"] = np.concatenate([
        np.repeat("Sema2Exposure", len(np.array(data["AZstudy"]["Sema2Exposure"][0:1], dtype=float))),
        np.repeat("Sema4_Q2DExposure", len(np.array(data["AZstudy"]["Sema4_Q2DExposure"][0:1], dtype=float))),
        np.repeat("Sema10Exposure", len(np.array(data["AZstudy"]["Sema10Exposure"][0:1], dtype=float))),
        np.repeat("Sema10PK_prestudy", len(np.array(data["AZstudy"]["Sema10PK_prestudy"][0:16], dtype=float)))
    ])

    return SimulatedValues


# - ------- Simulate PD study and export the simulated values at the datapoints -----
def Simulate_PDstudy(model_PK, model_GI,GIparams, PKparams):
    UnitConvertion       = 0.001*0.34;   #pMol/kg/min to nMol/min where 0.34 is rat kg
   
    model_GI.parameter_values   = np.asarray(GIparams, dtype=float)
    model_PK.parameter_values = np.asarray(PKparams, dtype=float)

    # Compute steady state for BOTH models together
    simulation_ss = sund.Simulation(time_unit='d', models=[model_PK, model_GI], time_vector=np.linspace(0, 100, 2))
    simulation_ss.simulate()
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    model_PK = assign_state_values_from_simulation(model_PK, simulation_ss)
    GI_ss_states = np.copy(model_GI.state_values)
    PK_ss_states = np.copy(model_PK.state_values)
    
    # IV_infusion_0
    model_GI.state_values = np.copy(GI_ss_states)
    model_PK.state_values = np.copy(PK_ss_states)
    activity_IV_infusion_0 = sund.Activity(time_unit="d")
    activity_IV_infusion_0.add_output(type="piecewise_constant", name="IVTransfusion", t=[300/1440], f=[0*UnitConvertion,0,])
    activity_IV_infusion_0.add_state_manipulation(mode="add", name="G_SC", t=[30/1440], f=[1.938])
    simulationIV_infusion_0 = sund.Simulation(time_unit="d", models=[model_PK,model_GI], activities=[activity_IV_infusion_0], time_vector=np.linspace(0, 120, 121)/1440)
    simulationIV_infusion_0.simulate()
    # IV_infusion_3
    model_GI.state_values = np.copy(GI_ss_states)
    model_PK.state_values = np.copy(PK_ss_states)
    activity_IV_infusion_3 = sund.Activity(time_unit="d")
    activity_IV_infusion_3.add_output(type="piecewise_constant", name="IVTransfusion", t=[300/1440], f=[3*UnitConvertion,0,])
    activity_IV_infusion_3.add_state_manipulation(mode="add", name="G_SC", t=[30/1440], f=[1.938])
    simulationIV_infusion_3 = sund.Simulation(time_unit="d", models=[model_PK,model_GI], activities=[activity_IV_infusion_3], time_vector=np.linspace(0, 120  , 121)/1440)
    simulationIV_infusion_3.simulate()
    # IV_infusion_30
    model_GI.state_values = np.copy(GI_ss_states)
    model_PK.state_values = np.copy(PK_ss_states)
    activity_IV_infusion_30 = sund.Activity(time_unit="d")
    activity_IV_infusion_30.add_output(type="piecewise_constant", name="IVTransfusion", t=[300/1440], f=[30*UnitConvertion,0,])
    activity_IV_infusion_30.add_state_manipulation(mode="add", name="G_SC", t=[30/1440], f=[1.938])
    simulationIV_infusion_30 = sund.Simulation(time_unit="d", models=[model_PK,model_GI], activities=[activity_IV_infusion_30], time_vector=np.linspace(0, 120  , 121)/1440)
    simulationIV_infusion_30.simulate()
    # IV_infusion_300
    model_GI.state_values = np.copy(GI_ss_states)
    model_PK.state_values = np.copy(PK_ss_states)
    activity_IV_infusion_300 = sund.Activity(time_unit="d")
    activity_IV_infusion_300.add_output(type="piecewise_constant", name="IVTransfusion", t=[300/1440], f=[300*UnitConvertion,0,])
    activity_IV_infusion_300.add_state_manipulation(mode="add", name="G_SC", t=[30/1440], f=[1.938])
    simulationIV_infusion_300 = sund.Simulation(time_unit="d", models=[model_PK,model_GI], activities=[activity_IV_infusion_300], time_vector=np.linspace(0, 120  , 121)/1440)
    simulationIV_infusion_300.simulate()
    # IV_infusion_3000
    model_GI.state_values = np.copy(GI_ss_states)
    model_PK.state_values = np.copy(PK_ss_states)
    activity_IV_infusion_3000 = sund.Activity(time_unit="d")
    activity_IV_infusion_3000.add_output(type="piecewise_constant", name="IVTransfusion", t=[300/1440], f=[3000*UnitConvertion,0,])
    activity_IV_infusion_3000.add_state_manipulation(mode="add", name="G_SC", t=[30/1440], f=[1.938])
    simulationIV_infusion_3000 = sund.Simulation(time_unit="d", models=[model_PK,model_GI], activities=[activity_IV_infusion_3000], time_vector=np.linspace(0, 120  , 121)/1440)
    simulationIV_infusion_3000.simulate()

    PDsimulations = {
        'simulationIV_infusion_0': simulationIV_infusion_0,
        'simulationIV_infusion_3': simulationIV_infusion_3,
        'simulationIV_infusion_30': simulationIV_infusion_30,
        'simulationIV_infusion_300': simulationIV_infusion_300,
        'simulationIV_infusion_3000': simulationIV_infusion_3000,
    }

    return PDsimulations
def CalcSimulatedValues_PDstudy(model_PK, model_GI, data, GIparams, PKparams):

    PDsimulations = Simulate_PDstudy(model_PK, model_GI, GIparams, PKparams)

    # Feature indices (fixed as requested)
    idx_G = PDsimulations['simulationIV_infusion_0'].feature_names.index("Plasma_Glucose")
    idx_I = PDsimulations['simulationIV_infusion_0'].feature_names.index("Plasma_Insulin")


    # Time indices (data is t=0 at event, simulation event at t=30)
    t_pd = np.array(data['RatPD']['Time'][0:12], dtype=int)
    t_sim = t_pd + 30   # <-- Shift to have the GTT at t=0

    # GLUCOSE
    G_0    = PDsimulations['simulationIV_infusion_0'].feature_values[t_sim, idx_G]
    G_3    = PDsimulations['simulationIV_infusion_3'].feature_values[t_sim, idx_G]
    G_30   = PDsimulations['simulationIV_infusion_30'].feature_values[t_sim, idx_G]
    G_300  = PDsimulations['simulationIV_infusion_300'].feature_values[t_sim, idx_G]
    G_3000 = PDsimulations['simulationIV_infusion_3000'].feature_values[t_sim, idx_G]

    # INSULIN
    I_0    = PDsimulations['simulationIV_infusion_0'].feature_values[t_sim, idx_I]
    I_3    = PDsimulations['simulationIV_infusion_3'].feature_values[t_sim, idx_I]
    I_30   = PDsimulations['simulationIV_infusion_30'].feature_values[t_sim, idx_I]
    I_300  = PDsimulations['simulationIV_infusion_300'].feature_values[t_sim, idx_I]
    I_3000 = PDsimulations['simulationIV_infusion_3000'].feature_values[t_sim, idx_I]


    # OUTPUT
    SimulatedValues = {}

    SimulatedValues['values'] = np.concatenate([
        G_0, G_3, G_30, G_300, G_3000,
        I_0, I_3, I_30, I_300, I_3000
    ])

    SimulatedValues['groups'] = np.concatenate([
        np.repeat("glucose_0", 12),
        np.repeat("glucose_3", 12),
        np.repeat("glucose_30", 12),
        np.repeat("glucose_300", 12),
        np.repeat("glucose_3000", 12),

        np.repeat("insulin_0", 12),
        np.repeat("insulin_3", 12),
        np.repeat("insulin_30", 12),
        np.repeat("insulin_300", 12),
        np.repeat("insulin_3000", 12)
    ])

    SimulatedValues['time'] = np.concatenate([
        t_pd, t_pd, t_pd, t_pd, t_pd,
        t_pd, t_pd, t_pd, t_pd, t_pd
    ])

    return SimulatedValues


# --------- Appetite ---------
def Simulate_HallAppetiteStudy(Model_Appetite,model_BW, data, Appetite_params, BW_params, SimulationRes, UseModelAppetite):

    lambda_chow = BW_params[-2]
    lambda_HF   = BW_params[-1]
                       
    Model_Appetite.parameter_values = np.asarray(Appetite_params, dtype=float) 
    model_BW.parameter_values       = np.asarray(BW_params[:-2], dtype=float)

    # ---------- Lookup state indices ----------
    idx_state_ffm = model_BW.state_names.index("FFM")
    idx_state_fm = model_BW.state_names.index("FM")
    idx_state_lambda_delay = model_BW.state_names.index("lambda_delay")

    # ---------- Helper to initialise states ----------
    def baseline_state(ffm0, fm0, lambda_vehicle, state0):
        state0 = np.array(model_BW.state_values, dtype=float, copy=True)
        state0[idx_state_ffm]          = ffm0
        state0[idx_state_fm]           = fm0
        state0[idx_state_lambda_delay] = lambda_vehicle
        return state0

    t_end       = 232 # days
    sim_t       = np.linspace(0, t_end, int(t_end * SimulationRes) + 1)
    FatRatio_LF = 0.12
    FatRatio_HF = 0.64

    # ss_Control
    control_bw0 = float(data["Hall_BW"]["ControlBW_mean"][0])
    control_fm0 = float(data["Hall_BW"]["ControlFM_mean"][0])
    control_ffm0 = control_bw0 - control_fm0
    activity_ss = sund.Activity(time_unit="d")
    activity_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_LF])
    activity_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_chow])
    activity_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[control_fm0])
    activity_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[control_ffm0])
    activity_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0]) # 13 weeks old
    # Control
    activity_control = sund.Activity(time_unit="d")
    activity_control.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_LF])
    activity_control.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_chow])
    activity_control.add_output(type="piecewise_constant", name="FMinit", t=[], f=[control_fm0])
    activity_control.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[control_ffm0])
    activity_control.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 13 weeks old

    #  4HF-C 
    HF4_C_bw0 = float(data["Hall_BW"]["HF4_CBW_mean"][0])
    HF4_C_fm0 = float(data["Hall_BW"]["HF4_CFM_mean"][0])
    HF4_C_ffm0 = HF4_C_bw0 - HF4_C_fm0
    activity_4HF_C = sund.Activity(time_unit="d")
    activity_4HF_C.add_output(type="piecewise_constant", name="lambda", t=[28], f=[lambda_HF, lambda_chow])
    activity_4HF_C.add_output(type="piecewise_constant", name="FMinit", t=[], f=[HF4_C_fm0])
    activity_4HF_C.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[HF4_C_ffm0])
    activity_4HF_C.add_output(type="piecewise_constant", name="FatRatio", t=[28], f=[FatRatio_HF, FatRatio_LF])
    activity_4HF_C.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 13 weeks old
    
    # HF7_C
    HF7_C_bw0 = float(data["Hall_BW"]["HF7_CBW_mean"][0])
    HF7_C_fm0 = float(data["Hall_BW"]["HF7_CFM_mean"][0])
    HF7_C_ffm0 = HF7_C_bw0 - HF7_C_fm0
    activity_7HF_C = sund.Activity(time_unit="d")
    activity_7HF_C.add_output(type="piecewise_constant", name="lambda", t=[49], f=[lambda_HF, lambda_chow])
    activity_7HF_C.add_output(type="piecewise_constant", name="FMinit", t=[], f=[HF7_C_fm0])
    activity_7HF_C.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[HF7_C_ffm0])
    activity_7HF_C.add_output(type="piecewise_constant", name="FatRatio", t=[49], f=[FatRatio_HF, FatRatio_LF])
    activity_7HF_C.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 13 weeks old

    # HF20_C
    HF20_C_bw0 = float(data["Hall_BW"]["HF20_CBW_mean"][0])
    HF20_C_fm0 = float(data["Hall_BW"]["HF20_CFM_mean"][0])
    HF20_C_ffm0 = HF20_C_bw0 - HF20_C_fm0
    activity_20HF_C = sund.Activity(time_unit="d")
    activity_20HF_C.add_output(type="piecewise_constant", name="lambda", t=[140], f=[lambda_HF, lambda_chow])
    activity_20HF_C.add_output(type="piecewise_constant", name="FMinit", t=[], f=[HF20_C_fm0])
    activity_20HF_C.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[HF20_C_ffm0])
    activity_20HF_C.add_output(type="piecewise_constant", name="FatRatio", t=[140], f=[FatRatio_HF, FatRatio_LF])
    activity_20HF_C.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 13 weeks old

    # HF_C_HF_C
    HF_C_HF_C_bw0 = float(data["Hall_BW"]["HF_C_HF_CBW_mean"][0])
    HF_C_HF_C_fm0 = float(data["Hall_BW"]["HF_C_HF_CFM_mean"][0])
    HF_C_HF_C_ffm0 = HF_C_HF_C_bw0 - HF_C_HF_C_fm0
    activity_HF_C_HF_C = sund.Activity(time_unit="d")
    activity_HF_C_HF_C.add_output(type="piecewise_constant", name="lambda", t=[49, 70, 140], f=[lambda_HF, lambda_chow, lambda_HF, lambda_chow])
    activity_HF_C_HF_C.add_output(type="piecewise_constant", name="FMinit", t=[], f=[HF_C_HF_C_fm0])
    activity_HF_C_HF_C.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[HF_C_HF_C_ffm0])
    activity_HF_C_HF_C.add_output(type="piecewise_constant", name="FatRatio", t=[49, 70, 140], f=[FatRatio_HF, FatRatio_LF, FatRatio_HF, FatRatio_LF])
    activity_HF_C_HF_C.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 13 weeks old

    if UseModelAppetite:
        # Compute steady state for BOTH models together
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,model_BW.state_values)
        simulation_ss = sund.Simulation(time_unit='d',models= [Model_Appetite,model_BW], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
        simulation_ss.simulate()
        Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
        model_BW       = assign_state_values_from_simulation(model_BW, simulation_ss)
        Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
        Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
        Model_Appetite_Reset = np.array(Model_Appetite.state_values, copy=True)
        Model_BW_Reset       = np.array(model_BW.state_values      , copy=True)

        # ---------- Simulate each group ----------
        Model_Appetite.state_values = Model_Appetite_Reset.copy()
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_control = sund.Simulation(time_unit="d", models=[Model_Appetite,model_BW], activities=[activity_control], time_vector=sim_t)
        sim_control.simulate()

        Model_Appetite.state_values = Model_Appetite_Reset.copy()
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_4HF_C = sund.Simulation(time_unit="d", models=[Model_Appetite,model_BW], activities=[activity_4HF_C], time_vector=sim_t)
        sim_4HF_C.simulate()

        Model_Appetite.state_values = Model_Appetite_Reset.copy()
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_7HF_C = sund.Simulation(time_unit="d", models=[Model_Appetite,model_BW], activities=[activity_7HF_C], time_vector=sim_t)
        sim_7HF_C.simulate()

        Model_Appetite.state_values = Model_Appetite_Reset.copy()
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_20HF_C = sund.Simulation(time_unit="d", models=[Model_Appetite,model_BW], activities=[activity_20HF_C], time_vector=sim_t)
        sim_20HF_C.simulate()

        Model_Appetite.state_values = Model_Appetite_Reset.copy()
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_HF_C_HF_C = sund.Simulation(time_unit="d", models=[Model_Appetite,model_BW], activities=[activity_HF_C_HF_C], time_vector=sim_t)
        sim_HF_C_HF_C.simulate()

    else:
        # ---------- Build EI activities because we arent using the appetite model ----------
        EI_control = np.array(data["Hall_EI"]["Control"], dtype=float)
        EI_7HF_C = np.array(data["Hall_EI"]["7HF-C"], dtype=float)
        EI_HF_C_HF_C = np.array(data["Hall_EI"]["HF-C-HF-C"], dtype=float)
        EI_20HF_C = np.array(data["Hall_EI"]["20HF-C"], dtype=float)
        EI_4HF_C = np.array(data["Hall_EI"]["4HF-C"], dtype=float)
        ei_days = np.array(data["Hall_EI"]["Time"], dtype=float)
        activity_control.add_output(type="piecewise_constant", name="EI", t=ei_days[1:], f=EI_control)
        activity_7HF_C.add_output(type="piecewise_constant", name="EI", t=ei_days[1:], f=EI_7HF_C)
        activity_HF_C_HF_C.add_output(type="piecewise_constant", name="EI", t=ei_days[1:], f=EI_HF_C_HF_C)
        activity_20HF_C.add_output(type="piecewise_constant", name="EI", t=ei_days[1:], f=EI_20HF_C)
        activity_4HF_C.add_output(type="piecewise_constant", name="EI", t=ei_days[1:], f=EI_4HF_C)
        # ---------- Simulate each group ----------
        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_control = sund.Simulation(time_unit="d", models=[model_BW], activities=[activity_control], time_vector=sim_t)
        sim_control.simulate()

        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_4HF_C = sund.Simulation(time_unit="d", models=[model_BW], activities=[activity_4HF_C], time_vector=sim_t)
        sim_4HF_C.simulate()

        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_7HF_C = sund.Simulation(time_unit="d", models=[model_BW], activities=[activity_7HF_C], time_vector=sim_t)
        sim_7HF_C.simulate()

        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_20HF_C = sund.Simulation(time_unit="d", models=[model_BW], activities=[activity_20HF_C], time_vector=sim_t)
        sim_20HF_C.simulate()

        model_BW.state_values = baseline_state(control_ffm0, control_fm0, lambda_chow,Model_BW_Reset)
        sim_HF_C_HF_C = sund.Simulation(time_unit="d", models=[model_BW], activities=[activity_HF_C_HF_C], time_vector=sim_t)
        sim_HF_C_HF_C.simulate()


    AppetiteSimulations = {"control": sim_control, "4HF_C": sim_4HF_C, "7HF_C": sim_7HF_C, "20HF_C": sim_20HF_C, "HF_C_HF_C": sim_HF_C_HF_C}
    return AppetiteSimulations
def CalcSimulatedValues_HallEI_ACstudy(Model_Appetite,model_BW, data, Appetite_params, BW_params, SimulationRes):

    sims = Simulate_HallAppetiteStudy(Model_Appetite,model_BW, data, Appetite_params, BW_params, SimulationRes, True)
    
    t_EI_data = np.array(data["Hall_EI"]["Time"], dtype=int)
    t_EI_idx = t_EI_data * SimulationRes
    t_BW = np.array(data["Hall_BW"]["Time"], dtype=int)
    t_BW_idx = t_BW * SimulationRes

    # -- Model index --
    idx_ei_ac = sims["control"].feature_names.index("EI_AC")
    idx_bw    = sims["control"].feature_names.index("Body_weight")
    idx_fm    = sims["control"].feature_names.index("FM")
    idx_eating= sims["control"].feature_names.index("Eating")

    # ---------- Average meals/day, meal-size, and daily energy intakes ----------
    meals_control  , meal_size_control  , daily_ei_control  , night_meals_control = average_meals_per_day_and_meal_size(sims["control"].feature_values[:, idx_ei_ac], sims["control"].feature_values[:, idx_eating], sims["control"].time_vector, SimulationRes)
    meals_4HF_C    , meal_size_4HF_C    , daily_ei_4HF_C    , night_meals_4HF_C = average_meals_per_day_and_meal_size(sims["4HF_C"].feature_values[:, idx_ei_ac], sims["4HF_C"].feature_values[:, idx_eating], sims["4HF_C"].time_vector, SimulationRes)
    meals_7HF_C    , meal_size_7HF_C    , daily_ei_7HF_C    , night_meals_7HF_C = average_meals_per_day_and_meal_size(sims["7HF_C"].feature_values[:, idx_ei_ac], sims["7HF_C"].feature_values[:, idx_eating], sims["7HF_C"].time_vector, SimulationRes)
    meals_HF_C_HF_C, meal_size_HF_C_HF_C, daily_ei_HF_C_HF_C, night_meals_HF_C_HF_C = average_meals_per_day_and_meal_size(sims["HF_C_HF_C"].feature_values[:, idx_ei_ac], sims["HF_C_HF_C"].feature_values[:, idx_eating], sims["HF_C_HF_C"].time_vector, SimulationRes)
    meals_20HF_C   , meal_size_20HF_C   , daily_ei_20HF_C   , night_meals_20HF_C = average_meals_per_day_and_meal_size(sims["20HF_C"].feature_values[:, idx_ei_ac], sims["20HF_C"].feature_values[:, idx_eating], sims["20HF_C"].time_vector, SimulationRes)
    
    # -- EI study
    EI_daily_control   = daily_ei_control[t_EI_data]
    EI_daily_4HF_C     = daily_ei_4HF_C[t_EI_data]
    EI_daily_7HF_C     = daily_ei_7HF_C[t_EI_data]
    EI_daily_20HF_C    = daily_ei_20HF_C[t_EI_data]
    EI_daily_HF_C_HF_C = daily_ei_HF_C_HF_C[t_EI_data]
    # -- BW study
    # ----- Control -----
    BW_control = sims["control"].feature_values[t_BW_idx, idx_bw]
    FM_control = sims["control"].feature_values[t_BW_idx, idx_fm]
    # ----- 7HF-C -----
    BW_7HF_C = sims["7HF_C"].feature_values[t_BW_idx, idx_bw]
    FM_7HF_C = sims["7HF_C"].feature_values[t_BW_idx, idx_fm]
    # ----- HF-C-HF-C -----
    BW_HF_C_HF_C = sims["HF_C_HF_C"].feature_values[t_BW_idx, idx_bw]
    FM_HF_C_HF_C = sims["HF_C_HF_C"].feature_values[t_BW_idx, idx_fm]
    # ----- 20HF-C -----
    BW_20HF_C = sims["20HF_C"].feature_values[t_BW_idx, idx_bw]
    FM_20HF_C = sims["20HF_C"].feature_values[t_BW_idx, idx_fm]
    # ----- 4HF-C -----
    BW_4HF_C = sims["4HF_C"].feature_values[t_BW_idx, idx_bw]
    FM_4HF_C = sims["4HF_C"].feature_values[t_BW_idx, idx_fm]

    SimulatedValues = {}
    SimulatedValues["values"] = np.concatenate([
        EI_daily_control,
        EI_daily_7HF_C,
        EI_daily_HF_C_HF_C,
        EI_daily_20HF_C,
        EI_daily_4HF_C,
        BW_control, FM_control,
        BW_7HF_C, FM_7HF_C,
        BW_HF_C_HF_C, FM_HF_C_HF_C,
        BW_20HF_C, FM_20HF_C,
        BW_4HF_C, FM_4HF_C
    ])
    SimulatedValues["AverageAmountOfMealsPerDay"] = np.array([meals_control       , meals_20HF_C])
    SimulatedValues["AverageMealSize"]            = np.array([meal_size_control   , meal_size_20HF_C])
    SimulatedValues["MealsAtNight"]               = np.array([night_meals_control , night_meals_20HF_C])
    SimulatedValues["groups"] = np.concatenate([
        np.repeat("Control", len(t_EI_idx))    ,
        np.repeat("7HF_C", len(t_EI_idx))      ,
        np.repeat("HF_C_HF_C", len(t_EI_idx))  ,
        np.repeat("20HF_C", len(t_EI_idx))     ,
        np.repeat("4HF_C", len(t_EI_idx))      ,
        np.repeat("BW_control", len(t_BW_idx))  , np.repeat("FM_control", len(t_BW_idx)),
        np.repeat("BW_7HF_C", len(t_BW_idx))    , np.repeat("FM_7HF_C", len(t_BW_idx)),
        np.repeat("BW_HF_C_HF_C", len(t_BW_idx)), np.repeat("FM_HF_C_HF_C", len(t_BW_idx)),
        np.repeat("BW_20HF_C", len(t_BW_idx))   , np.repeat("FM_20HF_C", len(t_BW_idx)),
        np.repeat("BW_4HF_C", len(t_BW_idx))    , np.repeat("FM_4HF_C", len(t_BW_idx))
    ])
    SimulatedValues["time"] = np.concatenate([
        t_EI_data,
        t_EI_data,
        t_EI_data,
        t_EI_data,
        t_EI_data,
        t_BW, t_BW,
        t_BW, t_BW,
        t_BW, t_BW,
        t_BW, t_BW,
        t_BW, t_BW
    ])

    return SimulatedValues

# -----  Mack studies -----

# Study 1
def Simulate_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, PKParams, IC50, SimulatePK):

    Model_Appetite.parameter_values = np.asarray(AppetiteParams, dtype=float)

    t_data = np.array(data["MackStudy1"]["Time"][0:7], dtype=float)
    t_end = float(np.max(t_data)) + 2/3

    # 5-minute resolution over the acute experiment
    sim_t = np.linspace(0, t_end, int(t_end * 24 * 12) + 1)

    dose_ug_per_kg = 2.4
    BW = 0.027
    MW_exenatide_ug_per_nmol = 4.1866
    A0_nmol = dose_ug_per_kg * BW / MW_exenatide_ug_per_nmol

    # ss
    activity_ss = sund.Activity(time_unit="d")
    activity_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.17]) # Chow diet LM-485 Teklad
    activity_ss.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW])
    activity_ss.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    activity_ss.add_output(type="piecewise_constant", name="FM", t=[], f=[0.1 * BW * 1000])
    activity_ss.add_output(type="piecewise_constant", name="AllowedToEat", t=[2/3], f=[0, 1])
    activity_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0])  # Assuming 10 weeks of age at the start of the study (declared in study as adult NIH mouse)

    # Vehicle
    activity_vehicle = sund.Activity(time_unit="d")
    activity_vehicle.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.17]) # Chow diet LM-485 Teklad
    activity_vehicle.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW])
    activity_vehicle.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
    activity_vehicle.add_output(type="piecewise_constant", name="FM", t=[], f=[0.1 * BW * 1000])
    activity_vehicle.add_output(type="piecewise_constant", name="AllowedToEat", t=[2/3], f=[0, 1])
    activity_vehicle.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7])

    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
    simulation_ss.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_delay")] = 10 # standrd diet (cEI) from the Hall et al. model.This is used because we dont know what they ate prior of the study
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    StateReset_appetite = np.asarray(Model_Appetite.state_values, dtype=float).copy()

    if SimulatePK:
        model_PK.parameter_values = np.asarray(PKParams, dtype=float)
        Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIC50")] = IC50
        Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIMax")] = 0.75

        activity_exenatide = sund.Activity(time_unit="d")
        activity_exenatide.add_state_manipulation(mode="add", name="A", t=[2/3], f=[A0_nmol])
        activity_exenatide.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.15]) # Chow diet LM-485 Teklad
        activity_exenatide.add_output(type="piecewise_constant", name="BW_kg", t=[], f=[BW])
        activity_exenatide.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.34])
        activity_exenatide.add_output(type="piecewise_constant", name="FM", t=[], f=[0.1 * BW * 1000])
        activity_exenatide.add_output(type="piecewise_constant", name="AllowedToEat", t=[2/3], f=[0, 1])
        activity_exenatide.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # Assuming 10 weeks of age at the start of the study (declared in study as adult NIH mouse)

        StateReset_PK = np.asarray(model_PK.state_values, dtype=float).copy()

        Model_Appetite.state_values = StateReset_appetite.copy()
        model_PK.state_values = StateReset_PK.copy()
        sim_vehicle = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK], activities=[activity_vehicle], time_vector=sim_t)
        sim_vehicle.simulate()

        Model_Appetite.state_values = StateReset_appetite.copy()
        model_PK.state_values = StateReset_PK.copy()
        sim_exenatide = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK], activities=[activity_exenatide], time_vector=sim_t)
        sim_exenatide.simulate()

        MackSimulations = {"vehicle": sim_vehicle, "exenatide": sim_exenatide}
    else:
        Model_Appetite.state_values = StateReset_appetite.copy()
        sim_vehicle = sund.Simulation(time_unit="d", models=[Model_Appetite], activities=[activity_vehicle], time_vector=sim_t)
        sim_vehicle.simulate()

        MackSimulations = {"vehicle": sim_vehicle}

    return MackSimulations
def CalcSimulatedValues_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, PKparams, IC50, SimulatePK):

    MackSimulations = Simulate_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, PKparams, IC50, SimulatePK)

    idx_ei_ac = MackSimulations["vehicle"].feature_names.index("EI_AC")
    t_data = np.array(data["MackStudy1"]["Time"][0:7], dtype=float)
    t_sim = t_data + 2/3
    EI_vehicle = np.interp(t_sim, MackSimulations["vehicle"].time_vector, MackSimulations["vehicle"].feature_values[:, idx_ei_ac])
    SimulatedValues = {}
    if SimulatePK:
        EI_exenatide = np.interp(t_sim, MackSimulations["exenatide"].time_vector, MackSimulations["exenatide"].feature_values[:, idx_ei_ac])
        SimulatedValues["values"] = np.concatenate([EI_vehicle, EI_exenatide])
        SimulatedValues["groups"] = np.concatenate([np.repeat("vehicle", len(t_data)), np.repeat("exenatide", len(t_data))])
        SimulatedValues["time"]   = np.concatenate([t_data, t_data])
    else:
        SimulatedValues["values"] = EI_vehicle
        SimulatedValues["groups"] = np.repeat("vehicle", len(t_data))
        SimulatedValues["time"] = t_data

    return SimulatedValues
def Uncertainty_MackStudy1(Model_Appetite, model_PK, data, AppetiteParams, Best_PKParams, IC50, SimulatePK):
    
    first = True
    for param_vector in AppetiteParams:

        full_param = np.asarray(param_vector[:-1], dtype=float)

        try:
            MackSimStudy1 = Simulate_MackStudy1(Model_Appetite, model_PK, data, full_param, Best_PKParams, IC50, SimulatePK)
        except Exception:
            continue

        idx_ei = MackSimStudy1["vehicle"].feature_names.index("EI_AC")
        idx_Appetite = MackSimStudy1["vehicle"].feature_names.index("Appetite")
        idx_EI_appetite = MackSimStudy1["vehicle"].feature_names.index("EI_appetite")

        # ----- vehicle -----
        EI_vehicle_new          = MackSimStudy1["vehicle"].feature_values[:, idx_ei]
        Appetite_vehicle_new    = MackSimStudy1["vehicle"].feature_values[:, idx_Appetite]
        EI_appetite_vehicle_new = MackSimStudy1["vehicle"].feature_values[:, idx_EI_appetite]/MackSimStudy1["vehicle"].feature_values[0, idx_EI_appetite]
        # ----- exenatide -----
        if SimulatePK:
            EI_exenatide_new          = MackSimStudy1["exenatide"].feature_values[:, idx_ei]
            Appetite_exenatide_new    = MackSimStudy1["exenatide"].feature_values[:, idx_Appetite]
            EI_appetite_exenatide_new = MackSimStudy1["exenatide"].feature_values[:, idx_EI_appetite]/MackSimStudy1["exenatide"].feature_values[0, idx_EI_appetite]
        if first:
            best_sim = MackSimStudy1
            # ----- vehicle -----
            EI_vehicle_min = EI_vehicle_new.copy()
            EI_vehicle_max = EI_vehicle_new.copy()
            Appetite_vehicle_min = Appetite_vehicle_new.copy()
            Appetite_vehicle_max = Appetite_vehicle_new.copy()
            EI_appetite_vehicle_min = EI_appetite_vehicle_new.copy()
            EI_appetite_vehicle_max = EI_appetite_vehicle_new.copy()
            # ----- exenatide -----
            if SimulatePK:
                EI_exenatide_min = EI_exenatide_new.copy()
                EI_exenatide_max = EI_exenatide_new.copy()
                Appetite_exenatide_min = Appetite_exenatide_new.copy()
                Appetite_exenatide_max = Appetite_exenatide_new.copy()
                EI_appetite_exenatide_min = EI_appetite_exenatide_new.copy()
                EI_appetite_exenatide_max = EI_appetite_exenatide_new.copy()
            first = False
        else:
            # ----- vehicle -----
            EI_vehicle_min = np.minimum(EI_vehicle_min, EI_vehicle_new)
            EI_vehicle_max = np.maximum(EI_vehicle_max, EI_vehicle_new)
            Appetite_vehicle_min = np.minimum(Appetite_vehicle_min, Appetite_vehicle_new)
            Appetite_vehicle_max = np.maximum(Appetite_vehicle_max, Appetite_vehicle_new)
            EI_appetite_vehicle_min = np.minimum(EI_appetite_vehicle_min, EI_appetite_vehicle_new)
            EI_appetite_vehicle_max = np.maximum(EI_appetite_vehicle_max, EI_appetite_vehicle_new)
            # ----- exenatide -----
            if SimulatePK:
                EI_exenatide_min = np.minimum(EI_exenatide_min, EI_exenatide_new)
                EI_exenatide_max = np.maximum(EI_exenatide_max, EI_exenatide_new)
                Appetite_exenatide_min = np.minimum(Appetite_exenatide_min, Appetite_exenatide_new)
                Appetite_exenatide_max = np.maximum(Appetite_exenatide_max, Appetite_exenatide_new)
                EI_appetite_exenatide_min = np.minimum(EI_appetite_exenatide_min, EI_appetite_exenatide_new)
                EI_appetite_exenatide_max = np.maximum(EI_appetite_exenatide_max, EI_appetite_exenatide_new)
    Output = {
        "best_sim": best_sim,
        "EI_vehicle_min": EI_vehicle_min,
        "EI_vehicle_max": EI_vehicle_max,
        "Appetite_vehicle_min": Appetite_vehicle_min,
        "Appetite_vehicle_max": Appetite_vehicle_max,
        "EI_appetite_vehicle_min": EI_appetite_vehicle_min,
        "EI_appetite_vehicle_max": EI_appetite_vehicle_max
    }
    if SimulatePK:
        Output["EI_exenatide_min"] = EI_exenatide_min
        Output["EI_exenatide_max"] = EI_exenatide_max
        Output["Appetite_exenatide_min"] = Appetite_exenatide_min
        Output["Appetite_exenatide_max"] = Appetite_exenatide_max
        Output["EI_appetite_exenatide_min"] = EI_appetite_exenatide_min
        Output["EI_appetite_exenatide_max"] = EI_appetite_exenatide_max

    return Output

# Study 2
def Simulate_MackStudy2(Model_Appetite, model_PK, model_GI, model_BW, data, AppetiteParams, PKParams, GIparams, BWparams,lambdaNewDiet,Recovery,IC50):

    Model_Appetite.parameter_values = np.asarray(AppetiteParams, dtype=float)
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIC50")] = IC50 # 0.00205 nM (PMID: 28698254); 0.093 nM (PMID: 8405712); 2.1 nM (PMID: 17138746)
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIMax")] = 0.75 # assumed from PMID: 28698254
    model_PK.parameter_values     = np.asarray(PKParams, dtype=float)
    model_GI.parameter_values       = np.asarray(GIparams, dtype=float)
    lambda_chow = BWparams[-2]
    lambda_HF   = lambdaNewDiet
    model_BW.parameter_values       = np.asarray(BWparams[:-2], dtype=float)
    model_BW.parameter_values[model_BW.parameter_names.index("b_CLGI")] = 4.8*GIparams[-1]  # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_EGP")]  = 70*GIparams[-1]   # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_Ins")]  = 3.87*GIparams[-1] # scaling of Simonsson et al due to changes in where its implemented in the model 

    # ---------- Study settings ----------
    BW0_g = 26.5
    BW0_kg = BW0_g / 1000
    FM0_g = 0.10 * BW0_g
    FFM0_g = BW0_g - FM0_g

    SCInfusion_E3_nmol_per_day  = 3 * BW0_kg  / (1440*4.1866)
    SCInfusion_E10_nmol_per_day = 10 * BW0_kg / (1440*4.1866)
    SCInfusion_E30_nmol_per_day = 30 * BW0_kg / (1440*4.1866)

    sim_t = np.linspace(0, 28, (28 * 48*2) + 1)

    # ---------- ss activity ----------
    activity_ss = sund.Activity(time_unit="d")
    activity_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.58])
    activity_ss.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.26])
    activity_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_ss.add_output(type="piecewise_constant", name="SCInfusion", t=[], f=[0])
    activity_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0])  # 4 weeks old
    # ---------- Vehicle activity ----------
    activity_vehicle = sund.Activity(time_unit="d")
    activity_vehicle.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_vehicle.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_vehicle.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.58])
    activity_vehicle.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.26])
    activity_vehicle.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_vehicle.add_output(type="piecewise_constant", name="SCInfusion", t=[], f=[0])
    activity_vehicle.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 4 weeks old

    # ---------- E3 activity ----------
    activity_E3 = sund.Activity(time_unit="d")
    activity_E3.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_E3.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_E3.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.58])
    activity_E3.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.26])
    activity_E3.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_E3.add_output(type="piecewise_constant", name="SCInfusion", t=[], f=[SCInfusion_E3_nmol_per_day])
    activity_E3.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 4 weeks old
    # ---------- E10 activity ----------
    activity_E10 = sund.Activity(time_unit="d")
    activity_E10.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_E10.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_E10.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.58])
    activity_E10.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.26])
    activity_E10.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_E10.add_output(type="piecewise_constant", name="SCInfusion", t=[], f=[SCInfusion_E10_nmol_per_day])
    activity_E10.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 4 weeks old
    # ---------- E30 activity ----------
    activity_E30 = sund.Activity(time_unit="d")
    activity_E30.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_E30.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_E30.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.58])
    activity_E30.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.26])
    activity_E30.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_E30.add_output(type="piecewise_constant", name="SCInfusion", t=[], f=[SCInfusion_E30_nmol_per_day])
    activity_E30.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 4 weeks old


    # ---------- Compute steady state ----------
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK, model_BW], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
    simulation_ss.simulate()

    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_PK = assign_state_values_from_simulation(model_PK, simulation_ss)
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss)
    model_BW.state_values[model_BW.state_names.index("FM")] = FM0_g
    model_BW.state_values[model_BW.state_names.index("FFM")] = FFM0_g
    model_BW.state_values[model_BW.state_names.index("lambda_delay")] = lambda_HF*Recovery
    
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK, model_BW], activities=[activity_ss], time_vector=np.linspace(0, 0.1, 2))
    simulation_ss.simulate()
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    Model_Appetite_state_values_reset = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    model_PK_state_values_reset = np.asarray(model_PK.state_values, dtype=float).copy()
    model_GI_state_values_reset = np.asarray(model_GI.state_values, dtype=float).copy()
    model_BW_state_values_reset = np.asarray(model_BW.state_values, dtype=float).copy()

    # ---------- Vehicle simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_PK.state_values = model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    sim_vehicle = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK, model_BW], activities=[activity_vehicle], time_vector=sim_t)
    sim_vehicle.simulate()
    # ---------- E3 simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_PK.state_values = model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    sim_E3 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK, model_BW], activities=[activity_E3], time_vector=sim_t)
    sim_E3.simulate()
    # ---------- E10 simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_PK.state_values = model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    sim_E10 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK, model_BW], activities=[activity_E10], time_vector=sim_t)
    sim_E10.simulate()
    # ---------- E30 simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_PK.state_values = model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    sim_E30 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_PK, model_BW], activities=[activity_E30], time_vector=sim_t)
    sim_E30.simulate()

    MackSimulations = {
        "vehicle": sim_vehicle,
        "E3": sim_E3,
        "E10": sim_E10,
        "E30": sim_E30,
    }

    return MackSimulations


# ---------- He study ----------
def Simulate_HEStudy(Model_Appetite, model_GI, model_BW, data, AppetiteParams, GIparams, BWparams,TimeRes):

    Model_Appetite.parameter_values = np.asarray(AppetiteParams, dtype=float)
    model_GI.parameter_values       = np.asarray(GIparams[:-4], dtype=float)

    lambda_chow = BWparams[-2]*GIparams[-3]
    lambda_HF   = BWparams[-1]*GIparams[-2]
    model_BW.parameter_values       = np.asarray(BWparams[:-2], dtype=float)
    model_BW.parameter_values[model_BW.parameter_names.index("b_CLGI")] = 4.8*GIparams[-1]  # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_EGP")]  = 70*GIparams[-1]   # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_Ins")]  = 3.87*GIparams[-1] # scaling of Simonsson et al due to changes in where its implemented in the model 


    BW0_g = float(np.nanmean([
        np.array(data["HeStudy"]["Control_BW"], dtype=float)[0],
        np.array(data["HeStudy"]["HFD_BW"], dtype=float)[0]
    ]))

    FM0_g  = 0.1 * BW0_g # Not obese
    FFM0_g = BW0_g - FM0_g

    sim_t_end = 148
    sim_t = np.linspace(0, sim_t_end, int(sim_t_end * TimeRes) + 1)


    # ---------- ss diet activity ----------
    activity_ss = sund.Activity(time_unit="d")
    activity_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.10]) # Research Diets D12450J
    activity_ss.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.70])
    activity_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_chow])
    activity_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0])  # 6 weeks old
    # ---------- Control diet activity ----------
    activity_control = sund.Activity(time_unit="d")
    activity_control.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_control.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_control.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.10]) # Research Diets D12450J
    activity_control.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.70])
    activity_control.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_chow])
    activity_control.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 6 weeks old
    # ---------- HFD activity ----------
    activity_HFD = sund.Activity(time_unit="d")
    activity_HFD.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_HFD.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_HFD.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60]) # Research Diets D12492
    activity_HFD.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_HFD.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_HFD.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 6 weeks old


    # ---------- Compute steady state on control diet ----------
    Initial_G_SS = model_GI.parameter_values[model_GI.parameter_names.index("G_SS")]
    model_GI.parameter_values[model_GI.parameter_names.index("G_SS")] = Initial_G_SS * GIparams[-4]
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, model_GI], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
    simulation_ss.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss)
    model_BW.state_values[model_BW.state_names.index("FM")] = FM0_g
    model_BW.state_values[model_BW.state_names.index("FFM")] = FFM0_g
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, model_GI], activities=[activity_ss], time_vector=np.linspace(0, 0.2, 2))
    simulation_ss.simulate()
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    Model_Appetite_state_values_reset = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    model_GI_state_values_reset       = np.asarray(model_GI.state_values, dtype=float).copy()
    model_BW_state_values_reset       = np.asarray(model_BW.state_values, dtype=float).copy()
    # ---------- Control diet simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_GI.state_values       = model_GI_state_values_reset.copy()
    model_BW.state_values       = model_BW_state_values_reset.copy()
    sim_control = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, model_GI], activities=[activity_control], time_vector=sim_t)
    sim_control.simulate()


    # ---------- Compute steady state on HFD diet ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_GI.state_values       = model_GI_state_values_reset.copy()
    model_BW.state_values       = model_BW_state_values_reset.copy()
    model_GI.parameter_values[model_GI.parameter_names.index("G_SS")] = Initial_G_SS * GIparams[-4]*1.1
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, model_GI], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
    simulation_ss.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss)
    model_BW.state_values[model_BW.state_names.index("FM")] = FM0_g
    model_BW.state_values[model_BW.state_names.index("FFM")] = FFM0_g
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, model_GI], activities=[activity_ss], time_vector=np.linspace(0, 0.2, 2))
    simulation_ss.simulate()
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    Model_Appetite_state_values_reset = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    model_GI_state_values_reset       = np.asarray(model_GI.state_values, dtype=float).copy()
    model_BW_state_values_reset       = np.asarray(model_BW.state_values, dtype=float).copy()
    # ---------- HFD simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_GI.state_values       = model_GI_state_values_reset.copy()
    model_BW.state_values       = model_BW_state_values_reset.copy()
    sim_HFD = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, model_GI], activities=[activity_HFD], time_vector=sim_t)
    sim_HFD.simulate()

    HeSimulations = {"control": sim_control, "HFD": sim_HFD}

    return HeSimulations
def CalcSimulatedValues_HEStudy(Model_Appetite, model_GI, model_BW, data, Appetite_params, GI_params, BW_params):

    TimeRes = 24
    sims = Simulate_HEStudy(Model_Appetite, model_GI, model_BW, data, Appetite_params, GI_params, BW_params,TimeRes)

    t_BW = np.array(data["HeStudy"]["time_BW"][0:15], dtype=float)
    t_BW_idx = np.round(t_BW * TimeRes).astype(int)
    t_Glucose = np.array(data["HeStudy"]["time_Glucose"][0:16], dtype=float)

    idx_bw      = sims["control"].feature_names.index("Body_weight")
    idx_glucose = sims["control"].feature_names.index("Plasma_Glucose")

    # ---------- BW study ----------
    BW_control      = sims["control"].feature_values[t_BW_idx, idx_bw]
    BW_HFD          = sims["HFD"].feature_values[t_BW_idx, idx_bw]

    def daily_min(values, points_per_day=2):
        values = np.asarray(values, dtype=float)
        n_days = (len(values) - 1) // points_per_day
        return values[:n_days*points_per_day].reshape(n_days, points_per_day).min(axis=1)
    Glucose_control_daily = daily_min(sims["control"].feature_values[:, idx_glucose],points_per_day=TimeRes)
    Glucose_HFD_daily = daily_min( sims["HFD"].feature_values[:, idx_glucose], points_per_day=TimeRes)
    Glucose_control = Glucose_control_daily[t_Glucose.astype(int)]
    Glucose_HFD     = Glucose_HFD_daily[t_Glucose.astype(int)]

    SimulatedValues = {}
    SimulatedValues["values"] = np.concatenate([
        BW_control,
        BW_HFD,
        Glucose_control,
        Glucose_HFD
    ])
    SimulatedValues["groups"] = np.concatenate([
        np.repeat("BW_control", len(t_BW)),
        np.repeat("BW_HFD", len(t_BW)),
        np.repeat("Glucose_control", len(t_Glucose)),
        np.repeat("Glucose_HFD", len(t_Glucose))
    ])
    
    SimulatedValues["time"] = np.concatenate([
        t_BW,
        t_BW,
        t_Glucose,
        t_Glucose
    ])
    return SimulatedValues


# ---------- Gabery study ----------
def Simulate_GaberyStudy(Model_Appetite, model_BW, Model_PK, data, AppetiteParams, BWParams, PKParams,PhysicalActivity,Recovery, IC50, hill):

    lambda_Control = PhysicalActivity
    Model_Appetite.parameter_values = np.asarray(AppetiteParams, dtype=float)
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIC50")] = IC50
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIMax")] = 0.9
    model_BW.parameter_values = np.asarray(BWParams[:-2], dtype=float)
    Model_PK.parameter_values = np.asarray(PKParams, dtype=float)

    # ---------- Lookup state indices ----------
    idx_state_ffm          = model_BW.state_names.index("FFM")
    idx_state_fm           = model_BW.state_names.index("FM")
    idx_state_lambda_delay = model_BW.state_names.index("lambda_delay")

    # ---------- Study settings ----------
    t_end = 24
    SimulationRes = 24
    sim_t = np.linspace(0, t_end, int(t_end * SimulationRes) + 1)
    FatRatio_HF = 0.45
    CarbRatio_HF = 0.35
    treatment_start = 3
    dosing_times_BID = np.sort(np.concatenate([
        np.arange(treatment_start, t_end, 1.0),
        np.arange(treatment_start + 0.3, t_end, 1.0)
    ]))

    Mean_bw0 = np.mean([
        data["GaberyStudy"]["VehicleBW"][0],
        data["GaberyStudy"]["Sema1BW"][0],
        data["GaberyStudy"]["Sema3BW"][0],
        data["GaberyStudy"]["Sema10BW"][0],
        data["GaberyStudy"]["Sema30BW"][0],
        data["GaberyStudy"]["Sema100BW"][0]
    ]).astype(float)
    Vehicle_bw0 = Sema1_bw0 = Sema3_bw0 = Sema10_bw0 = Sema30_bw0 = Sema100_bw0 = Mean_bw0

    # ---------- BID dose per administration ----------
    dose_Sema1_BID = 1 * (Sema1_bw0 / 1000) / 2
    dose_Sema3_BID = 3 * (Sema3_bw0 / 1000) / 2
    dose_Sema10_BID = 10 * (Sema10_bw0 / 1000) / 2
    dose_Sema30_BID = 30 * (Sema30_bw0 / 1000) / 2
    dose_Sema100_BID = 100 * (Sema100_bw0 / 1000) / 2


    # ---------- Initial body composition ----------
    Mice_BodyfatProcentage = 0.3

    Vehicle_fm0  = Mice_BodyfatProcentage * Vehicle_bw0
    Vehicle_ffm0 = Vehicle_bw0 - Vehicle_fm0
    Sema1_fm0  = Mice_BodyfatProcentage * Sema1_bw0
    Sema1_ffm0 = Sema1_bw0 - Sema1_fm0
    Sema3_fm0  = Mice_BodyfatProcentage * Sema3_bw0
    Sema3_ffm0 = Sema3_bw0 - Sema3_fm0
    Sema10_fm0  = Mice_BodyfatProcentage * Sema10_bw0
    Sema10_ffm0 = Sema10_bw0 - Sema10_fm0
    Sema30_fm0  = Mice_BodyfatProcentage * Sema30_bw0
    Sema30_ffm0 = Sema30_bw0 - Sema30_fm0
    Sema100_fm0  = Mice_BodyfatProcentage * Sema100_bw0
    Sema100_ffm0 = Sema100_bw0 - Sema100_fm0


    # ---------- ss  activity ----------
    activity_ss = sund.Activity(time_unit="d")
    activity_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_ss.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Vehicle_fm0])
    activity_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Vehicle_ffm0])
    activity_ss.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ss.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0])  # 24 weeks old

    # ---------- Vehicle  activity ----------
    activity_Vehicle = sund.Activity(time_unit="d")
    activity_Vehicle.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_Vehicle.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_Vehicle.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_Vehicle.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Vehicle_fm0])
    activity_Vehicle.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Vehicle_ffm0])
    activity_Vehicle.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Vehicle.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Vehicle.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 24 weeks old

    activity_Sema1 = sund.Activity(time_unit="d")
    activity_Sema1.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_Sema1.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_Sema1.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_Sema1.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Sema1_fm0])
    activity_Sema1.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Sema1_ffm0])
    activity_Sema1.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema1.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema1.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 24 weeks old
    activity_Sema1.add_state_manipulation(mode="add", name="A", t=dosing_times_BID.tolist(), f=np.repeat(dose_Sema1_BID, len(dosing_times_BID)).tolist())

    activity_Sema3 = sund.Activity(time_unit="d")
    activity_Sema3.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_Sema3.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_Sema3.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_Sema3.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Sema3_fm0])
    activity_Sema3.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Sema3_ffm0])
    activity_Sema3.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema3.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema3.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 24 weeks old
    activity_Sema3.add_state_manipulation(mode="add", name="A", t=dosing_times_BID.tolist(), f=np.repeat(dose_Sema3_BID, len(dosing_times_BID)).tolist())

    activity_Sema10 = sund.Activity(time_unit="d")
    activity_Sema10.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_Sema10.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_Sema10.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_Sema10.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Sema10_fm0])
    activity_Sema10.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Sema10_ffm0])
    activity_Sema10.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema10.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema10.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 24 weeks old
    activity_Sema10.add_state_manipulation(mode="add", name="A", t=dosing_times_BID.tolist(), f=np.repeat(dose_Sema10_BID, len(dosing_times_BID)).tolist())

    activity_Sema30 = sund.Activity(time_unit="d")
    activity_Sema30.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_Sema30.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_Sema30.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_Sema30.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Sema30_fm0])
    activity_Sema30.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Sema30_ffm0])
    activity_Sema30.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema30.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema30.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 24 weeks old
    activity_Sema30.add_state_manipulation(mode="add", name="A", t=dosing_times_BID.tolist(), f=np.repeat(dose_Sema30_BID, len(dosing_times_BID)).tolist())

    activity_Sema100 = sund.Activity(time_unit="d")
    activity_Sema100.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[FatRatio_HF])
    activity_Sema100.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[CarbRatio_HF])
    activity_Sema100.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_Control])
    activity_Sema100.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Sema100_fm0])
    activity_Sema100.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Sema100_ffm0])
    activity_Sema100.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema100.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema100.add_output(type="piecewise_linear", name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 24 weeks old
    activity_Sema100.add_state_manipulation(mode="add", name="A", t=dosing_times_BID.tolist(), f=np.repeat(dose_Sema100_BID, len(dosing_times_BID)).tolist())

    # ---------- Set vehicle BW states before steady state ----------
    vehicle_state0 = np.asarray(model_BW.state_values, dtype=float).copy()
    vehicle_state0[idx_state_ffm] = Vehicle_ffm0
    vehicle_state0[idx_state_fm] = Vehicle_fm0
    model_BW.state_values = vehicle_state0.copy()

    # ---------- Compute steady state on HFD ----------
    simulation_ss = sund.Simulation( time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
    simulation_ss.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    Model_PK = assign_state_values_from_simulation(Model_PK, simulation_ss)
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss)
    model_BW.state_values[idx_state_fm] = Vehicle_fm0
    model_BW.state_values[idx_state_ffm] = Vehicle_ffm0

    model_BW.state_values[model_BW.state_names.index("lambda_delay")] = lambda_Control*Recovery

    Appetite_state_values_reset = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    BW_state_values_reset = np.asarray(model_BW.state_values, dtype=float).copy()
    PK_state_values_reset = np.asarray(Model_PK.state_values, dtype=float).copy()

    # ---------- Construct group-specific BW states ----------
    Vehicle_state0 = BW_state_values_reset.copy()
    Vehicle_state0[idx_state_ffm] = Vehicle_ffm0
    Vehicle_state0[idx_state_fm] = Vehicle_fm0

    Sema1_state0 = BW_state_values_reset.copy()
    Sema1_state0[idx_state_ffm] = Sema1_ffm0
    Sema1_state0[idx_state_fm] = Sema1_fm0

    Sema3_state0 = BW_state_values_reset.copy()
    Sema3_state0[idx_state_ffm] = Sema3_ffm0
    Sema3_state0[idx_state_fm] = Sema3_fm0

    Sema10_state0 = BW_state_values_reset.copy()
    Sema10_state0[idx_state_ffm] = Sema10_ffm0
    Sema10_state0[idx_state_fm] = Sema10_fm0

    Sema30_state0 = BW_state_values_reset.copy()
    Sema30_state0[idx_state_ffm] = Sema30_ffm0
    Sema30_state0[idx_state_fm] = Sema30_fm0

    Sema100_state0 = BW_state_values_reset.copy()
    Sema100_state0[idx_state_ffm] = Sema100_ffm0
    Sema100_state0[idx_state_fm] = Sema100_fm0

    # ---------- Vehicle ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = Vehicle_state0.copy()
    Model_PK.state_values = PK_state_values_reset.copy()
    simulation_Vehicle = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_Vehicle], time_vector=sim_t)
    simulation_Vehicle.simulate()
    # ---------- Semaglutide 1 nmol/kg/day BID ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = Sema1_state0.copy()
    Model_PK.state_values = PK_state_values_reset.copy()
    simulation_Sema1 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_Sema1], time_vector=sim_t)
    simulation_Sema1.simulate()
    # ---------- Semaglutide 3 nmol/kg/day BID ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = Sema3_state0.copy()
    Model_PK.state_values = PK_state_values_reset.copy()
    simulation_Sema3 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_Sema3], time_vector=sim_t)
    simulation_Sema3.simulate()
    # ---------- Semaglutide 10 nmol/kg/day BID ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = Sema10_state0.copy()
    Model_PK.state_values = PK_state_values_reset.copy()
    simulation_Sema10 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_Sema10], time_vector=sim_t)
    simulation_Sema10.simulate()
    # ---------- Semaglutide 30 nmol/kg/day BID ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = Sema30_state0.copy()
    Model_PK.state_values = PK_state_values_reset.copy()
    simulation_Sema30 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_Sema30], time_vector=sim_t)
    simulation_Sema30.simulate()
    # ---------- Semaglutide 100 nmol/kg/day BID ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = Sema100_state0.copy()
    Model_PK.state_values = PK_state_values_reset.copy()
    simulation_Sema100 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK], activities=[activity_Sema100], time_vector=sim_t)
    simulation_Sema100.simulate()

    # ---------- Restore models after all independent simulations ----------
    Model_Appetite.state_values = Appetite_state_values_reset.copy()
    model_BW.state_values = BW_state_values_reset.copy()
    Model_PK.state_values = PK_state_values_reset.copy()

    GaberySimulations = {
        "Vehicle": simulation_Vehicle,
        "Sema1": simulation_Sema1,
        "Sema3": simulation_Sema3,
        "Sema10": simulation_Sema10,
        "Sema30": simulation_Sema30,
        "Sema100": simulation_Sema100
    }

    return GaberySimulations


# ---------- Az study ----------
def Simulate_AZstudy(Model_Appetite, model_BW, Model_PK, model_GI, data, AppetiteParams, BWparams, PKparams, GIparams, TimeRes_exposure,PhysicalActivity,Recovery,IC50,hill):

    Model_Appetite.parameter_values = np.asarray(AppetiteParams, dtype=float)
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIC50")] = IC50
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIMax")] = 0.9
    
    Model_PK.parameter_values = np.asarray(PKparams, dtype=float)

    model_GI.parameter_values = np.asarray(GIparams[:-4], dtype=float)
    model_GI.parameter_values[model_GI.parameter_names.index("G_SS")] *= 1.55
    model_GI.parameter_values[model_GI.parameter_names.index("I_SS")] *= 2.75

    lambda_HF = PhysicalActivity
    model_BW.parameter_values = np.asarray(BWparams[:-2], dtype=float)
    model_BW.parameter_values[model_BW.parameter_names.index("b_CLGI")] = 4.8*GIparams[-1]  # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_EGP")]  = 70*GIparams[-1]   # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_Ins")]  = 3.87*GIparams[-1] # scaling of Simonsson et al due to changes in where its implemented in the model 

    
    BW0_g = float(np.nanmean([
        np.array(data["AZstudy"]["VehicleBW"], dtype=float)[0],
        np.array(data["AZstudy"]["Sema0p3BW"], dtype=float)[0],
        np.array(data["AZstudy"]["Sema2BW"], dtype=float)[0],
        np.array(data["AZstudy"]["Sema4_Q2DBW"], dtype=float)[0],
        np.array(data["AZstudy"]["Sema10BW"], dtype=float)[0]
    ]))

    FM0_g = 0.3 * BW0_g
    FFM0_g = BW0_g - FM0_g

    BW_kg = BW0_g / 1000
    dose_Sema0p3 = 0.3 * BW_kg
    dose_Sema2 = 2 * BW_kg
    dose_Sema4_Q2D = 4 * BW_kg
    dose_Sema10 = 10 * BW_kg

    # ---------- Simulation time vectors ----------
    sim_t_end_regular = 22
    sim_t_end_Sema10 = 40

    sim_t_regular = np.linspace(0, sim_t_end_regular, int(sim_t_end_regular * 24 * TimeRes_exposure) + 1)
    sim_t_Sema10 = np.linspace(0, sim_t_end_Sema10, int(sim_t_end_Sema10 * 24 * TimeRes_exposure) + 1)

    # ---------- Dosing times ----------
    t_Sema0p3_QD = np.arange(0, 22, 1, dtype=float)
    t_Sema2_QD = np.arange(0, 22, 1, dtype=float)
    t_Sema4_Q2D  = np.append(np.arange(0, 22, 2, dtype=float), 21.0)
    t_Sema10_QD = np.arange(0, 21, 1, dtype=float)



    # ---------- ss  activity ----------
    activity_ss = sund.Activity(time_unit="d")
    activity_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_ss.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_ss.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ss.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0])  # 31 weeks old


    # ---------- HFD vehicle activity ----------
    activity_Vehicle = sund.Activity(time_unit="d")
    activity_Vehicle.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_Vehicle.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_Vehicle.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_Vehicle.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_Vehicle.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_Vehicle.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Vehicle.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Vehicle.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 31 weeks old

    # ---------- HFD and 0.3 nmol/kg QD activity ----------
    activity_Sema0p3 = sund.Activity(time_unit="d")
    activity_Sema0p3.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_Sema0p3.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_Sema0p3.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_Sema0p3.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_Sema0p3.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_Sema0p3.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema0p3.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema0p3.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 31 weeks old
    activity_Sema0p3.add_state_manipulation(mode="add", name="A", t=t_Sema0p3_QD.tolist(), f=np.repeat(dose_Sema0p3, len(t_Sema0p3_QD)).tolist())

    # ---------- HFD and 2 nmol/kg QD activity ----------
    activity_Sema2 = sund.Activity(time_unit="d")
    activity_Sema2.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_Sema2.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_Sema2.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_Sema2.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_Sema2.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_Sema2.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema2.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema2.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 31 weeks old
    activity_Sema2.add_state_manipulation(mode="add", name="A", t=t_Sema2_QD.tolist(), f=np.repeat(dose_Sema2, len(t_Sema2_QD)).tolist())

    # ---------- HFD and 4 nmol/kg Q2D activity ----------
    activity_Sema4_Q2D = sund.Activity(time_unit="d")
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema4_Q2D.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema4_Q2D.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 31 weeks old
    activity_Sema4_Q2D.add_state_manipulation(mode="add", name="A", t=t_Sema4_Q2D.tolist(), f=np.repeat(dose_Sema4_Q2D, len(t_Sema4_Q2D)).tolist())

    # ---------- HFD and 10 nmol/kg QD activity ----------
    activity_Sema10 = sund.Activity(time_unit="d")
    activity_Sema10.add_output(type="piecewise_constant", name="FMinit", t=[], f=[FM0_g])
    activity_Sema10.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[FFM0_g])
    activity_Sema10.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_Sema10.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_Sema10.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HF])
    activity_Sema10.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_Sema10.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[hill])
    activity_Sema10.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) # 31 weeks old
    activity_Sema10.add_state_manipulation(mode="add", name="A", t=t_Sema10_QD.tolist(), f=np.repeat(dose_Sema10, len(t_Sema10_QD)).tolist())

    # ---------- Store original state values before steady-state simulation ----------
    Model_Appetite_state_values_original = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    model_BW_state_values_original = np.asarray(model_BW.state_values, dtype=float).copy()
    Model_PK_state_values_original = np.asarray(Model_PK.state_values, dtype=float).copy()
    model_GI_state_values_original = np.asarray(model_GI.state_values, dtype=float).copy()

    # ---------- Compute steady state on HFD without semaglutide ----------
    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ss], time_vector=np.linspace(0, 28.5, 2))
    simulation_ss.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss)
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    Model_PK = assign_state_values_from_simulation(Model_PK, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_BW.state_values[model_BW.state_names.index("FM")] = FM0_g
    model_BW.state_values[model_BW.state_names.index("FFM")] = FFM0_g
    Model_PK.state_values = Model_PK_state_values_original.copy()

    simulation_ss = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ss], time_vector=np.linspace(0, 1, 2))
    simulation_ss.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss)
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss)
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss)
    Model_PK = assign_state_values_from_simulation(Model_PK, simulation_ss)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 12
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_BW.state_values[model_BW.state_names.index("FM")] = FM0_g
    model_BW.state_values[model_BW.state_names.index("FFM")] = FFM0_g
    Model_PK.state_values = Model_PK_state_values_original.copy()

    model_BW.state_values[model_BW.state_names.index("lambda_delay")] = lambda_HF*Recovery

    Model_Appetite_state_values_reset = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    model_BW_state_values_reset = np.asarray(model_BW.state_values, dtype=float).copy()
    Model_PK_state_values_reset = np.asarray(Model_PK.state_values, dtype=float).copy()
    model_GI_state_values_reset = np.asarray(model_GI.state_values, dtype=float).copy()


    # ---------- Vehicle simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    Model_PK.state_values = Model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()

    simulation_Vehicle = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_Vehicle], time_vector=sim_t_regular)
    simulation_Vehicle.simulate()

    # ---------- 0.3 nmol/kg QD simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    Model_PK.state_values = Model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()

    simulation_Sema0p3 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_Sema0p3], time_vector=sim_t_regular)
    simulation_Sema0p3.simulate()

    # ---------- 2 nmol/kg QD simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    Model_PK.state_values = Model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()

    simulation_Sema2 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_Sema2], time_vector=sim_t_regular)
    simulation_Sema2.simulate()

    # ---------- 4 nmol/kg Q2D simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    Model_PK.state_values = Model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()

    simulation_Sema4_Q2D = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_Sema4_Q2D], time_vector=sim_t_regular)
    simulation_Sema4_Q2D.simulate()

    # ---------- 10 nmol/kg QD simulation ----------
    Model_Appetite.state_values = Model_Appetite_state_values_reset.copy()
    model_BW.state_values = model_BW_state_values_reset.copy()
    Model_PK.state_values = Model_PK_state_values_reset.copy()
    model_GI.state_values = model_GI_state_values_reset.copy()

    simulation_Sema10 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_Sema10], time_vector=sim_t_Sema10)
    simulation_Sema10.simulate()

    # ---------- Restore model states present when the function was called ----------
    Model_Appetite.state_values = Model_Appetite_state_values_original.copy()
    model_BW.state_values = model_BW_state_values_original.copy()
    Model_PK.state_values = Model_PK_state_values_original.copy()
    model_GI.state_values = model_GI_state_values_original.copy()

    AZsimulations = {
        "Vehicle": simulation_Vehicle,
        "Sema0p3": simulation_Sema0p3,
        "Sema2": simulation_Sema2,
        "Sema4_Q2D": simulation_Sema4_Q2D,
        "Sema10": simulation_Sema10
    }

    return AZsimulations


# ---------- Digital populations ----------
def Simulate_DigitalPopulations(Model_Appetite, model_BW, Model_PK, model_GI, AppetiteParams, BWparams, PKparams, GIparams, TimeRes_exposure, IC50):

    Model_Appetite.parameter_values = np.asarray(AppetiteParams, dtype=float)
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIC50")] = IC50
    Model_Appetite.parameter_values[Model_Appetite.parameter_names.index("ExenatideIMax")] = 0.9
    Model_PK.parameter_values = np.asarray(PKparams, dtype=float)
    model_GI.parameter_values = np.asarray(GIparams[:-4], dtype=float)
    model_GI.parameter_values[model_GI.parameter_names.index("G_SS")] *= 0.775
    model_GI.parameter_values[model_GI.parameter_names.index("I_SS")] *= 1.375

    model_BW.parameter_values = np.asarray(BWparams[:-2], dtype=float)
    model_BW.parameter_values[model_BW.parameter_names.index("b_CLGI")] = 4.8*GIparams[-1]  # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_EGP")]  = 70*GIparams[-1]   # scaling of Simonsson et al due to changes in where its implemented in the model 
    model_BW.parameter_values[model_BW.parameter_names.index("b_Ins")]  = 3.87*GIparams[-1] # scaling of Simonsson et al due to changes in where its implemented in the model 

    # ---------- Initial body weights and composition ----------
    Healthy_BW0_g = 30.0
    Healthy_BodyfatProcentage = 0.10
    Healthy_FM0_g = Healthy_BodyfatProcentage * Healthy_BW0_g
    Healthy_FFM0_g = Healthy_BW0_g - Healthy_FM0_g

    Obese_BW0_g = 40.0
    Obese_BodyfatProcentage = 0.30
    Obese_FM0_g = Obese_BodyfatProcentage * Obese_BW0_g
    Obese_FFM0_g = Obese_BW0_g - Obese_FM0_g

    # ---------- Assume physical activity calculations ----------
    HealthyMice = Healthy_BW0_g # g
    ObeseMice   = Obese_BW0_g   # g

    FatRatio_D12450J = 0.10 # %kcal
    FatRatio_D12492  = 0.60
    FatRatio_D12331  = 0.58
    FatRatio_D124512 = 0.45
    FatRatio_LM_485  = 0.17

    lambda_ChowD12450J_HealthyAdult    =  -0.044795*FatRatio_D12450J + 0.414238 -0.006515*HealthyMice
    lambda_HFD12492_HealthyAdult       =  -0.044795*FatRatio_D12492  + 0.414238 -0.006515*HealthyMice 
    lambda_HFD12331_HealthyAdult       =  -0.044795*FatRatio_D12331  + 0.414238 -0.006515*HealthyMice 
    lambda_Control_124512_HealthyAdult =  -0.044795*FatRatio_D124512 + 0.414238 -0.006515*HealthyMice 
    lambda_Control_LM_485_HealthyAdult =  -0.044795*FatRatio_LM_485  + 0.414238 -0.006515*HealthyMice 

    lambda_ChowD12450J_ObeseAdult =  -0.044795*FatRatio_D12450J + 0.414238 - 0.006515*ObeseMice
    lambda_HFD12492_ObeseAdult    =  -0.044795*FatRatio_D12492  + 0.414238 - 0.006515*ObeseMice

    idx_state_ffm = model_BW.state_names.index("FFM")
    idx_state_fm = model_BW.state_names.index("FM")

    # ---------- Simulation time vector ----------
    simulation_days = 56
    sim_t = np.linspace(0, simulation_days, int(simulation_days * 24 * TimeRes_exposure) + 1)

    # ---------- Semaglutide dosing ----------
    t_Sema_QD   = np.arange(14, simulation_days, 1, dtype=float)
    dose_Sema2  = 2 * (Obese_BW0_g / 1000)
    dose_Sema10 = 10 * (Obese_BW0_g / 1000)


    # ---------- Healthy Activities ----------
    #  Chow (He study; D12450J
    activity_HealthyChow_D12450J_ss = sund.Activity(time_unit="d")
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.10])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.70])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_ChowD12450J_HealthyAdult])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_HealthyChow_D12450J_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0]) 
    #  HFD (He study; D12492) 
    activity_ObeseHFD_D12492_ss = sund.Activity(time_unit="d")
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FM0_g])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HFD12492_ObeseAdult])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_ObeseHFD_D12492_ss.add_output(type="piecewise_constant", name="Weeks", t=[], f=[0])


    # ---------- Healthy Activities ----------
    #  Chow (He study; D12450J
    activity_HealthyChow_D12450J = sund.Activity(time_unit="d")
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.10])
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.70])
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_ChowD12450J_HealthyAdult])
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_HealthyChow_D12450J.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_HealthyChow_D12450J.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) 
    #  HFD (He study; D12492) 
    activity_HealthyHFD_D12492 = sund.Activity(time_unit="d")
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="FatRatio", t=[14], f=[0.10, 0.60])
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="CarbRatio", t=[14], f=[0.70, 0.20])
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="lambda", t=[14], f=[lambda_ChowD12450J_HealthyAdult,lambda_HFD12492_HealthyAdult])
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_HealthyHFD_D12492.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_HealthyHFD_D12492.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) 
    #  HFD (Mack study; D12331) 
    activity_HealthyHFD_D12331 = sund.Activity(time_unit="d")
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="FatRatio", t=[14], f=[0.10, 0.58])
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="CarbRatio", t=[14], f=[0.70, 0.26])
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="lambda", t=[14], f=[lambda_ChowD12450J_HealthyAdult,lambda_HFD12331_HealthyAdult])
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_HealthyHFD_D12331.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_HealthyHFD_D12331.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) 
    #  Control (Gabery study; D124512) 
    activity_HealthyControl_D124512 = sund.Activity(time_unit="d")
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="FatRatio", t=[14], f=[0.10, 0.45])
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="CarbRatio", t=[14], f=[0.70, 0.35])
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="lambda", t=[14], f=[lambda_ChowD12450J_HealthyAdult,lambda_Control_124512_HealthyAdult])
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_HealthyControl_D124512.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_HealthyControl_D124512.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7])
    #  Control (Mack study; Teklad LM-485 ) 
    activity_HealthyControl_TekladLM_485 = sund.Activity(time_unit="d")
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="FatRatio", t=[14], f=[0.10, 0.17])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="CarbRatio", t=[14], f=[0.70, 0.58])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="lambda", t=[14], f=[lambda_ChowD12450J_HealthyAdult,lambda_Control_LM_485_HealthyAdult])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_HealthyControl_TekladLM_485.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7])

    # ---------- Obese Activities ----------
    #  Chow (He study; D12450J
    activity_ObeseChow_D12450J = sund.Activity(time_unit="d")
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="FatRatio", t=[14], f=[0.60, 0.10])
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="CarbRatio", t=[14], f=[0.20, 0.70])
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="lambda", t=[14], f=[lambda_HFD12492_ObeseAdult, lambda_ChowD12450J_ObeseAdult])
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ObeseChow_D12450J.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_ObeseChow_D12450J.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) 
    #  HFD (He study; D12492) 
    activity_ObeseHFD_D12492 = sund.Activity(time_unit="d")
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HFD12492_ObeseAdult])
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ObeseHFD_D12492.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_ObeseHFD_D12492.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7]) 

    # ---------- Treatment Activities ----------
    #  Obese HFD + 10 nmol/kg 
    activity_ObeseHFD_Sema2 = sund.Activity(time_unit="d")
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HFD12492_ObeseAdult])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_ObeseHFD_Sema2.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7])
    activity_ObeseHFD_Sema2.add_state_manipulation(mode="add", name="A", t=t_Sema_QD.tolist(), f=np.repeat(dose_Sema2, len(t_Sema_QD)).tolist())
    #  Obese Chow + 10 nmol/kg 
    activity_ObeseHFD_Sema10 = sund.Activity(time_unit="d")
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="FMinit", t=[], f=[Healthy_FM0_g])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="FFMinit", t=[], f=[Healthy_FFM0_g])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="FatRatio", t=[], f=[0.60])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="CarbRatio", t=[], f=[0.20])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="lambda", t=[], f=[lambda_HFD12492_ObeseAdult])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="BW_ref", t=[], f=[0.05])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_constant", name="ExenatideIHill", t=[], f=[2])
    activity_ObeseHFD_Sema10.add_output(type="piecewise_linear"  , name="Weeks", t=[0,1000], f=[0,0+1000/7])
    activity_ObeseHFD_Sema10.add_state_manipulation(mode="add", name="A", t=t_Sema_QD.tolist(), f=np.repeat(dose_Sema10, len(t_Sema_QD)).tolist())


    # ---------- Store original state values ----------
    Model_Appetite_state_values_original = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    model_BW_state_values_original       = np.asarray(model_BW.state_values, dtype=float).copy()
    Model_PK_state_values_original       = np.asarray(Model_PK.state_values, dtype=float).copy()
    model_GI_state_values_original       = np.asarray(model_GI.state_values, dtype=float).copy()


    # ---------- STEADY STATE ----------
    # Chow healthy
    Model_Appetite.state_values = Model_Appetite_state_values_original.copy()
    model_BW.state_values = model_BW_state_values_original.copy()
    Model_PK.state_values = Model_PK_state_values_original.copy()
    model_GI.state_values = model_GI_state_values_original.copy()
    model_BW.state_values[idx_state_fm] = Healthy_FM0_g
    model_BW.state_values[idx_state_ffm] = Healthy_FFM0_g
    simulation_ss_HealthyChow = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_HealthyChow_D12450J_ss], time_vector=np.linspace(0, 200, 200+1))
    simulation_ss_HealthyChow.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss_HealthyChow)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 0
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss_HealthyChow)
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss_HealthyChow)
    Model_PK = assign_state_values_from_simulation(Model_PK, simulation_ss_HealthyChow)

    Appetite_state_HealthyChow = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    BW_state_HealthyChow = np.asarray(model_BW.state_values, dtype=float).copy()
    PK_state_HealthyChow = np.asarray(Model_PK.state_values, dtype=float).copy()
    GI_state_HealthyChow = np.asarray(model_GI.state_values, dtype=float).copy()

    # HFD diseased
    Model_Appetite.state_values = Model_Appetite_state_values_original.copy()
    model_BW.state_values = model_BW_state_values_original.copy()
    Model_PK.state_values = Model_PK_state_values_original.copy()
    model_GI.state_values = model_GI_state_values_original.copy()
    model_BW.state_values[idx_state_fm] = Obese_FM0_g
    model_BW.state_values[idx_state_ffm] = Obese_FFM0_g
    simulation_ss_ObeseHFD = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ObeseHFD_D12492_ss], time_vector=np.linspace(0, 200, 200+1))
    simulation_ss_ObeseHFD.simulate()
    Model_Appetite = assign_state_values_from_simulation(Model_Appetite, simulation_ss_ObeseHFD)
    Model_Appetite.state_values[Model_Appetite.state_names.index("Hours")] = 0
    Model_Appetite.state_values[Model_Appetite.state_names.index("EI_AC")] = 0
    model_BW = assign_state_values_from_simulation(model_BW, simulation_ss_ObeseHFD)
    model_GI = assign_state_values_from_simulation(model_GI, simulation_ss_ObeseHFD)
    Model_PK = assign_state_values_from_simulation(Model_PK, simulation_ss_ObeseHFD)

    Appetite_state_ObeseHFD = np.asarray(Model_Appetite.state_values, dtype=float).copy()
    BW_state_ObeseHFD = np.asarray(model_BW.state_values, dtype=float).copy()
    PK_state_ObeseHFD = np.asarray(Model_PK.state_values, dtype=float).copy()
    GI_state_ObeseHFD = np.asarray(model_GI.state_values, dtype=float).copy()


    # ---------- SIMULATIONS ----------
    #  Healthy into chow (D12450J) 
    Model_Appetite.state_values = Appetite_state_HealthyChow.copy()
    model_BW.state_values       = BW_state_HealthyChow.copy()
    Model_PK.state_values       = PK_state_HealthyChow.copy()
    model_GI.state_values       = GI_state_HealthyChow.copy()
    simulation_HealthyChow_D12450J = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_HealthyChow_D12450J], time_vector=sim_t)
    simulation_HealthyChow_D12450J.simulate()
    #  Healthy into HFD (D12492) 
    Model_Appetite.state_values = Appetite_state_HealthyChow.copy()
    model_BW.state_values       = BW_state_HealthyChow.copy()
    Model_PK.state_values       = PK_state_HealthyChow.copy()
    model_GI.state_values       = GI_state_HealthyChow.copy()
    simulation_HealthyHFD_D12492 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_HealthyHFD_D12492], time_vector=sim_t)
    simulation_HealthyHFD_D12492.simulate()
    #  Healthy into HFD (D12331)
    Model_Appetite.state_values = Appetite_state_HealthyChow.copy()
    model_BW.state_values       = BW_state_HealthyChow.copy()
    Model_PK.state_values       = PK_state_HealthyChow.copy()
    model_GI.state_values       = GI_state_HealthyChow.copy()
    simulation_HealthyHFD_D12331 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_HealthyHFD_D12331], time_vector=sim_t)
    simulation_HealthyHFD_D12331.simulate()
    #  Healthy into control (D124512) 
    Model_Appetite.state_values = Appetite_state_HealthyChow.copy()
    model_BW.state_values       = BW_state_HealthyChow.copy()
    Model_PK.state_values       = PK_state_HealthyChow.copy()
    model_GI.state_values       = GI_state_HealthyChow.copy()
    simulation_HealthyControl_D124512 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_HealthyControl_D124512], time_vector=sim_t)
    simulation_HealthyControl_D124512.simulate()
    #  Healthy into Control (Teklad LM-485) 
    Model_Appetite.state_values = Appetite_state_HealthyChow.copy()
    model_BW.state_values       = BW_state_HealthyChow.copy()
    Model_PK.state_values       = PK_state_HealthyChow.copy()
    model_GI.state_values       = GI_state_HealthyChow.copy()
    simulation_HealthyControl_TekladLM_485 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_HealthyControl_TekladLM_485], time_vector=sim_t)
    simulation_HealthyControl_TekladLM_485.simulate()

    #  Obese into chow (D12450J) 
    Model_Appetite.state_values = Appetite_state_ObeseHFD.copy()
    model_BW.state_values       = BW_state_ObeseHFD.copy()
    Model_PK.state_values       = PK_state_ObeseHFD.copy()
    model_GI.state_values       = GI_state_ObeseHFD.copy()
    simulation_ObeseChow_D12450J = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ObeseChow_D12450J], time_vector=sim_t)
    simulation_ObeseChow_D12450J.simulate()
    #  Obese into HFD (D12492) 
    Model_Appetite.state_values = Appetite_state_ObeseHFD.copy()
    model_BW.state_values       = BW_state_ObeseHFD.copy()
    Model_PK.state_values       = PK_state_ObeseHFD.copy()
    model_GI.state_values       = GI_state_ObeseHFD.copy()
    simulation_ObeseHFD_D12492 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ObeseHFD_D12492], time_vector=sim_t)
    simulation_ObeseHFD_D12492.simulate()

    #  Obese into HFD (D12492) with semaglutide 2 nmol/kg QD 
    Model_Appetite.state_values = Appetite_state_ObeseHFD.copy()
    model_BW.state_values       = BW_state_ObeseHFD.copy()
    Model_PK.state_values       = PK_state_ObeseHFD.copy()
    model_GI.state_values       = GI_state_ObeseHFD.copy()
    simulation_ObeseHFD_Sema2 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ObeseHFD_Sema2], time_vector=sim_t)
    simulation_ObeseHFD_Sema2.simulate()
    #  Obese into HFD (D12492) with semaglutide 10 nmol/kg QD 
    Model_Appetite.state_values = Appetite_state_ObeseHFD.copy()
    model_BW.state_values       = BW_state_ObeseHFD.copy()
    Model_PK.state_values       = PK_state_ObeseHFD.copy()
    model_GI.state_values       = GI_state_ObeseHFD.copy()
    simulation_ObeseChow_Sema10 = sund.Simulation(time_unit="d", models=[Model_Appetite, model_BW, Model_PK, model_GI], activities=[activity_ObeseHFD_Sema10], time_vector=sim_t)
    simulation_ObeseChow_Sema10.simulate()


    # ---------- Restore original model states ----------
    Model_Appetite.state_values = Model_Appetite_state_values_original.copy()
    model_BW.state_values = model_BW_state_values_original.copy()
    Model_PK.state_values = Model_PK_state_values_original.copy()
    model_GI.state_values = model_GI_state_values_original.copy()

    DigitalPopulationSimulations = {
        "HealthyChow_D12450J": simulation_HealthyChow_D12450J,
        "HealthyHFD_D12492": simulation_HealthyHFD_D12492,
        "HealthyHFD_D12331": simulation_HealthyHFD_D12331,
        "HealthyControl_D124512": simulation_HealthyControl_D124512,
        "HealthyControl_TekladLM_485": simulation_HealthyControl_TekladLM_485,

        "ObeseChow_D12450J": simulation_ObeseChow_D12450J,
        "ObeseHFD_D12492": simulation_ObeseHFD_D12492,

        "ObeseHFD_Sema2": simulation_ObeseHFD_Sema2,
        "ObeseHFD_Sema10": simulation_ObeseChow_Sema10,
    }

    return DigitalPopulationSimulations