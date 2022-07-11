# This file generates the data used for Fig. 4 in M. Nägele, C. Schweizer, F. Roy, S. Filipp, 
# Effective non-local parity-dependent couplings in qubit chains (2022), https://arxiv.org/abs/2203.07331

# This file was run with the q-optimize package (https://github.com/q-optimize/c3) on the dev branch
# with HEAD at 8e747d5fb23d7c78041ec3af8a6f5869df30efee 

import pickle
import copy
import numpy as np
from scipy.linalg import expm


from c3.c3objs import Quantity as Qty

from c3.libraries import fidelities, hamiltonians, envelopes, algorithms
from c3.libraries.constants import PAULIS
from c3.optimizers.c1 import C1
from c3.system import chip
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.system.model import Model as Mdl
from c3.generator.generator import Generator as Gnr
from c3.generator import devices
from  c3.signal import pulse, gates
import c3.utils.qt_utils as qt_utils



def calc_t_fac(gamma: float) -> float:
    """gamma: between 0 and 1, scales transfer angle of FST,
    returns factor to scale time compared to theta=pi"""
    phi = np.pi * gamma
    return np.sqrt((np.pi - phi/2) * phi) / (np.pi/np.sqrt(2))

def calc_delta(gamma: float) -> float:
    """gamma: between 0 and 1, scales transfer angle of FST,
    returns estimated detuning of coupler drives compared to theta=pi"""
    phi = np.pi * gamma
    return -2 * (np.pi - phi) / np.sqrt((np.pi - phi / 2) * phi) * np.pi/(2*150e-9)/ (2*np.pi)

def create_prefect_para_gate(gen_list:list, angle: float)-> np.ndarray:
    """gen_list: List of lists, each sublist contains names of pauli operators used as generators,
    angle: angle for which the generator is applied,
    returns: unitary created by evolving under the generator for the specified angle"""
    generator = 0
    for gen in gen_list:
        kron = [PAULIS[key] for key in gen]
        generator += qt_utils.np_kron_n(kron)
    return expm(-1.0j / 2 * angle * generator)


# Define Model Parameters ----------------------------------------------------------------------
lindblad = False
dressed = False
max_excitations = 5

rise_time = Qty(0.3e-9, unit="s")
v2hz = Qty(1, unit="Hz 2pi/V")
sim_res = 25e9
awg_res = 2.4e9

generator_dict = {"rise_time": rise_time, 
                    "sim_res": sim_res,
                    "V_to_Hz" : v2hz,
                     "awg_res" : awg_res}

n_lvls = 3
anhar_coupler = -350e6
anhar_qubit = -300e6


wq1 = 5.05e+09
wq2 = 5.00e+09
wq3 = 5.075e9

wtc1 = 7269.69e6
wtc2 = 7293e6
        
q1_lvls = n_lvls
freq_q1 = Qty(wq1, min_val=2.0e9, max_val=10.0e9, unit="Hz 2pi")
anhar_q1 = Qty(anhar_qubit, min_val=-600e6, max_val=100.0e9, unit="Hz 2pi")
phi_0_q1 = Qty(1, min_val=1 * 0.9, max_val=1 * 1.1, unit="Wb")
fluxpoint_q1 = Qty(0, min_val=-5.0 * 1, max_val=5.0 * 1, unit="Wb")
d_q1 = Qty(1.5e-01, min_val=0.0, max_val=1, unit="")
q1_dict = {"freq": freq_q1, "anhar": anhar_q1, "hilbert_dim" : q1_lvls, "phi0": phi_0_q1,
            "d": d_q1, "phi" : fluxpoint_q1}

q2_lvls = n_lvls
freq_q2 = Qty(wq2, min_val=1e9, max_val=50.1e9, unit="Hz 2pi")
anhar_q2 = Qty(anhar_qubit, min_val=-600e6, max_val=100.0e9, unit="Hz 2pi")
phi_0_q2 = Qty(1, min_val=1 * 0.9, max_val=1 * 1.1, unit="Wb")
fluxpoint_q2 = Qty(0, min_val=-5.0 * 1, max_val=5.0 * 1, unit="Wb")
d_q2 = Qty(1.5e-01, min_val=0.0, max_val=1, unit="")
q2_dict = {"freq": freq_q2, "anhar": anhar_q2, "hilbert_dim" : q2_lvls, "phi0": phi_0_q2,
            "d": d_q2, "phi" : fluxpoint_q2}

q3_lvls = n_lvls
freq_q3 = Qty(wq3, min_val=2.0e9, max_val=10.0e9, unit="Hz 2pi")
anhar_q3 = Qty(anhar_qubit, min_val=-600e6, max_val=100.0e9, unit="Hz 2pi")
phi_0_q3 = Qty(1, min_val=1 * 0.9, max_val=1 * 1.1, unit="Wb")
fluxpoint_q3 = Qty(0, min_val=-5.0 * 1, max_val=5.0 * 1, unit="Wb")
d_q3 = Qty(1.5e-01, min_val=0.0, max_val=1, unit="")
q3_dict = {"freq": freq_q3, "anhar": anhar_q3, "hilbert_dim" : q3_lvls, "phi0": phi_0_q3,
            "d": d_q3, "phi" : fluxpoint_q3}

tc1_lvls = n_lvls
freq_tc1 = Qty(wtc1, min_val=2.0e9, max_val=100.0e9, unit="Hz 2pi")
anhar_tc1 = Qty(anhar_coupler, min_val=-600e6, max_val=100e9, unit="Hz 2pi")
phi_0_tc1 = Qty(1, min_val=1 * 0.9, max_val=1 * 1.1, unit="Wb")
fluxpoint_tc1 = Qty(0.3, min_val=-5.0 * 1, max_val=5.0 * 1, unit="Wb")
d_tc1 = Qty(0.5, min_val=0.0, max_val=1, unit="")
tc1_dict = {"freq": freq_tc1, "anhar": anhar_tc1, "hilbert_dim" : tc1_lvls, "phi0": phi_0_tc1,
            "d": d_tc1, "phi" : fluxpoint_tc1}

tc2_lvls = n_lvls
freq_tc2 = Qty(wtc2, min_val=2.0e9, max_val=100.0e9, unit="Hz 2pi")
anhar_tc2 = Qty(anhar_coupler, min_val=-600e6, max_val=100e9, unit="Hz 2pi")
phi_0_tc2 = Qty(1, min_val=1 * 0.9, max_val=1 * 1.1, unit="Wb")
fluxpoint_tc2 = Qty(0.3, min_val=-5.0 * 1, max_val=5.0 * 1, unit="Wb")
d_tc2 = Qty(0.5, min_val=0.0, max_val=1, unit="")
tc2_dict = {"freq": freq_tc2, "anhar": anhar_tc2, "hilbert_dim" : tc2_lvls, "phi0": phi_0_tc2,
            "d": d_tc2, "phi" : fluxpoint_tc2}


qubit_paras = [q1_dict, q2_dict, q3_dict]
  
coup =  100e6
coupling_strength_q1tc1= Qty(coup, min_val=-150e6, max_val=150e6, unit="Hz 2pi")
coupling_strength_tc1q2= Qty(-coup, min_val=-150e6, max_val=150e6, unit="Hz 2pi")
coupling_strength_q2tc2= Qty(coup, min_val=-150e6, max_val=150e6, unit="Hz 2pi")
coupling_strength_tc2q3= Qty(-coup, min_val=-150e6, max_val=150e6, unit="Hz 2pi")

direct_coupling = 6.6e6
coupling_strength_q1q2= Qty(-direct_coupling, min_val=-150e6, max_val=150e6, unit="Hz 2pi")
coupling_strength_q2q3= Qty(-direct_coupling, min_val=-150e6, max_val=150e6, unit="Hz 2pi")



coupling_q1tc1 = {"strength" : coupling_strength_q1tc1,
                "connected": ["Q1", "TC1"],
                "hamiltonian_func" : hamiltonians.int_YY}
coupling_tc1q2 = {"strength" : coupling_strength_tc1q2,
                "connected": ["Q2", "TC1"],
                "hamiltonian_func" : hamiltonians.int_YY}
coupling_q2tc2 = {"strength" : coupling_strength_q2tc2,
                "connected": ["Q2", "TC2"],
                "hamiltonian_func" : hamiltonians.int_YY}
coupling_tc2q3 = {"strength" : coupling_strength_tc2q3,
                "connected": ["Q3", "TC2"],
                "hamiltonian_func" : hamiltonians.int_YY}

coupling_q1q2 = {"strength" : coupling_strength_q1q2,
                "connected": ["Q1", "Q2"],
                "hamiltonian_func" : hamiltonians.int_YY}
coupling_q2q3 = {"strength" : coupling_strength_q2q3,
                "connected": ["Q2", "Q3"],
                "hamiltonian_func" : hamiltonians.int_YY}



coupling_paras = [coupling_q1tc1, coupling_tc1q2, coupling_q2tc2, coupling_tc2q3, coupling_q1q2, coupling_q2q3]


# Create Model----------------------------------------------------------------------

q1 = chip.Transmon(
            name="Q1",
            hilbert_dim=q1_dict["hilbert_dim"],
            freq=q1_dict["freq"],
            anhar=q1_dict["anhar"],
            phi=q1_dict["phi"],
            phi_0=q1_dict["phi0"],
            d=q1_dict["d"])
drive1 = chip.Drive(
                name="Q1_Xdrive",
                desc="Drive on Q1",
                connected=["Q1"],
                hamiltonian_func=hamiltonians.x_drive,)

q2 = chip.Transmon(
            name="Q2",
            hilbert_dim=q2_dict["hilbert_dim"],
            freq=q2_dict["freq"],
            anhar=q2_dict["anhar"],
            phi=q2_dict["phi"],
            phi_0=q2_dict["phi0"],
            d=q2_dict["d"])
drive2 = chip.Drive(
                name="Q2_Xdrive",
                desc="Drive on Q2",
                connected=["Q2"],
                hamiltonian_func=hamiltonians.x_drive,)


q3 = chip.Transmon(
            name="Q3",
            hilbert_dim=q3_dict["hilbert_dim"],
            freq=q3_dict["freq"],
            anhar=q3_dict["anhar"],
            phi=q3_dict["phi"],
            phi_0=q3_dict["phi0"],
            d=q3_dict["d"])
drive3 = chip.Drive(
                name="Q3_Xdrive",
                desc="Drive on Q3",
                connected=["Q3"],
                hamiltonian_func=hamiltonians.x_drive,)
tc1 = chip.Transmon(
            name="TC1",
            hilbert_dim=tc1_dict["hilbert_dim"],
            freq=tc1_dict["freq"],
            anhar=tc1_dict["anhar"],
            phi=tc1_dict["phi"],
            phi_0=tc1_dict["phi0"],
            d=tc1_dict["d"])
fluxtc1 = chip.Drive(
            name="TC1_Zdrive",
            connected=["TC1"],
            hamiltonian_func=hamiltonians.z_drive,)

tc2 = chip.Transmon(
            name="TC2",
            hilbert_dim=tc2_dict["hilbert_dim"],
            freq=tc2_dict["freq"],
            anhar=tc2_dict["anhar"],
            phi=tc2_dict["phi"],
            phi_0=tc2_dict["phi0"],
            d=tc2_dict["d"])
fluxtc2 = chip.Drive(
            name="TC2_Zdrive",
            connected=["TC2"],
            hamiltonian_func=hamiltonians.z_drive,)




phys_components = [q1, q2, q3, tc1, tc2]
line_components = [drive1, drive2, drive3, fluxtc1, fluxtc2]

for i, coupling_dict in enumerate(coupling_paras):
    c = chip.Coupling(
        name=f"c{i+1}",
        connected=coupling_dict["connected"]
        if "connected" in coupling_dict.keys()
        else None,
        strength=coupling_dict["strength"]
        if "strength" in coupling_dict.keys()
        else None,
        hamiltonian_func=coupling_dict["hamiltonian_func"]
        if "hamiltonian_func" in coupling_dict.keys()
        else None,)
    line_components.append(c)
model = Mdl(
        phys_components,
        line_components,
        [],
        max_excitations=max_excitations)
model.set_dressed(True)
model.set_lindbladian(lindblad)
model.update_model()


# Create Generator----------------------------------------------------------------------

lo = devices.LO(name="lo", resolution=sim_res)
awg = devices.AWG(name="awg", resolution=awg_res)
dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)
resp = devices.ResponseFFT(name="resp", rise_time=generator_dict["rise_time"], resolution=generator_dict["sim_res"],)
mixer = devices.Mixer(name="mixer")

fluxbiastc1 = devices.FluxTuning(
        name="fluxbias1",
        phi_0=tc1_dict["phi0"],
        phi=tc1_dict["phi"],
        d=tc1_dict["d"],
        omega_0=tc1_dict["freq"],
        anhar=tc1_dict["anhar"],
        )
fluxbiastc2 =devices.FluxTuning(
        name="fluxbias2",
        phi_0=tc2_dict["phi0"],
        phi=tc2_dict["phi"],
        d=tc2_dict["d"],
        omega_0=tc2_dict["freq"],
        anhar=tc2_dict["anhar"],
        )

v_to_hz = devices.VoltsToHertz(name="v2hz", V_to_Hz=generator_dict["V_to_Hz"],)
device_dict = {
    dev.name: dev
    for dev in [lo, awg, mixer, dig_to_an, resp, v_to_hz, fluxbiastc1, fluxbiastc2]
}

chains = {"TC1_Zdrive": ["lo", "awg", "dac", "resp", "mixer", "fluxbias1"], 
          "TC2_Zdrive": ["lo", "awg", "dac", "resp", "mixer", "fluxbias2"],
        }
for qubit in phys_components:
        if qubit.name[0] == "Q":
            chains[qubit.name + "_Xdrive"] = ["lo", "awg", "dac", "resp", "mixer", "v2hz"]

generator = Gnr(devices=device_dict, chains=chains)

# Gate parameters----------------------------------------------------------------------

end_time_All = 150e-9 * np.sqrt(2)

# Parameters for coupler 2
gate_time_23All = Qty(value=end_time_All,min_val=0.5 * end_time_All, max_val=1.5 * end_time_All,unit="s")
fluxamp_23All= Qty(value=0.053415465943562175, min_val=0.052, max_val=0.055, unit="V")
t_up_23All= Qty(value=4e-09, min_val=0.0 * end_time_All, max_val=0.5 * end_time_All, unit="s")
t_down_23All = Qty(value=end_time_All - 4e-09, min_val=0.5 * end_time_All, max_val=1.0 * end_time_All, unit="s")
risefall_23All= Qty(value=2e-09, min_val=0.0 * end_time_All, max_val=1.0 * end_time_All, unit="s")

flux_params_23_All = {"amp": fluxamp_23All, "t_final": gate_time_23All, "t_up": t_up_23All, "t_down": t_down_23All, "risefall": risefall_23All,}

flux_carrier_parameters_23_All ={"freq": Qty(value=85469419.28505991, min_val=0.e9, max_val=1e9, unit="Hz 2pi"),
                            "framechange": Qty(0, min_val=-np.pi, max_val=np.pi, unit="rad")}

# Parameters for coupler 1
gate_time_12All = Qty(value=end_time_All,min_val=0.5 * end_time_All, max_val=1.5 * end_time_All,unit="s")
fluxamp_12All= Qty(value=0.05286786931947968, min_val=0.051, max_val=0.054, unit="V")
t_up_12All= Qty(value=4e-09, min_val=0.0 * end_time_All, max_val=0.5 * end_time_All, unit="s")
t_down_12All = Qty(value=end_time_All - 4e-09, min_val=0.5 * end_time_All, max_val=1.0 * end_time_All, unit="s")
risefall_12All= Qty(value=2e-09, min_val=0.0 * end_time_All, max_val=1.0 * end_time_All, unit="s")

flux_params_12_All = {"amp": fluxamp_12All, "t_final": gate_time_12All, "t_up": t_up_12All, "t_down": t_down_12All, "risefall": risefall_12All,}

flux_carrier_parameters_12_All ={"freq": Qty(value=60614345.236043975, min_val=0.e9, max_val=1e9, unit="Hz 2pi"),
                            "framechange": Qty(0, min_val=-np.pi, max_val=np.pi, unit="rad")}


swap_All_gen = [[ "X", "Z", "X"], ["Y", "Z", "Y"]]
swapAll_perfect = create_prefect_para_gate(swap_All_gen, -np.pi/2)
framechanges_All =  ([ 0., 0., 0.])

# Define gate----------------------------------------------------------------------
no_drive_carrier_parameters = {
                "freq": Qty(0),
                "framechange": Qty(0, min_val=-np.pi, max_val=np.pi, unit="rad"),
            }
no_drive_carr = pulse.Carrier(
                name="carrier",
                desc="Frequency of the local oscillator",
                params=no_drive_carrier_parameters,
            )



flux_envtc2_All = pulse.Envelope(
    name="flux2",
    desc="Flux bias for tunable coupler",
    params=flux_params_23_All,
    shape=envelopes.flattop,
)
flux_carr2_All = pulse.Carrier(
                name="carrier",
                desc="Frequency of the local oscillator",
                params=flux_carrier_parameters_23_All,
            )

flux_envtc1_All = pulse.Envelope(
    name="flux2",
    desc="Flux bias for tunable coupler",
    params=flux_params_12_All,
    shape=envelopes.flattop,
)
flux_carr1_All = pulse.Carrier(
                name="carrier",
                desc="Frequency of the local oscillator",
                params=flux_carrier_parameters_12_All,
            )


gate_All = gates.Instruction(
    name="swapAll",
    targets=[1,2,3],
    t_start=0.0,
    t_end=flux_params_12_All["t_final"],
    channels=["TC1_Zdrive", "TC2_Zdrive", "Q1_Xdrive", "Q2_Xdrive", "Q3_Xdrive"],
    ideal=swapAll_perfect
)

gate_All.add_component(flux_envtc1_All, "TC1_Zdrive") 
gate_All.add_component(flux_carr1_All, "TC1_Zdrive")

gate_All.add_component(flux_envtc2_All, "TC2_Zdrive") 
gate_All.add_component(flux_carr2_All, "TC2_Zdrive")






nodrive_env = pulse.Envelope(
            name="no_drive", params={}, shape=envelopes.no_drive
        )

for framech, gate in zip([framechanges_All],[gate_All]):
    for i, q in enumerate(phys_components):
        if q.name[0] == "Q":
            carrier_parameters = {
                "freq": Qty(0),
                "framechange": Qty(framech[i], min_val=-np.pi, max_val=np.pi, unit="rad"),
            }
            carr = pulse.Carrier(
                name="carrier",
                desc="Frequency of the local oscillator",
                params=carrier_parameters,
            )
            gate.add_component(carr, q.name + "_Xdrive")
            gate.add_component(nodrive_env, q.name + "_Xdrive")

instructions = [gate_All]


# Make Experiment---------------------------------------------------------------------------

parameter_map = PMap(instructions=instructions, model=model, generator=generator)
exp = Exp(pmap=parameter_map)



# Optimize fid for different theta--------------------------------------------------------------------------

gamma_list = np.linspace(0.01, 1, 100)

fid_map_t = np.zeros(len(gamma_list))
params_t = []

prop_list = []



for i, gamma  in list(enumerate(gamma_list)):
    experiment = copy.deepcopy(exp)

    print(i)
    print(f"Starting better_Swap_all_{gamma=}, {i=}")
    
    # Calculate expected duration of gate
    t_fac = calc_t_fac(gamma)
    # Calculate expected detuning of coupler drives
    delta = calc_delta(gamma)

    print(f"{t_fac=}, {delta/1e6=}")


    t_final = Qty(t_fac * 150e-9 * np.sqrt(2))
    experiment.pmap.instructions['swapAll[1, 2, 3]'].t_end = t_final


    # Create target unitary
    swap_All_gen = [[ "X", "Z", "X"], ["Y", "Z", "Y"]]
    swapAll_perfect = create_prefect_para_gate(swap_All_gen, np.pi/2 * gamma)
    experiment.pmap.instructions['swapAll[1, 2, 3]'].ideal = swapAll_perfect


    # Set initial detuning
    wtc1 = exp.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['carrier'].params['freq'].get_value()/(2*np.pi) + delta
    wtc2 = exp.pmap.instructions['swapAll[1, 2, 3]'].comps['TC2_Zdrive']['carrier'].params['freq'].get_value()/(2*np.pi) + delta
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['carrier'].params['freq'].set_value(wtc1)
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC2_Zdrive']['carrier'].params['freq'].set_value(wtc2)

    # Scale pulse envelope
    tup = Qty(t_fac * exp.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['t_up'].get_value())
    tdown = Qty(t_fac * exp.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['t_down'].get_value())
    risefall = Qty(t_fac * exp.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['risefall'].get_value())

    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['t_final'] = t_final
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC2_Zdrive']['flux2'].params['t_final'] = t_final

    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['t_up'] = tup
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC2_Zdrive']['flux2'].params['t_up'] = tup

    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['t_down'] = tdown
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC2_Zdrive']['flux2'].params['t_down'] = tdown

    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC1_Zdrive']['flux2'].params['risefall'] = risefall
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['TC2_Zdrive']['flux2'].params['risefall'] = risefall

    
    # Optimize framechanges
    opt_gates = ["swapAll[1, 2, 3]"]
    gateset_opt_map=[
                    [("swapAll[1, 2, 3]", "Q1_Xdrive", "carrier", "framechange")],
                    [("swapAll[1, 2, 3]", "Q2_Xdrive", "carrier", "framechange")],
                    [("swapAll[1, 2, 3]", "Q3_Xdrive", "carrier", "framechange")],
                    ]

    experiment.pmap.set_opt_map(gateset_opt_map)

    opt = C1(
        dir_path="c3logs",
        fid_func=fidelities.average_infid_set,
        fid_subspace=["Q1", "Q2", "Q3"],
        pmap=experiment.pmap,
        algorithm=algorithms.lbfgs,
        run_name="better_Swap_all",
    )
    experiment.set_opt_gates(opt_gates)
    opt.set_exp(experiment)

    opt.optimize_controls()

    framech = opt.current_best_params
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['Q1_Xdrive']['carrier'].params['framechange'].set_limits(-np.pi-1, np.pi+1)
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['Q2_Xdrive']['carrier'].params['framechange'].set_limits(-np.pi-1, np.pi+1)
    experiment.pmap.instructions['swapAll[1, 2, 3]'].comps['Q3_Xdrive']['carrier'].params['framechange'].set_limits(-np.pi-1, np.pi+1)


    # Optimize framechanges and frequencies
    opt_gates = ["swapAll[1, 2, 3]"]
    gateset_opt_map=[
                    [("swapAll[1, 2, 3]", "Q1_Xdrive", "carrier", "framechange")],
                    [("swapAll[1, 2, 3]", "Q2_Xdrive", "carrier", "framechange")],
                    [("swapAll[1, 2, 3]", "Q3_Xdrive", "carrier", "framechange")],

                    [("swapAll[1, 2, 3]", "TC1_Zdrive", "carrier", "freq")],

                    [("swapAll[1, 2, 3]", "TC2_Zdrive", "carrier", "freq")],
                    ]

    experiment.pmap.set_opt_map(gateset_opt_map)

    opt = C1(
        dir_path="c3logs",
        fid_func=fidelities.average_infid_set,
        fid_subspace=["Q1", "Q2", "Q3"],
        pmap=experiment.pmap,
        algorithm=algorithms.lbfgs,
        run_name="better_Swap_all",
    )
    experiment.set_opt_gates(opt_gates)
    opt.set_exp(experiment)

    opt.optimize_controls()
    

    # Save results
    fid_map_t[i] = opt.current_best_goal
    params_t.append(opt.current_best_params)

    prop = experiment.compute_propagators()
    prop_list.append(prop)
    
    with open(f"fid_phi_t_test_{i}.obj","wb") as filehandler:
        pickle.dump(fid_map_t, filehandler)

    with open(f"paras_phi_t_{i}.obj","wb") as filehandler:
        pickle.dump(params_t, filehandler)

    with open(f"prop_list.obj","wb") as filehandler:
        pickle.dump(prop_list, filehandler)


