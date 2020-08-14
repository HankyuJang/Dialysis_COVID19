"""
Dialysis COVID19 simulation 

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: August, 2020

Run 12 sets of simulations (each set with 500 replicates) on the baseline
- 2 scenarios: Scenario 1 (infection source = morning patient), Scenario 2 (infection source = morning HCW) 
- 2 shedding models: exp/exp(20%), exp/exp(60%)
- 3 Target R0s: 2.0, 2.5, 3.0

Note: This script uses multiprocessing module which runs in many cores.
Specify the number of cores to use in the `multiprocessing.Pool` method.
"""

import argparse
import pandas as pd
import numpy as np
import random as rd
import datetime 
# from COVID19_simulator import *
from COVID19_simulator_v5 import *
import multiprocessing
from functools import partial

def simulate(W, T, inf, alpha, beta, gamma, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hcw_hcw_contact, hcw_patient_contact, patient_patient_contact, morning_patients, morning_hcws, rep):
    np.random.seed(rd.randint(0, 10000000))
    simul = Simulation(W, T, inf, alpha, beta, gamma, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hcw_hcw_contact, hcw_patient_contact, patient_patient_contact, morning_patients, morning_hcws)
    simul.simulate()
    return simul.n_inf_rec, simul.transmission_route, simul.population, simul.R0, simul.generation_time

# Make hcw_hcw_contact in the same shape as other contact arrays, then zero out the contacts on Sunday
def hhc_expand_dims(hcw_hcw_contact, simulation_period):
    hcw_hcw_contact = np.repeat(np.expand_dims(hcw_hcw_contact, axis=0), simulation_period, axis=0)
    hcw_hcw_contact[6,:,:,:] = 0
    hcw_hcw_contact[13,:,:,:] = 0
    hcw_hcw_contact[20,:,:,:] = 0
    hcw_hcw_contact[27,:,:,:] = 0
    return hcw_hcw_contact

# morning hcws: any HCW whose badge start time is before noon.
def get_morning_hcws(df_hcw_locations, timestep_at_noon):
    df_temp = df_hcw_locations[df_hcw_locations.time < timestep_at_noon]
    return df_temp.ID.unique() - 1 # hcw ID in df_hcw_locations file starts from 1

# morning patients: patients where the dialysis sessin starts before 9 am.
# Note: all the morning sessions in our data starts before 9 am.
# dim0: simulation days, dim1: number of hcws, dim2: number of patients, dim3: total timesteps in a day
# >>> hpc_original.shape  # for day 10
# (30, 11, 40, 6822)
def get_morning_patients(hpc_original, timestep_at_nine):
    return hpc_original[0, :, :, :timestep_at_nine].sum(axis=(0,2)).nonzero()[0]

def print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx):
    R0_dict = {0:"2.0", 1:"2.5", 2:"3.0"}
    shedding_dict = {2:"exp/exp(20%)", 3:"exp/exp(60%)"}
    print("Shedding model: {}, Target R0: {}".format(shedding_dict[Dtype], R0_dict[alpha_idx]))
    print("Oberved R0: {:.2f}".format(R0.mean()))
    transmission_source_in_P = (n_inf_rec[:,0,0,:] + n_inf_rec[:,0,1,:]).sum()
    transmission_source_in_S = (n_inf_rec[:,1,0,:] + n_inf_rec[:,1,1,:]).sum()
    transmission_total = transmission_source_in_P + transmission_source_in_S
    print("Infections, when source in P (presymptomatic): {:.2f}".format(transmission_source_in_P / repetition))
    print("Infections, when source in S (symptomatic): {:.2f}".format(transmission_source_in_S / repetition))
    print("Percentage of transmission occurring prior to symptom onset: {:.2f}%".format(100 * transmission_source_in_P / transmission_total))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    parser.add_argument('-cpu', '--cpu', type=int, default=60,
                        help= 'number of cpu to use for simulation')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance
    cpu = args.cpu

    # Read date_time.txt that contains start, end times
    # df starts from 0. Day1 = 0th row. Day10 = 9th row
    df = pd.read_table("dialysis/data/date_time.txt", sep='.', header=None, names = ['year','month','day','hour','minute','second'])
    row = df.iloc[day-1]
    start_time_of_day = datetime.datetime(row.year, row.month, row.day, row.hour, row.minute, row.second)

    noon = datetime.datetime(row.year, row.month, row.day, 12, 0, 0)
    timestep_at_noon = (noon - start_time_of_day).seconds // 8

    nine = datetime.datetime(row.year, row.month, row.day, 9, 0, 0)
    timestep_at_nine = (nine - start_time_of_day).seconds // 8
    
    df_hcw_locations = pd.read_csv("dialysis/data/HCP_locations/latent_positions_day_{}.csv".format(day))

    # Loading precomputed contact arrays: these are boolean, which indicates either there's a contact between two agents at a typical timestep of a day.
    # >>> hpc_original.shape
    # (30, 11, 40, 6822)  # for day 10
    # >>> ppc_original.shape
    # (30, 40, 40, 6822)  # for day 10
    npzfile = np.load("dialysis/contact_data/patient_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hpc_original = npzfile["hcw_patient_contact_arrays"]
    ppc_original = npzfile["patient_patient_contact_arrays"]
    npzfile.close()

    # hhc (original)
    # HCW-HCW contacts are divided into 4: (1) same chair, (2) adjacent chair, (3) both at the center, (4) or other places.
    # If you sum up these four HCW-HCW contacts, it becomes hhc_total array.
    npzfile = np.load("dialysis/contact_data/hhc_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hhc_same_chair = npzfile["hhc_same_chair"]
    hhc_adj_chair = npzfile["hhc_adj_chair"]
    hhc_both_center = npzfile["hhc_both_center"]
    hhc_other_places = npzfile["hhc_other_places"]
    hhc_total = npzfile["hhc_total"]
    npzfile.close()

    # hhc, ppc (reduced)
    # I've precomputed these reduced contacts; these are used in combination with hhc original contacts to properly reduce contacts 
    # only on specific spaces in the unit.
    npzfile = np.load("dialysis/contact_data/contact_arrays_sd_mc_day{}_{}ft.npz".format(day, contact_distance))
    hhc_adj_chair_rr25 = npzfile["hhc_adj_chair_rr25"]
    hhc_both_center_rr25 = npzfile["hhc_both_center_rr25"]
    hhc_other_places_rr25 = npzfile["hhc_other_places_rr25"]
    ppc_rr25 = npzfile["ppc_rr25"]
    hhc_adj_chair_rr50 = npzfile["hhc_adj_chair_rr50"]
    hhc_both_center_rr50 = npzfile["hhc_both_center_rr50"]
    hhc_other_places_rr50 = npzfile["hhc_other_places_rr50"]
    ppc_rr50 = npzfile["ppc_rr50"]
    hhc_adj_chair_rr75 = npzfile["hhc_adj_chair_rr75"]
    hhc_both_center_rr75 = npzfile["hhc_both_center_rr75"]
    hhc_other_places_rr75 = npzfile["hhc_other_places_rr75"]
    ppc_rr75 = npzfile["ppc_rr75"]
    npzfile.close()

    simulation_period = hpc_original.shape[0]
    n_hcw = hpc_original.shape[1]
    n_patient = hpc_original.shape[2]

    # Simulation parameters (these are default parameters)
    # Note that for the interventions that require differnet parameter, then it's specified
    # before the simulation. Refer to Simulator script `COVID19_simulator_v5` for details.
    W = 6
    T = 7
    inf = 1 # this parameter is not used in current simulator. It does nothing
    QC = 1 # HCW quarantine compliance rate
    asymp_rate = 0.4 # 20 % of infections remain asymptomatic
    asymp_shedding = 0.75 # asymptomatic shed 50% less 
    QS = W + 1 # this parameter also do nothing. The qurantine start dates for the two hcw surveillance strategies are directly implemented in the simulator.
    QT = 14 # HCW quarantine period. 14 days.
    Dtype = 2 # type of the shedding model. 2 is exp/exp(5%) model and 3 is exp/exp(35%) model.
    k=1 # number of HCWs to replace on the early replacement strategy.

    # Current attack rate (active cases / population) of Johnson County, Iowa as of Aug 5, 2020.
    # roughly 0.00347 (525 / 151140)
    community_attack_rate = 525 / 151140
    # dim0: hcw mask efficacy, dim1: patient mask efficacy, dim2: efficacy of N95 (this is added just for Bppp intervention). In general, change dim0 and dim1 to control the efficacy of mask on hcws and patient.
    mask_efficacy = np.array([0.4, 0.4, 0.93])
    # The set of interventions to impose - change index of the array to True if you want to impose certain intervention.
    # intervention[0,0]; HCW self quarantine
    # intervention[0,1]; HCW active surveillance
    # intervention[0,2]; HCW mask intervention
    # intervention[0,3]; HCW early replacement
    # intervention[1,1]; Patient isolation (first symptomatic patient)
    # intervention[1,2]; Patient mask intervention
    # intervention[2,2]; N95 all HCWs upon detection of 1st symptomatic patient
    # You can eaily add other interventions to the Simulator class in this way.
    intervention = np.zeros((3, 5)).astype(bool)

    Dtype_list = [0, 1, 2, 3] # 0 and 1 are not used in this project.
    # This is the alpha (scaling down parameters). naming is bad for this, since I simply plugged into the parater that was not used in the simulator.
    df_alpha = pd.read_csv("dialysis/data/df_alpha.csv", index_col=0)
    alpha_array = df_alpha.values
    # Beta and gamma are ramp-up and ramp-down factors.
    df_beta = pd.read_csv("dialysis/data/df_beta.csv", index_col=0)
    beta_array = df_beta.values
    df_gamma = pd.read_csv("dialysis/data/df_gamma.csv", index_col=0)
    gamma_array = df_gamma.values

    QC_list0 = [0.5, 0.7] # HCW quarantine compliance rates for (self-quaranine)
    QC_list1 = [1] # compliance rate for active surveillance
    rr_list = [0.25, 0.5, 0.75, 1.00] # contact removal rats
    k_list_H3P1 = [1, 2, 3, 4, 5] # H3P1 is Patient isolation & early replacement. Naming follows the intervention array (first row=hcw, second row=patient third row=somewhat mixed). k is the nubmer of HCws to replace early
    k_list_Bpp = [1]

    # Multiprocess
    repetition = 500 # number of replicates per each setting of simulation.
    n_cpu = cpu # number of cpu to use. Note that this script uses multi processors

    rep = range(repetition)
    
    ##########################################
    # These are arrays to store lots of information during the simulation
    # Later used for drawing plots and tables

    # Initialize arrays to save results
    # dim1: [during incubation period, during symptomatic period, outside source (coming in infected)]
    # dim2: [hcw_infected, patient_infected, hcw_recovered, patient_recovered]
    n_inf_rec = np.zeros((repetition, 3, 4, simulation_period)).astype(int)
    # dim1: [h->p, p->h, h->h, p->p]
    transmission_route = np.zeros((repetition, 4, simulation_period)).astype(int)
    population = np.zeros((repetition, simulation_period)).astype(int)
    R0 = np.zeros((repetition)).astype(int)
    generation_time = np.zeros((repetition)).astype(int)
    ##########################################

    n_Dtype = alpha_array.shape[0]
    n_cases = alpha_array.shape[1]
    n_QC0 = len(QC_list0)
    n_QC1 = len(QC_list1)
    n_rr = len(rr_list)
    n_k_H3P1 = len(k_list_H3P1)
    n_k_Bpp = len(k_list_Bpp)

    ############################
    # Arrays below saves the result on each intervention setting.
    # In this way, I can store whatever information I need during the Simulation
    ############################
    # Baseline
    B_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, 3, 4, simulation_period))
    B_transmission_route = np.zeros((repetition, n_Dtype, n_cases, 4, simulation_period))
    B_population = np.zeros((repetition, n_Dtype, n_cases, simulation_period))
    B_R0 = np.zeros((repetition, n_Dtype, n_cases))
    B_generation_time = np.zeros((repetition, n_Dtype, n_cases))

    counter_seed = 0
    # scenario=0: source=patient
    # scenario=1: source=hcw

    ####################################################################################################
    # Simulation start
    ####################################################################################################
    # Run simulations on Dtype 2, 3 and R0 = 2, 2.5, 3 (alpha_idx=0, 1, 2)
    for scenario in range(2):
        print("*"*40)
        if scenario == 0: # This is Scenario 1 in the paper
            print("Infection source: Patient")
            # Get morning patients to use as infection source in the simulation
            morning_patients = get_morning_patients(hpc_original, timestep_at_nine)
            morning_hcws = np.array([])
            print("ID of morning session patients. One of them are selected in the Simulator as the infection source.")
            print(morning_patients)
        elif scenario == 1: # This is Scenario 2 in the paper
            print("Infection source: HCW")
            # Get morning hcws to use as infection source in the simulation
            morning_patients = np.array([])
            morning_hcws = get_morning_hcws(df_hcw_locations, timestep_at_noon)
            print("ID of morning hcws. One of them are selected in the Simulator as the infection source.")
            print(morning_hcws)

        for Dtype_idx, Dtype in enumerate(Dtype_list):
            # We use two shedding models that are exp/exp(20%) and exp/exp(60%) that are Dtype 2 and 3, respectively.
            if Dtype_idx in [0, 1]:
                continue
            for alpha_idx, alpha in enumerate(alpha_array[Dtype_idx]):
                beta = beta_array[Dtype_idx, alpha_idx]
                gamma = gamma_array[Dtype_idx, alpha_idx]
                # We run total set of simulations when target R0=3
                # if alpha_idx in [0, 1]:
                    # continue

                print("*"*40)
                print("Simulation: Baseline")
                intervention[:] = False
                mask_efficacy = np.array([0.4, 0.4, 0.93]) # default mask setting. agents don't wear masks unless activated by intervention[:,2]
                hhc = hhc_expand_dims(hhc_total, simulation_period)
                hpc = hpc_original
                ppc = ppc_original

                rd.seed(counter_seed)
                counter_seed += 1

                pool = multiprocessing.Pool(processes=n_cpu)
                func = partial(simulate, W, T, inf, alpha, beta, gamma, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hhc, hpc, ppc, morning_patients, morning_hcws)
                r = pool.map(func, rep)
                pool.close()

                for i, result in enumerate(r):
                    n_inf_rec[i] = result[0]
                    transmission_route[i] = result[1]
                    population[i] = result[2]
                    R0[i] = result[3]
                    generation_time[i] = result[4]

                B_n_inf_rec[:, Dtype_idx, alpha_idx, :, :, :] = n_inf_rec
                B_transmission_route[:, Dtype_idx, alpha_idx, :, :] = transmission_route
                B_population[:, Dtype_idx, alpha_idx, :] = population
                B_R0[:, Dtype_idx, alpha_idx] = R0
                B_generation_time[:, Dtype_idx, alpha_idx] = generation_time / R0
                print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

        np.savez("dialysis/results/day{}/baseline_scenario{}".format(day, scenario),
                B_n_inf_rec = B_n_inf_rec,
                B_transmission_route = B_transmission_route,
                B_population = B_population,
                B_R0 = B_R0,
                B_generation_time = B_generation_time
                )
            ####################################################################################################
            # Interventon on HCW end
            ####################################################################################################
