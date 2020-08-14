"""
Dialysis COVID19 simulation 

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: August, 2020

Run simulations on all interventions in the paper 
Scenario 1 (infection source = morning patient) 
Scenario 2 (infection source = morning HCW)

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

    # H0: Self quarantine without masks
    H0_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_QC0, 3, 4, simulation_period))
    H0_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_QC0, 4, simulation_period))
    H0_population = np.zeros((repetition, n_Dtype, n_cases, n_QC0, simulation_period))
    H0_R0 = np.zeros((repetition, n_Dtype, n_cases, n_QC0))
    H0_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_QC0))

    # H1: Active surveillance without masks
    H1_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_QC1, 3, 4, simulation_period))
    H1_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_QC1, 4, simulation_period))
    H1_population = np.zeros((repetition, n_Dtype, n_cases, n_QC1, simulation_period))
    H1_R0 = np.zeros((repetition, n_Dtype, n_cases, n_QC1))
    H1_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_QC1))

    # P2: Patients surgical mask on
    P2_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, 3, 4, simulation_period))
    P2_transmission_route = np.zeros((repetition, n_Dtype, n_cases, 4, simulation_period))
    P2_population = np.zeros((repetition, n_Dtype, n_cases, simulation_period))
    P2_R0 = np.zeros((repetition, n_Dtype, n_cases))
    P2_generation_time = np.zeros((repetition, n_Dtype, n_cases))

    # H2P2v2: Patients surgical mask on, HCPs surgical mask on
    H2P2v2_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, 3, 4, simulation_period))
    H2P2v2_transmission_route = np.zeros((repetition, n_Dtype, n_cases, 4, simulation_period))
    H2P2v2_population = np.zeros((repetition, n_Dtype, n_cases, simulation_period))
    H2P2v2_R0 = np.zeros((repetition, n_Dtype, n_cases))
    H2P2v2_generation_time = np.zeros((repetition, n_Dtype, n_cases))

    # H2P2: Patients surgical mask on, HCPs N95 mask on
    H2P2_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, 3, 4, simulation_period))
    H2P2_transmission_route = np.zeros((repetition, n_Dtype, n_cases, 4, simulation_period))
    H2P2_population = np.zeros((repetition, n_Dtype, n_cases, simulation_period))
    H2P2_R0 = np.zeros((repetition, n_Dtype, n_cases))
    H2P2_generation_time = np.zeros((repetition, n_Dtype, n_cases))

    # N0 Social distancing
    N0_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_rr, 3, 4, simulation_period))
    N0_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_rr, 4, simulation_period))
    N0_population = np.zeros((repetition, n_Dtype, n_cases, n_rr, simulation_period))
    N0_R0 = np.zeros((repetition, n_Dtype, n_cases, n_rr))
    N0_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_rr))

    # N1 Move chairs apart
    N1_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_rr, 3, 4, simulation_period))
    N1_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_rr, 4, simulation_period))
    N1_population = np.zeros((repetition, n_Dtype, n_cases, n_rr, simulation_period))
    N1_R0 = np.zeros((repetition, n_Dtype, n_cases, n_rr))
    N1_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_rr))

    # H3P1 symptomatic patient isolation and early HCP replacement of k HCPs
    H3P1_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_k_H3P1, 3, 4, simulation_period))
    H3P1_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_k_H3P1, 4, simulation_period))
    H3P1_population = np.zeros((repetition, n_Dtype, n_cases, n_k_H3P1, simulation_period))
    H3P1_R0 = np.zeros((repetition, n_Dtype, n_cases, n_k_H3P1))
    H3P1_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_k_H3P1))

    # Bp: social distancing + move chairs apart + surgical masks for everyone
    Bp_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, 3, 4, simulation_period))
    Bp_transmission_route = np.zeros((repetition, n_Dtype, n_cases, 4, simulation_period))
    Bp_population = np.zeros((repetition, n_Dtype, n_cases, simulation_period))
    Bp_R0 = np.zeros((repetition, n_Dtype, n_cases))
    Bp_generation_time = np.zeros((repetition, n_Dtype, n_cases))

    # Bpp: social distancing + move chairs apart + surgical masks for everyone + early replacement
    Bpp_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp, 3, 4, simulation_period))
    Bpp_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp, 4, simulation_period))
    Bpp_population = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp, simulation_period))
    Bpp_R0 = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp))
    Bpp_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp))

    # Bppp: social distancing + move chairs apart + surgical masks for everyone + early replacement + N95 on HCPs for 2 weeks since the symptomatic patient
    Bppp_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp, 3, 4, simulation_period))
    Bppp_transmission_route = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp, 4, simulation_period))
    Bppp_population = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp, simulation_period))
    Bppp_R0 = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp))
    Bppp_generation_time = np.zeros((repetition, n_Dtype, n_cases, n_k_Bpp))

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

                print("*"*40)
                print("Simulation: H0: Self quarantine without masks")
                intervention[:] = False
                intervention[0, 0] = True
                mask_efficacy = np.array([0.4, 0.4, 0.93]) # default mask setting. agents don't wear masks unless activated by intervention[:,2]
                hhc = hhc_expand_dims(hhc_total, simulation_period)
                hpc = hpc_original
                ppc = ppc_original

                for QC_idx, QC in enumerate(QC_list0):
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

                    H0_n_inf_rec[:, Dtype_idx, alpha_idx, QC_idx, :, :, :] = n_inf_rec
                    H0_transmission_route[:, Dtype_idx, alpha_idx, QC_idx, :, :] = transmission_route
                    H0_population[:, Dtype_idx, alpha_idx, QC_idx, :] = population
                    H0_R0[:, Dtype_idx, alpha_idx, QC_idx] = R0
                    H0_generation_time[:, Dtype_idx, alpha_idx, QC_idx] = generation_time / R0
                    print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)
                
                print("*"*40)
                print("Simulation: H1: Active surveillance without masks")
                QC=1 # this doesn't do anything, but just keep it here to set QC = 1 
                intervention[:] = False
                intervention[0, 1] = True
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

                H1_n_inf_rec[:, Dtype_idx, alpha_idx, 0, :, :, :] = n_inf_rec
                H1_transmission_route[:, Dtype_idx, alpha_idx, 0, :, :] = transmission_route
                H1_population[:, Dtype_idx, alpha_idx, 0, :] = population
                H1_R0[:, Dtype_idx, alpha_idx, 0] = R0
                H1_generation_time[:, Dtype_idx, alpha_idx, 0] = generation_time / R0
                print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: P2: Patients surgical mask on")
                intervention[:] = False
                intervention[1, 2] = True
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

                P2_n_inf_rec[:, Dtype_idx, alpha_idx, :, :, :] = n_inf_rec
                P2_transmission_route[:, Dtype_idx, alpha_idx, :, :] = transmission_route
                P2_population[:, Dtype_idx, alpha_idx, :] = population
                P2_R0[:, Dtype_idx, alpha_idx] = R0
                P2_generation_time[:, Dtype_idx, alpha_idx] = generation_time / R0
                print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)


                print("*"*40)
                print("Simulation: H2P2v2: Patients surgical mask on, HCPs surgical mask on")
                intervention[:] = False
                intervention[1, 2] = True
                intervention[0, 2] = True
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

                H2P2v2_n_inf_rec[:, Dtype_idx, alpha_idx, :, :, :] = n_inf_rec
                H2P2v2_transmission_route[:, Dtype_idx, alpha_idx, :, :] = transmission_route
                H2P2v2_population[:, Dtype_idx, alpha_idx, :] = population
                H2P2v2_R0[:, Dtype_idx, alpha_idx] = R0
                H2P2v2_generation_time[:, Dtype_idx, alpha_idx] = generation_time / R0
                print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: H2P2: Patients surgical mask on, HCPs N95 respirator on")
                intervention[:] = False
                intervention[1, 2] = True
                intervention[0, 2] = True
                mask_efficacy = np.array([0.93, 0.4, 0.93])
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

                H2P2_n_inf_rec[:, Dtype_idx, alpha_idx, :, :, :] = n_inf_rec
                H2P2_transmission_route[:, Dtype_idx, alpha_idx, :, :] = transmission_route
                H2P2_population[:, Dtype_idx, alpha_idx, :] = population
                H2P2_R0[:, Dtype_idx, alpha_idx] = R0
                H2P2_generation_time[:, Dtype_idx, alpha_idx] = generation_time / R0
                print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: N0: HCP Social Distancing")
                intervention[:] = False
                mask_efficacy = np.array([0.4, 0.4, 0.93]) # default mask setting. agents don't wear masks unless activated by intervention[:,2]
                hpc = hpc_original
                ppc = ppc_original

                for rr_idx, rr in enumerate(rr_list):
                    if rr_idx == 0:
                        hhc = hhc_same_chair + hhc_adj_chair + hhc_both_center_rr25 + hhc_other_places_rr25
                        hhc = hhc_expand_dims(hhc, simulation_period)
                    elif rr_idx == 1:
                        hhc = hhc_same_chair + hhc_adj_chair + hhc_both_center_rr50 + hhc_other_places_rr50
                        hhc = hhc_expand_dims(hhc, simulation_period)
                    elif rr_idx == 2:
                        hhc = hhc_same_chair + hhc_adj_chair + hhc_both_center_rr75 + hhc_other_places_rr75
                        hhc = hhc_expand_dims(hhc, simulation_period)
                    elif rr_idx == 3:
                        hhc = hhc_same_chair + hhc_adj_chair
                        hhc = hhc_expand_dims(hhc, simulation_period)

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

                    N0_n_inf_rec[:, Dtype_idx, alpha_idx, rr_idx, :, :, :] = n_inf_rec
                    N0_transmission_route[:, Dtype_idx, alpha_idx, rr_idx, :, :] = transmission_route
                    N0_population[:, Dtype_idx, alpha_idx, rr_idx, :] = population
                    N0_R0[:, Dtype_idx, alpha_idx, rr_idx] = R0
                    N0_generation_time[:, Dtype_idx, alpha_idx, rr_idx] = generation_time / R0
                    print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: N1: move chair further apart")
                intervention[:] = False
                mask_efficacy = np.array([0.4, 0.4, 0.93]) # default mask setting. agents don't wear masks unless activated by intervention[:,2]
                hpc = hpc_original
                for rr_idx, rr in enumerate(rr_list):
                    if rr_idx == 0:
                        hhc = hhc_same_chair + hhc_adj_chair_rr25 + hhc_both_center + hhc_other_places
                        hhc = hhc_expand_dims(hhc, simulation_period)
                        ppc = ppc_rr25
                    elif rr_idx == 1:
                        hhc = hhc_same_chair + hhc_adj_chair_rr50 + hhc_both_center + hhc_other_places
                        hhc = hhc_expand_dims(hhc, simulation_period)
                        ppc = ppc_rr50
                    elif rr_idx == 2:
                        hhc = hhc_same_chair + hhc_adj_chair_rr75 + hhc_both_center + hhc_other_places
                        hhc = hhc_expand_dims(hhc, simulation_period)
                        ppc = ppc_rr75
                    elif rr_idx == 3:
                        hhc = hhc_same_chair + hhc_both_center + hhc_other_places
                        hhc = hhc_expand_dims(hhc, simulation_period)
                        ppc = np.zeros((ppc_rr75.shape)).astype(bool)

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

                    N1_n_inf_rec[:, Dtype_idx, alpha_idx, rr_idx, :, :, :] = n_inf_rec
                    N1_transmission_route[:, Dtype_idx, alpha_idx, rr_idx, :, :] = transmission_route
                    N1_population[:, Dtype_idx, alpha_idx, rr_idx, :] = population
                    N1_R0[:, Dtype_idx, alpha_idx, rr_idx] = R0
                    N1_generation_time[:, Dtype_idx, alpha_idx, rr_idx] = generation_time / R0
                    print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: H3P1: one patient isolation & top k HCW replacement")
                mask_efficacy = np.array([0.4, 0.4, 0.93])
                intervention[:] = False
                intervention[0,3] = True
                intervention[1,1] = True
                ppc = ppc_original
                hpc = hpc_original
                hhc = hhc_expand_dims(hhc_total, simulation_period)
                for k_idx, k in enumerate(k_list_H3P1):
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

                    H3P1_n_inf_rec[:, Dtype_idx, alpha_idx, k_idx, :, :, :] = n_inf_rec
                    H3P1_transmission_route[:, Dtype_idx, alpha_idx, k_idx, :, :] = transmission_route
                    H3P1_population[:, Dtype_idx, alpha_idx, k_idx, :] = population
                    H3P1_R0[:, Dtype_idx, alpha_idx, k_idx] = R0
                    H3P1_generation_time[:, Dtype_idx, alpha_idx, k_idx] = generation_time / R0
                    print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: Bp: social distancing (25%) + move chairs apart (75%) + surgical masks for everyone")
                mask_efficacy = np.array([0.4, 0.4, 0.93])
                intervention[:] = False
                intervention[0, 2] = True
                intervention[1, 2] = True
                hpc = hpc_original
                hhc = hhc_same_chair + hhc_adj_chair_rr75 + hhc_both_center_rr25 + hhc_other_places_rr25
                hhc = hhc_expand_dims(hhc, simulation_period)
                ppc = ppc_rr75

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

                Bp_n_inf_rec[:, Dtype_idx, alpha_idx, :, :, :] = n_inf_rec
                Bp_transmission_route[:, Dtype_idx, alpha_idx, :, :] = transmission_route
                Bp_population[:, Dtype_idx, alpha_idx, :] = population
                Bp_R0[:, Dtype_idx, alpha_idx] = R0
                Bp_generation_time[:, Dtype_idx, alpha_idx] = generation_time / R0
                print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: Bpp: social distancing (25%) + move chairs apart (75%) + surgical masks for everyone + one sympatomatic patient isolation + early replacement of k HCPs")
                mask_efficacy = np.array([0.4, 0.4, 0.93])
                intervention[:] = False
                intervention[0,3] = True
                intervention[1,1] = True
                intervention[0,2] = True
                intervention[1,2] = True
                hpc = hpc_original
                hhc = hhc_same_chair + hhc_adj_chair_rr75 + hhc_both_center_rr25 + hhc_other_places_rr25
                hhc = hhc_expand_dims(hhc, simulation_period)
                ppc = ppc_rr75
                for k_idx, k in enumerate(k_list_Bpp):
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

                    Bpp_n_inf_rec[:, Dtype_idx, alpha_idx, k_idx, :, :, :] = n_inf_rec
                    Bpp_transmission_route[:, Dtype_idx, alpha_idx, k_idx, :, :] = transmission_route
                    Bpp_population[:, Dtype_idx, alpha_idx, k_idx, :] = population
                    Bpp_R0[:, Dtype_idx, alpha_idx, k_idx] = R0
                    Bpp_generation_time[:, Dtype_idx, alpha_idx, k_idx] = generation_time / R0
                    print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

                print("*"*40)
                print("Simulation: Bppp: social distancing (25%) + move chairs apart (75%) + surgical masks for everyone + one sympatomatic patient isolation + early replacement of k HCPs + N95 for HCPs for 2 weeks when the first patient start showing symptoms")
                mask_efficacy = np.array([0.4, 0.4, 0.93])
                intervention[:] = False
                intervention[0,3] = True
                intervention[1,1] = True
                intervention[0,2] = True
                intervention[1,2] = True
                intervention[2,2] = True
                hpc = hpc_original
                hhc = hhc_same_chair + hhc_adj_chair_rr75 + hhc_both_center_rr25 + hhc_other_places_rr25
                hhc = hhc_expand_dims(hhc, simulation_period)
                ppc = ppc_rr75
                for k_idx, k in enumerate(k_list_Bpp):
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

                    Bppp_n_inf_rec[:, Dtype_idx, alpha_idx, k_idx, :, :, :] = n_inf_rec
                    Bppp_transmission_route[:, Dtype_idx, alpha_idx, k_idx, :, :] = transmission_route
                    Bppp_population[:, Dtype_idx, alpha_idx, k_idx, :] = population
                    Bppp_R0[:, Dtype_idx, alpha_idx, k_idx] = R0
                    Bppp_generation_time[:, Dtype_idx, alpha_idx, k_idx] = generation_time / R0
                    print_results(R0, n_inf_rec, repetition, Dtype, alpha_idx)

        np.savez("dialysis/results/day{}/final_scenario{}".format(day, scenario),
                B_n_inf_rec = B_n_inf_rec,
                B_transmission_route = B_transmission_route,
                B_population = B_population,
                B_R0 = B_R0,
                B_generation_time = B_generation_time,
                H0_n_inf_rec = H0_n_inf_rec,
                H0_transmission_route = H0_transmission_route,
                H0_population = H0_population,
                H0_R0 = H0_R0,
                H0_generation_time = H0_generation_time,
                H1_n_inf_rec = H1_n_inf_rec,
                H1_transmission_route = H1_transmission_route,
                H1_population = H1_population,
                H1_R0 = H1_R0,
                H1_generation_time = H1_generation_time,
                P2_n_inf_rec = P2_n_inf_rec,
                P2_transmission_route = P2_transmission_route,
                P2_population = P2_population,
                P2_R0 = P2_R0,
                P2_generation_time = P2_generation_time,
                H2P2v2_n_inf_rec = H2P2v2_n_inf_rec,
                H2P2v2_transmission_route = H2P2v2_transmission_route,
                H2P2v2_population = H2P2v2_population,
                H2P2v2_R0 = H2P2v2_R0,
                H2P2v2_generation_time = H2P2v2_generation_time,
                H2P2_n_inf_rec = H2P2_n_inf_rec,
                H2P2_transmission_route = H2P2_transmission_route,
                H2P2_population = H2P2_population,
                H2P2_R0 = H2P2_R0,
                H2P2_generation_time = H2P2_generation_time,
                N0_n_inf_rec = N0_n_inf_rec,
                N0_transmission_route = N0_transmission_route,
                N0_population = N0_population,
                N0_R0 = N0_R0,
                N0_generation_time = N0_generation_time,
                N1_n_inf_rec = N1_n_inf_rec,
                N1_transmission_route = N1_transmission_route,
                N1_population = N1_population,
                N1_R0 = N1_R0,
                N1_generation_time = N1_generation_time,
                H3P1_n_inf_rec = H3P1_n_inf_rec,
                H3P1_transmission_route = H3P1_transmission_route,
                H3P1_population = H3P1_population,
                H3P1_R0 = H3P1_R0,
                H3P1_generation_time = H3P1_generation_time,
                Bp_n_inf_rec = Bp_n_inf_rec,
                Bp_transmission_route = Bp_transmission_route,
                Bp_population = Bp_population,
                Bp_R0 = Bp_R0,
                Bp_generation_time = Bp_generation_time,
                Bpp_n_inf_rec = Bpp_n_inf_rec,
                Bpp_transmission_route = Bpp_transmission_route,
                Bpp_population = Bpp_population,
                Bpp_R0 = Bpp_R0,
                Bpp_generation_time = Bpp_generation_time,
                Bppp_n_inf_rec = Bppp_n_inf_rec,
                Bppp_transmission_route = Bppp_transmission_route,
                Bppp_population = Bppp_population,
                Bppp_R0 = Bppp_R0,
                Bppp_generation_time = Bppp_generation_time
                )
            ####################################################################################################
            # Interventon on HCW end
            ####################################################################################################
