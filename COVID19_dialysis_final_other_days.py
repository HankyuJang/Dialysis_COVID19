"""
Dialysis COVID19 simulation 

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: June, 2020

Run simulations on B,B+,B++,B+++ on other days

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

def simulate(W, T, inf, sus, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hcw_hcw_contact, hcw_patient_contact, patient_patient_contact, morning_patients, morning_hcws, rep):
    np.random.seed(rd.randint(0, 10000000))
    simul = Simulation(W, T, inf, sus, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hcw_hcw_contact, hcw_patient_contact, patient_patient_contact, morning_patients, morning_hcws)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance

    # Read date_time.txt that contains start, end times
    # df starts from 0. Day1 = 0th row. Day10 = 9th row
    df = pd.read_table("dialysis/data/date_time.txt", sep='.', header=None, names = ['year','month','day','hour','minute','second'])
    row = df.iloc[day-1]
    session_start = datetime.datetime(row.year, row.month, row.day, row.hour, row.minute, row.second)
    morning_start = datetime.datetime(row.year, row.month, row.day, 9, 0, 0)
    morning = (morning_start - session_start).seconds // 8
    afternoon = morning + 5 * 60 * 60 // 8

    # Load Patient arrays
    npzfile = np.load("dialysis/contact_data/patient_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hpc_original = npzfile["hcw_patient_contact_arrays"]
    ppc_original = npzfile["patient_patient_contact_arrays"]
    npzfile.close()

    # hhc (original)
    npzfile = np.load("dialysis/contact_data/hhc_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hhc_same_chair = npzfile["hhc_same_chair"]
    hhc_adj_chair = npzfile["hhc_adj_chair"]
    hhc_both_center = npzfile["hhc_both_center"]
    hhc_other_places = npzfile["hhc_other_places"]
    hhc_total = npzfile["hhc_total"]
    npzfile.close()

    # hhc, ppc (reduced)
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
    W = 5
    T = 7
    inf = 1
    QC = 1
    asymp_rate = 0.2
    asymp_shedding = 0.5
    QS = W + 1
    QT = 14
    Dtype = 2
    k=1
    # Current attack rate of Johnson County, Iowa
    # 80 / 151140 ~ 0.0005
    community_attack_rate = 80 / 151140
    mask_efficacy = np.array([0.4, 0.4, 0.93])
    intervention = np.zeros((3, 5)).astype(bool)

    # Parameters
    Dtype_list = [0, 1, 2, 3]
    sus_array = np.load("dialysis/data/alpha_array.npy")
    QC_list0 = [0.5, 0.7]
    QC_list1 = [1]
    rr_list = [0.25, 0.5, 0.75, 1.00]
    k_list_H3P1 = [1, 2, 3, 4, 5]
    k_list_Bpp = [1]

    # Multiprocess
    repetition = 500
    n_cpu = 60

    rep = range(repetition)
    
    # Initialize arrays to save results
    # dim1: [during incubation period, during symptomatic period, outside source (coming in infected)]
    # dim2: [hcw_infected, patient_infected, hcw_recovered, patient_recovered]
    n_inf_rec = np.zeros((repetition, 3, 4, simulation_period)).astype(int)
    # dim1: [h->p, p->h, h->h, p->p]
    transmission_route = np.zeros((repetition, 4, simulation_period)).astype(int)
    population = np.zeros((repetition, simulation_period)).astype(int)
    R0 = np.zeros((repetition)).astype(int)
    generation_time = np.zeros((repetition)).astype(int)

    n_Dtype = sus_array.shape[0]
    n_cases = sus_array.shape[1]
    n_QC0 = len(QC_list0)
    n_QC1 = len(QC_list1)
    n_rr = len(rr_list)
    n_k_H3P1 = len(k_list_H3P1)
    n_k_Bpp = len(k_list_Bpp)

    # Baseline
    B_n_inf_rec = np.zeros((repetition, n_Dtype, n_cases, 3, 4, simulation_period))
    B_transmission_route = np.zeros((repetition, n_Dtype, n_cases, 4, simulation_period))
    B_population = np.zeros((repetition, n_Dtype, n_cases, simulation_period))
    B_R0 = np.zeros((repetition, n_Dtype, n_cases))
    B_generation_time = np.zeros((repetition, n_Dtype, n_cases))

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
    # Run simulations on Dtype 2, 3 and R0 = 2, 2.5, 3 (sus_idx=0, 1, 2)
    for scenario in range(2):
        print("*"*40)
        if scenario == 0:
            print("Infection source: Patient")
            morning_patients = np.array(range(n_patient))
            morning_hcws = np.array([])
        elif scenario == 1:
            print("Infection source: HCW")
            morning_patients = np.array([])
            morning_hcws = np.array(range(n_hcw))

        for Dtype_idx, Dtype in enumerate(Dtype_list):
            if Dtype_idx in [0, 1]:
                continue
            for sus_idx, sus in enumerate(sus_array[Dtype_idx]):

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
                func = partial(simulate, W, T, inf, sus, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hhc, hpc, ppc, morning_patients, morning_hcws)
                r = pool.map(func, rep)
                pool.close()

                for i, result in enumerate(r):
                    n_inf_rec[i] = result[0]
                    transmission_route[i] = result[1]
                    population[i] = result[2]
                    R0[i] = result[3]
                    generation_time[i] = result[4]

                B_n_inf_rec[:, Dtype_idx, sus_idx, :, :, :] = n_inf_rec
                B_transmission_route[:, Dtype_idx, sus_idx, :, :] = transmission_route
                B_population[:, Dtype_idx, sus_idx, :] = population
                B_R0[:, Dtype_idx, sus_idx] = R0
                B_generation_time[:, Dtype_idx, sus_idx] = generation_time / R0
                print("D{} alpha{}".format(Dtype, sus))

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
                func = partial(simulate, W, T, inf, sus, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hhc, hpc, ppc, morning_patients, morning_hcws)
                r = pool.map(func, rep)
                pool.close()

                for i, result in enumerate(r):
                    n_inf_rec[i] = result[0]
                    transmission_route[i] = result[1]
                    population[i] = result[2]
                    R0[i] = result[3]
                    generation_time[i] = result[4]

                Bp_n_inf_rec[:, Dtype_idx, sus_idx, :, :, :] = n_inf_rec
                Bp_transmission_route[:, Dtype_idx, sus_idx, :, :] = transmission_route
                Bp_population[:, Dtype_idx, sus_idx, :] = population
                Bp_R0[:, Dtype_idx, sus_idx] = R0
                Bp_generation_time[:, Dtype_idx, sus_idx] = generation_time / R0
                print("D{} alpha{}".format(Dtype, sus))

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
                    func = partial(simulate, W, T, inf, sus, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hhc, hpc, ppc, morning_patients, morning_hcws)
                    r = pool.map(func, rep)
                    pool.close()

                    for i, result in enumerate(r):
                        n_inf_rec[i] = result[0]
                        transmission_route[i] = result[1]
                        population[i] = result[2]
                        R0[i] = result[3]
                        generation_time[i] = result[4]

                    Bpp_n_inf_rec[:, Dtype_idx, sus_idx, k_idx, :, :, :] = n_inf_rec
                    Bpp_transmission_route[:, Dtype_idx, sus_idx, k_idx, :, :] = transmission_route
                    Bpp_population[:, Dtype_idx, sus_idx, k_idx, :] = population
                    Bpp_R0[:, Dtype_idx, sus_idx, k_idx] = R0
                    Bpp_generation_time[:, Dtype_idx, sus_idx, k_idx] = generation_time / R0
                    print("D{} alpha{}".format(Dtype, sus))

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
                    func = partial(simulate, W, T, inf, sus, QC, asymp_rate, asymp_shedding, QS, QT, Dtype, community_attack_rate, k, mask_efficacy, intervention, hhc, hpc, ppc, morning_patients, morning_hcws)
                    r = pool.map(func, rep)
                    pool.close()

                    for i, result in enumerate(r):
                        n_inf_rec[i] = result[0]
                        transmission_route[i] = result[1]
                        population[i] = result[2]
                        R0[i] = result[3]
                        generation_time[i] = result[4]

                    Bppp_n_inf_rec[:, Dtype_idx, sus_idx, k_idx, :, :, :] = n_inf_rec
                    Bppp_transmission_route[:, Dtype_idx, sus_idx, k_idx, :, :] = transmission_route
                    Bppp_population[:, Dtype_idx, sus_idx, k_idx, :] = population
                    Bppp_R0[:, Dtype_idx, sus_idx, k_idx] = R0
                    Bppp_generation_time[:, Dtype_idx, sus_idx, k_idx] = generation_time / R0
                    print("D{} alpha{}".format(Dtype, sus))

        np.savez("dialysis/results/day{}/B_Bp_Bpp_Bppp_scenario{}".format(day, scenario),
                B_n_inf_rec = B_n_inf_rec,
                B_transmission_route = B_transmission_route,
                B_population = B_population,
                B_R0 = B_R0,
                B_generation_time = B_generation_time,
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
