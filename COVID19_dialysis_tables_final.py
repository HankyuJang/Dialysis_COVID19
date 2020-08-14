"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Aug, 2020

Description: This script generates tables

Tables are saved in `dialysis/tables/`
The table that shows attack rates on B,B+,B++,B+++ that is inserted in the paper is the following two files:
dialysis/tables/day10/n_total_attack_rate_scenario0.csv
dialysis/tables/day10/n_total_attack_rate_scenario1.csv
Note that in these scripts, scenario0 = Scenario 1 and scenario1 = Scenario 2 in the paper
"""
import argparse
import numpy as np
import pandas as pd

def get_days_missed(HCP_replaced, QT_array, n_repeat, simulation_period):
    for n in range(n_repeat):
        for d in range(simulation_period-1,0,-1):
            HCP_replaced[n,d] -= HCP_replaced[n,d-1]
    # print(SQ_HCP_replaced[0])
    days_missed = (HCP_replaced * QT_array).sum() / n_repeat
    return days_missed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw cumulative incidence')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-s', '--scenario', type=int, default=0,
                        help= 'Infection source: scenario 0 = patient, scenario 1 = hcw')
    args = parser.parse_args()

    day = args.day
    s = args.scenario
    n_patients = 40
    n_HCPs = 11

    if s == 0:
        source = "patient"
    elif s == 1:
        source = "hcw"

    ############################################################################################################3
    # Read npzfile
    npzfile = np.load("dialysis/results/day{}/final_scenario{}.npz".format(day, s))
    B_n_inf_rec = npzfile["B_n_inf_rec"]
    B_transmission_route = npzfile["B_transmission_route"]
    B_population = npzfile["B_population"]
    B_R0 = npzfile["B_R0"]
    B_generation_time = npzfile["B_generation_time"]
    H0_n_inf_rec = npzfile["H0_n_inf_rec"]
    H0_transmission_route = npzfile["H0_transmission_route"]
    H0_population = npzfile["H0_population"]
    H0_R0 = npzfile["H0_R0"]
    H0_generation_time = npzfile["H0_generation_time"]
    H1_n_inf_rec = npzfile["H1_n_inf_rec"]
    H1_transmission_route = npzfile["H1_transmission_route"]
    H1_population = npzfile["H1_population"]
    H1_R0 = npzfile["H1_R0"]
    H1_generation_time = npzfile["H1_generation_time"]
    P2_n_inf_rec = npzfile["P2_n_inf_rec"]
    P2_transmission_route = npzfile["P2_transmission_route"]
    P2_population = npzfile["P2_population"]
    P2_R0 = npzfile["P2_R0"]
    P2_generation_time = npzfile["P2_generation_time"]
    H2P2_n_inf_rec = npzfile["H2P2_n_inf_rec"]
    H2P2_transmission_route = npzfile["H2P2_transmission_route"]
    H2P2_population = npzfile["H2P2_population"]
    H2P2_R0 = npzfile["H2P2_R0"]
    H2P2_generation_time = npzfile["H2P2_generation_time"]
    H2P2v2_n_inf_rec = npzfile["H2P2v2_n_inf_rec"]
    H2P2v2_transmission_route = npzfile["H2P2v2_transmission_route"]
    H2P2v2_population = npzfile["H2P2v2_population"]
    H2P2v2_R0 = npzfile["H2P2v2_R0"]
    H2P2v2_generation_time = npzfile["H2P2v2_generation_time"]
    N0_n_inf_rec = npzfile["N0_n_inf_rec"]
    N0_transmission_route = npzfile["N0_transmission_route"]
    N0_population = npzfile["N0_population"]
    N0_R0 = npzfile["N0_R0"]
    N0_generation_time = npzfile["N0_generation_time"]
    N1_n_inf_rec = npzfile["N1_n_inf_rec"]
    N1_transmission_route = npzfile["N1_transmission_route"]
    N1_population = npzfile["N1_population"]
    N1_R0 = npzfile["N1_R0"]
    N1_generation_time = npzfile["N1_generation_time"]
    H3P1_n_inf_rec = npzfile["H3P1_n_inf_rec"]
    H3P1_transmission_route = npzfile["H3P1_transmission_route"]
    H3P1_population = npzfile["H3P1_population"]
    H3P1_R0 = npzfile["H3P1_R0"]
    H3P1_generation_time = npzfile["H3P1_generation_time"]
    Bp_n_inf_rec = npzfile["Bp_n_inf_rec"]
    Bp_transmission_route = npzfile["Bp_transmission_route"]
    Bp_population = npzfile["Bp_population"]
    Bp_R0 = npzfile["Bp_R0"]
    Bp_generation_time = npzfile["Bp_generation_time"]
    Bpp_n_inf_rec = npzfile["Bpp_n_inf_rec"]
    Bpp_transmission_route = npzfile["Bpp_transmission_route"]
    Bpp_population = npzfile["Bpp_population"]
    Bpp_R0 = npzfile["Bpp_R0"]
    Bpp_generation_time = npzfile["Bpp_generation_time"]
    Bppp_n_inf_rec = npzfile["Bppp_n_inf_rec"]
    Bppp_transmission_route = npzfile["Bppp_transmission_route"]
    Bppp_population = npzfile["Bppp_population"]
    Bppp_R0 = npzfile["Bppp_R0"]
    Bppp_generation_time = npzfile["Bppp_generation_time"]
    npzfile.close()

    npzfile = np.load("dialysis/results/day{}/baseline_scenario{}.npz".format(day, s))
    B_n_inf_rec = npzfile["B_n_inf_rec"]
    B_transmission_route = npzfile["B_transmission_route"]
    B_population = npzfile["B_population"]
    B_R0 = npzfile["B_R0"]
    B_generation_time = npzfile["B_generation_time"]
    npzfile.close()

    n_repeat = B_R0.shape[0]
    simulation_period = B_R0.shape[-1]

    ############################################################################################################3
    # Preprocess some of the result tables
    ############################################################################################################3
    B_H_infection = B_n_inf_rec[:,:,:,0,0,:] + B_n_inf_rec[:,:,:,1,0,:] + B_n_inf_rec[:,:,:,2,0,:] 
    B_P_infection = B_n_inf_rec[:,:,:,0,1,:] + B_n_inf_rec[:,:,:,1,1,:] + B_n_inf_rec[:,:,:,2,1,:] 
    B_T_infection = B_H_infection + B_P_infection
    B_T_cum_infection = np.cumsum(B_T_infection, axis=-1)
    B_T_cum_attack_rate = B_T_cum_infection / B_population

    H0_H_infection = H0_n_inf_rec[:,:,:,:,0,0,:] + H0_n_inf_rec[:,:,:,:,1,0,:] + H0_n_inf_rec[:,:,:,:,2,0,:] 
    H0_P_infection = H0_n_inf_rec[:,:,:,:,0,1,:] + H0_n_inf_rec[:,:,:,:,1,1,:] + H0_n_inf_rec[:,:,:,:,2,1,:] 
    H0_T_infection = H0_H_infection + H0_P_infection
    H0_T_cum_infection = np.cumsum(H0_T_infection, axis=-1)
    H0_T_cum_attack_rate = H0_T_cum_infection / H0_population

    H1_H_infection = H1_n_inf_rec[:,:,:,:,0,0,:] + H1_n_inf_rec[:,:,:,:,1,0,:] + H1_n_inf_rec[:,:,:,:,2,0,:] 
    H1_P_infection = H1_n_inf_rec[:,:,:,:,0,1,:] + H1_n_inf_rec[:,:,:,:,1,1,:] + H1_n_inf_rec[:,:,:,:,2,1,:] 
    H1_T_infection = H1_H_infection + H1_P_infection
    H1_T_cum_infection = np.cumsum(H1_T_infection, axis=-1)
    H1_T_cum_attack_rate = H1_T_cum_infection / H1_population

    P2_H_infection = P2_n_inf_rec[:,:,:,0,0,:] + P2_n_inf_rec[:,:,:,1,0,:] + P2_n_inf_rec[:,:,:,2,0,:] 
    P2_P_infection = P2_n_inf_rec[:,:,:,0,1,:] + P2_n_inf_rec[:,:,:,1,1,:] + P2_n_inf_rec[:,:,:,2,1,:] 
    P2_T_infection = P2_H_infection + P2_P_infection
    P2_T_cum_infection = np.cumsum(P2_T_infection, axis=-1)
    P2_T_cum_attack_rate = P2_T_cum_infection / P2_population

    H2P2_H_infection = H2P2_n_inf_rec[:,:,:,0,0,:] + H2P2_n_inf_rec[:,:,:,1,0,:] + H2P2_n_inf_rec[:,:,:,2,0,:] 
    H2P2_P_infection = H2P2_n_inf_rec[:,:,:,0,1,:] + H2P2_n_inf_rec[:,:,:,1,1,:] + H2P2_n_inf_rec[:,:,:,2,1,:] 
    H2P2_T_infection = H2P2_H_infection + H2P2_P_infection
    H2P2_T_cum_infection = np.cumsum(H2P2_T_infection, axis=-1)
    H2P2_T_cum_attack_rate = H2P2_T_cum_infection / H2P2_population

    H2P2v2_H_infection = H2P2v2_n_inf_rec[:,:,:,0,0,:] + H2P2v2_n_inf_rec[:,:,:,1,0,:] + H2P2v2_n_inf_rec[:,:,:,2,0,:] 
    H2P2v2_P_infection = H2P2v2_n_inf_rec[:,:,:,0,1,:] + H2P2v2_n_inf_rec[:,:,:,1,1,:] + H2P2v2_n_inf_rec[:,:,:,2,1,:] 
    H2P2v2_T_infection = H2P2v2_H_infection + H2P2v2_P_infection
    H2P2v2_T_cum_infection = np.cumsum(H2P2v2_T_infection, axis=-1)
    H2P2v2_T_cum_attack_rate = H2P2v2_T_cum_infection / H2P2v2_population

    N0_H_infection = N0_n_inf_rec[:,:,:,:,0,0,:] + N0_n_inf_rec[:,:,:,:,1,0,:] + N0_n_inf_rec[:,:,:,:,2,0,:] 
    N0_P_infection = N0_n_inf_rec[:,:,:,:,0,1,:] + N0_n_inf_rec[:,:,:,:,1,1,:] + N0_n_inf_rec[:,:,:,:,2,1,:] 
    N0_T_infection = N0_H_infection + N0_P_infection
    N0_T_cum_infection = np.cumsum(N0_T_infection, axis=-1)
    N0_T_cum_attack_rate = N0_T_cum_infection / N0_population

    N1_H_infection = N1_n_inf_rec[:,:,:,:,0,0,:] + N1_n_inf_rec[:,:,:,:,1,0,:] + N1_n_inf_rec[:,:,:,:,2,0,:] 
    N1_P_infection = N1_n_inf_rec[:,:,:,:,0,1,:] + N1_n_inf_rec[:,:,:,:,1,1,:] + N1_n_inf_rec[:,:,:,:,2,1,:] 
    N1_T_infection = N1_H_infection + N1_P_infection
    N1_T_cum_infection = np.cumsum(N1_T_infection, axis=-1)
    N1_T_cum_attack_rate = N1_T_cum_infection / N1_population
    
    H3P1_H_infection = H3P1_n_inf_rec[:,:,:,:,0,0,:] + H3P1_n_inf_rec[:,:,:,:,1,0,:] + H3P1_n_inf_rec[:,:,:,:,2,0,:] 
    H3P1_P_infection = H3P1_n_inf_rec[:,:,:,:,0,1,:] + H3P1_n_inf_rec[:,:,:,:,1,1,:] + H3P1_n_inf_rec[:,:,:,:,2,1,:] 
    H3P1_T_infection = H3P1_H_infection + H3P1_P_infection
    H3P1_T_cum_infection = np.cumsum(H3P1_T_infection, axis=-1)
    H3P1_T_cum_attack_rate = H3P1_T_cum_infection / H3P1_population

    Bp_H_infection = Bp_n_inf_rec[:,:,:,0,0,:] + Bp_n_inf_rec[:,:,:,1,0,:] + Bp_n_inf_rec[:,:,:,2,0,:] 
    Bp_P_infection = Bp_n_inf_rec[:,:,:,0,1,:] + Bp_n_inf_rec[:,:,:,1,1,:] + Bp_n_inf_rec[:,:,:,2,1,:] 
    Bp_T_infection = Bp_H_infection + Bp_P_infection
    Bp_T_cum_infection = np.cumsum(Bp_T_infection, axis=-1)
    Bp_T_cum_attack_rate = Bp_T_cum_infection / Bp_population

    Bpp_H_infection = Bpp_n_inf_rec[:,:,:,:,0,0,:] + Bpp_n_inf_rec[:,:,:,:,1,0,:] + Bpp_n_inf_rec[:,:,:,:,2,0,:] 
    Bpp_P_infection = Bpp_n_inf_rec[:,:,:,:,0,1,:] + Bpp_n_inf_rec[:,:,:,:,1,1,:] + Bpp_n_inf_rec[:,:,:,:,2,1,:] 
    Bpp_T_infection = Bpp_H_infection + Bpp_P_infection
    Bpp_T_cum_infection = np.cumsum(Bpp_T_infection, axis=-1)
    Bpp_T_cum_attack_rate = Bpp_T_cum_infection / Bpp_population

    Bppp_H_infection = Bppp_n_inf_rec[:,:,:,:,0,0,:] + Bppp_n_inf_rec[:,:,:,:,1,0,:] + Bppp_n_inf_rec[:,:,:,:,2,0,:] 
    Bppp_P_infection = Bppp_n_inf_rec[:,:,:,:,0,1,:] + Bppp_n_inf_rec[:,:,:,:,1,1,:] + Bppp_n_inf_rec[:,:,:,:,2,1,:] 
    Bppp_T_infection = Bppp_H_infection + Bppp_P_infection
    Bppp_T_cum_infection = np.cumsum(Bppp_T_infection, axis=-1)
    Bppp_T_cum_attack_rate = Bppp_T_cum_infection / Bppp_population

    H0_replaced_HCPs = np.mean(H0_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    H1_replaced_HCPs = np.mean(H1_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    H3P1_replaced_HCPs = np.mean(H3P1_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bpp_replaced_HCPs = np.mean(Bpp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)
    Bppp_replaced_HCPs = np.mean(Bppp_population[:,:,:,:,-1], axis=0) - (n_HCPs + n_patients)

    ############################################################################################################3
    # Shape of arrays (B*)
    # dim0: n_repeat
    # dim1: n_Dtype
    # dim2: n_cases
    ############################################################################################################3
    # Shape of arrays (H*)
    # dim0: n_repeat
    # dim1: n_Dtype
    # dim2: n_cases
    # dim3: QC_list
    ############################################################################################################3
    # >>> B_n_inf_rec.shape
    # (500, 4, 3, 3, 4, 30)
    # dim-3: [during incubation period, during symptomatic period, outside source (coming in infected)]
    # dim-2: [hcw_infected, patient_infected, hcw_recovered, patient_recovered]
    # dim-1: simulation period
    ############################################################################################################3
    # >>> B_transmission_route.shape
    # (500, 4, 3, 4, 30)
    # dim-2: [h->p, p->h, h->h, p->p]
    # dim-1: simulation period
    ############################################################################################################3
    # >>> B_population.shape
    # (500, 4, 3, 30)
    # dim-1: simulation period
    ############################################################################################################3
    # >>> B_generation_time.shape
    # (500, 4, 3)
    ############################################################################################################3
    # >>> B_R0.shape
    # (500, 4, 3)
    ############################################################################################################3

    ############################################################
    # n_inf_rec
    ############################################################
    mean_df_n_hcw_inf = pd.DataFrame({
        "B": np.sum(np.mean(B_H_infection, axis=0), axis=-1).flatten(),
        "Bp": np.sum(np.mean(Bp_H_infection, axis=0), axis=-1).flatten(),
        "Bpp:k1": np.sum(np.mean(Bpp_H_infection, axis=0)[:,:,0], axis=-1).flatten(),
        "Bppp:k1": np.sum(np.mean(Bppp_H_infection, axis=0)[:,:,0], axis=-1).flatten(),
        })
    mean_df_n_hcw_inf.to_csv("dialysis/tables/day{}/n_hcw_inf_mean_scenario{}.csv".format(day, s), index=False)


    mean_df_n_patient_inf = pd.DataFrame({
        "B": np.sum(np.mean(B_P_infection, axis=0), axis=-1).flatten(),
        "Bp": np.sum(np.mean(Bp_P_infection, axis=0), axis=-1).flatten(),
        "Bpp:k1": np.sum(np.mean(Bpp_P_infection, axis=0)[:,:,0], axis=-1).flatten(),
        "Bppp:k1": np.sum(np.mean(Bppp_P_infection, axis=0)[:,:,0], axis=-1).flatten(),
        })
    mean_df_n_patient_inf.to_csv("dialysis/tables/day{}/n_patient_inf_mean_scenario{}.csv".format(day, s), index=False)

    ############################################################
    # Percentage of simulations with no additional infection
    ############################################################
    df_no_infection = pd.DataFrame({
        "B": (((B_T_infection).sum(axis=-1) <= 1).sum(axis=0) / n_repeat).flatten(),
        "Bp": (((Bp_T_infection).sum(axis=-1) <= 1).sum(axis=0) / n_repeat).flatten(),
        "Bpp:k1": (((Bpp_T_infection[:,:,:,0,:]).sum(axis=-1) <= 1).sum(axis=0) / n_repeat).flatten(),
        "Bppp:k1": (((Bppp_T_infection[:,:,:,0,:]).sum(axis=-1) <= 1).sum(axis=0) / n_repeat).flatten(),
        })
    df_no_infection.to_csv("dialysis/tables/day{}/no_infection_scenario{}.csv".format(day, s), index=False)

    ############################################################
    # Percentage of simulations with at most 1 additional infection
    ############################################################
    df_one_infection = pd.DataFrame({
        "B": (((B_T_infection).sum(axis=-1) <= 2).sum(axis=0) / n_repeat).flatten(),
        "Bp": (((Bp_T_infection).sum(axis=-1) <= 2).sum(axis=0) / n_repeat).flatten(),
        "Bpp:k1": (((Bpp_T_infection[:,:,:,0,:]).sum(axis=-1) <= 2).sum(axis=0) / n_repeat).flatten(),
        "Bppp:k1": (((Bppp_T_infection[:,:,:,0,:]).sum(axis=-1) <= 2).sum(axis=0) / n_repeat).flatten(),
        })
    df_one_infection.to_csv("dialysis/tables/day{}/one_infection_scenario{}.csv".format(day, s), index=False)

    ############################################################
    # HCPs replaced
    ############################################################
    mean_df_population = pd.DataFrame({
        "B": np.mean(B_population[:,:,:,-1], axis=0).flatten(),
        "Bp": np.mean(Bp_population[:,:,:,-1], axis=0).flatten(),
        "Bpp:k1": np.mean(Bpp_population[:,:,:,:,-1], axis=0)[:,:,0].flatten(),
        "Bppp:k1": np.mean(Bppp_population[:,:,:,:,-1], axis=0)[:,:,0].flatten(),
        })
    mean_df_replaced_hcw = mean_df_population - (n_patients + n_HCPs)
    mean_df_replaced_hcw.to_csv("dialysis/tables/day{}/replaced_hcw_mean_scenario{}.csv".format(day, s), index=False)

    #########################################################
    # total infection
    #########################################################
    mean_df_n_total_inf = mean_df_n_hcw_inf + mean_df_n_patient_inf
    mean_df_n_total_inf.to_csv("dialysis/tables/day{}/n_total_inf_mean_scenario{}.csv".format(day, s), index=False)

    #########################################################
    # total attack rate
    #########################################################
    mean_df_total_attack_rate = mean_df_n_total_inf / mean_df_population
    mean_df_total_attack_rate.to_csv("dialysis/tables/day{}/n_total_attack_rate_scenario{}.csv".format(day, s), index=False)

    #########################################################
    # a diminishing returns on the reduction in the number of infections. 
    # But, there is linear increase in the cost (e.g., number of missed days). 
    #########################################################
    # H3P1
    H3P1_mean_df_n_patient_inf = pd.DataFrame({
        "H3P1:k1": np.sum(np.mean(H3P1_P_infection, axis=0)[:,:,0], axis=-1).flatten(),
        "H3P1:k2": np.sum(np.mean(H3P1_P_infection, axis=0)[:,:,1], axis=-1).flatten(),
        "H3P1:k3": np.sum(np.mean(H3P1_P_infection, axis=0)[:,:,2], axis=-1).flatten(),
        "H3P1:k4": np.sum(np.mean(H3P1_P_infection, axis=0)[:,:,3], axis=-1).flatten(),
        "H3P1:k5": np.sum(np.mean(H3P1_P_infection, axis=0)[:,:,4], axis=-1).flatten(),
        })
    H3P1_mean_df_n_patient_inf.to_csv("dialysis/tables/day{}/H3P1_n_patient_inf_mean_scenario{}.csv".format(day, s), index=False)

    H3P1_mean_df_n_hcw_inf = pd.DataFrame({
        "H3P1:k1": np.sum(np.mean(H3P1_H_infection, axis=0)[:,:,0], axis=-1).flatten(),
        "H3P1:k2": np.sum(np.mean(H3P1_H_infection, axis=0)[:,:,1], axis=-1).flatten(),
        "H3P1:k3": np.sum(np.mean(H3P1_H_infection, axis=0)[:,:,2], axis=-1).flatten(),
        "H3P1:k4": np.sum(np.mean(H3P1_H_infection, axis=0)[:,:,3], axis=-1).flatten(),
        "H3P1:k5": np.sum(np.mean(H3P1_H_infection, axis=0)[:,:,4], axis=-1).flatten(),
        })
    H3P1_mean_df_n_hcw_inf.to_csv("dialysis/tables/day{}/H3P1_n_hcw_inf_mean_scenario{}.csv".format(day, s), index=False)

    H3P1_mean_df_n_total_inf = H3P1_mean_df_n_hcw_inf + H3P1_mean_df_n_patient_inf
    H3P1_mean_df_n_total_inf.to_csv("dialysis/tables/day{}/H3P1_n_total_inf_mean_scenario{}.csv".format(day, s), index=False)

    H3P1_mean_df_population = pd.DataFrame({
        "H3P1:k1": np.mean(H3P1_population[:,:,:,:,-1], axis=0)[:,:,0].flatten(),
        "H3P1:k2": np.mean(H3P1_population[:,:,:,:,-1], axis=0)[:,:,1].flatten(),
        "H3P1:k3": np.mean(H3P1_population[:,:,:,:,-1], axis=0)[:,:,2].flatten(),
        "H3P1:k4": np.mean(H3P1_population[:,:,:,:,-1], axis=0)[:,:,3].flatten(),
        "H3P1:k5": np.mean(H3P1_population[:,:,:,:,-1], axis=0)[:,:,4].flatten(),
        })
    H3P1_mean_df_replaced_hcw = H3P1_mean_df_population - (n_patients + n_HCPs)
    H3P1_mean_df_replaced_hcw.to_csv("dialysis/tables/day{}/H3P1_replaced_hcw_mean_scenario{}.csv".format(day, s), index=False)

    H3P1_mean_df_total_attack_rate = H3P1_mean_df_n_total_inf / H3P1_mean_df_population
    H3P1_mean_df_total_attack_rate.to_csv("dialysis/tables/day{}/H3P1_n_total_attack_rate_scenario{}.csv".format(day, s), index=False)

    # # Bpp
    # Bpp_mean_df_n_patient_inf = pd.DataFrame({
        # "Bpp:k1": np.sum(np.mean(Bpp_P_infection, axis=0)[:,:,0], axis=-1).flatten(),
        # "Bpp:k2": np.sum(np.mean(Bpp_P_infection, axis=0)[:,:,1], axis=-1).flatten(),
        # "Bpp:k3": np.sum(np.mean(Bpp_P_infection, axis=0)[:,:,2], axis=-1).flatten(),
        # "Bpp:k4": np.sum(np.mean(Bpp_P_infection, axis=0)[:,:,3], axis=-1).flatten(),
        # "Bpp:k5": np.sum(np.mean(Bpp_P_infection, axis=0)[:,:,4], axis=-1).flatten(),
        # })
    # Bpp_mean_df_n_patient_inf.to_csv("dialysis/tables/day{}/Bpp_n_patient_inf_mean_scenario{}.csv".format(day, s), index=False)

    # Bpp_mean_df_n_hcw_inf = pd.DataFrame({
        # "Bpp:k1": np.sum(np.mean(Bpp_H_infection, axis=0)[:,:,0], axis=-1).flatten(),
        # "Bpp:k2": np.sum(np.mean(Bpp_H_infection, axis=0)[:,:,1], axis=-1).flatten(),
        # "Bpp:k3": np.sum(np.mean(Bpp_H_infection, axis=0)[:,:,2], axis=-1).flatten(),
        # "Bpp:k4": np.sum(np.mean(Bpp_H_infection, axis=0)[:,:,3], axis=-1).flatten(),
        # "Bpp:k5": np.sum(np.mean(Bpp_H_infection, axis=0)[:,:,4], axis=-1).flatten(),
        # })
    # Bpp_mean_df_n_hcw_inf.to_csv("dialysis/tables/day{}/Bpp_n_hcw_inf_mean_scenario{}.csv".format(day, s), index=False)

    # Bpp_mean_df_n_total_inf = Bpp_mean_df_n_hcw_inf + Bpp_mean_df_n_patient_inf
    # Bpp_mean_df_n_total_inf.to_csv("dialysis/tables/day{}/Bpp_n_total_inf_mean_scenario{}.csv".format(day, s), index=False)

    # Bpp_mean_df_population = pd.DataFrame({
        # "Bpp:k1": np.mean(Bpp_population[:,:,:,:,-1], axis=0)[:,:,0].flatten(),
        # "Bpp:k2": np.mean(Bpp_population[:,:,:,:,-1], axis=0)[:,:,1].flatten(),
        # "Bpp:k3": np.mean(Bpp_population[:,:,:,:,-1], axis=0)[:,:,2].flatten(),
        # "Bpp:k4": np.mean(Bpp_population[:,:,:,:,-1], axis=0)[:,:,3].flatten(),
        # "Bpp:k5": np.mean(Bpp_population[:,:,:,:,-1], axis=0)[:,:,4].flatten(),
        # })
    # Bpp_mean_df_replaced_hcw = Bpp_mean_df_population - (n_patients + n_HCPs)
    # Bpp_mean_df_replaced_hcw.to_csv("dialysis/tables/day{}/Bpp_replaced_hcw_mean_scenario{}.csv".format(day, s), index=False)

    # Bpp_mean_df_total_attack_rate = Bpp_mean_df_n_total_inf / Bpp_mean_df_population
    # Bpp_mean_df_total_attack_rate.to_csv("dialysis/tables/day{}/Bpp_n_total_attack_rate_scenario{}.csv".format(day, s), index=False)


