"""
Dialysis COVID19 simulation 

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Aug, 2020

Computes alpha to match the desired R0.
Furthermore, it guarantees the total shedding between two shedding models to be same.
"""

import argparse
import pandas as pd
import numpy as np
import datetime 

def get_daily_shedding(W, T, beta, gamma):
    daily_shedding = np.zeros((W + T))
    daily_shedding[T-1] = 1
    # Infectivity during incubation period
    for idx in range(T, W + T):
        daily_shedding[idx] = 1/beta * daily_shedding[idx-1]
    # Infectivity during infectious period
    for idx in range(T-2, -1, -1):
        daily_shedding[idx] = 1/gamma * daily_shedding[idx+1]
    return daily_shedding

def compute_R0(alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient):

    p = alpha * daily_shedding
    p_hpc = np.tile(p, (n_mp, n_hcw, 1)).swapaxes(0, 2)
    p_ppc = np.tile(p, (n_mp, n_patient, 1)).swapaxes(0, 2)

    R0_hpc = (1 - np.prod(np.power((1-p_hpc), hpc), axis=0)).sum() / n_mp
    R0_ppc = (1 - np.prod(np.power((1-p_ppc), ppc), axis=0)).sum() / n_mp

    return R0_hpc + R0_ppc

# This functions finds alpha for the target R0 by binary search.
def get_alpha(target_R0, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient):
    diff = 1
    low_alpha = 0.00001
    high_alpha = 1
    # low_R0 is almost 0 and high_R0 is about 12;
    # hence, we can use these low_alpha and high_alpha in binary search to find target R0
    low_R0 = compute_R0(low_alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient)
    high_R0 = compute_R0(high_alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient)
    while True:
        # print("alpha: {}, R0: {}".format(alpha, R0))
        alpha = (low_alpha + high_alpha) / 2.0
        R0 = compute_R0(alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient)
        diff = abs(target_R0 - R0)
        if diff < 0.00001:
            # print("alpha: {:.4f}, R0: {:.1f}".format(alpha, R0))
            return alpha
            # alpha_array[i, j] = alpha
            # break
        elif R0 > target_R0:
            high_alpha = alpha
        elif R0 < target_R0:
            low_alpha = alpha

def get_volume(P_shedding_rate, alpha, gamma, T):
    # symptomatic_shedding_total before multiplying alpha
    S_shedding_total = sum([1/pow(gamma,s) for s in range(T)])
    return (1 + P_shedding_rate/(1-P_shedding_rate)) * alpha * S_shedding_total

def get_P_volume(alpha, beta, W):
    P_shedding_total = sum([1/pow(beta,s) for s in range(1,W+1)])
    return alpha * P_shedding_total

def update_gamma(current_V, target_V, P_shedding_rate, alpha, T):
    low_gamma = 1
    high_gamma = 10
    high_V = get_volume(P_shedding_rate, alpha, low_gamma, T)
    low_V = get_volume(P_shedding_rate, alpha, high_gamma, T)

    while True:
        new_gamma = (low_gamma + high_gamma) / 2.0
        new_V = get_volume(P_shedding_rate, alpha, new_gamma, T)
        diff = abs(target_V - new_V)
        # print("gamma: {:.4f}, current V: {:.4f}, target V: {:.4f}".format(new_gamma, new_V, target_V))
        if diff < 0.00001:
            return new_gamma
        elif new_V > target_V:
            low_gamma = new_gamma
        elif new_V < target_V:
            high_gamma = new_gamma

def get_beta(V, P_shedding_rate, alpha, W):
    P_V = V * P_shedding_rate
    low_beta = 1
    high_beta = 10
    while True:
        new_beta = (low_beta + high_beta) / 2.0
        new_P_V = get_P_volume(alpha, new_beta, W)
        diff = abs(P_V - new_P_V)
        # print("beta: {:.4f}, current P_V: {:.4f}, target P_V: {:.4f}".format(new_beta, new_P_V, P_V))
        if diff < 0.00001:
            return new_beta
        elif new_P_V > P_V:
            low_beta = new_beta
        elif new_P_V < P_V:
            high_beta = new_beta
    

# morning patients: patients where the dialysis sessin starts before 9 am.
# Note: all the morning sessions in our data starts before 9 am.
# dim0: simulation days, dim1: number of hcws, dim2: number of patients, dim3: total timesteps in a day
# >>> hpc_original.shape  # for day 10
# (30, 11, 40, 6822)
def get_morning_patients(hpc_original, timestep_at_nine):
    return hpc_original[0, :, :, :timestep_at_nine].sum(axis=(0,2)).nonzero()[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance

    np.set_printoptions(precision=6)
    np.set_printoptions(suppress=True)

    # Read date_time.txt that contains start, end times
    # df starts from 0. Day1 = 0th row. Day10 = 9th row
    df = pd.read_table("dialysis/data/date_time.txt", sep='.', header=None, names = ['year','month','day','hour','minute','second'])
    row = df.iloc[day-1]

    start_time_of_day = datetime.datetime(row.year, row.month, row.day, row.hour, row.minute, row.second)

    nine = datetime.datetime(row.year, row.month, row.day, 9, 0, 0)
    timestep_at_nine = (nine - start_time_of_day).seconds // 8

    # presymptomatic, symptomatic period
    W = 6
    T = 7
    d = W + T

    # Load Patient arrays
    npzfile = np.load("dialysis/contact_data/patient_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hcw_patient_contact = npzfile["hcw_patient_contact_arrays"]
    patient_patient_contact = npzfile["patient_patient_contact_arrays"]
    npzfile.close()

    # Days in simulation
    simulation_period = hcw_patient_contact.shape[0]
    n_hcw = hcw_patient_contact.shape[1]
    n_patient = hcw_patient_contact.shape[2]
    max_time = hcw_patient_contact.shape[3]

    # Fill lower triangle of patient_patient_contact
    for day in range(simulation_period):
        for t in range(max_time):
            patient_patient_contact[day,:,:,t] += patient_patient_contact[day,:,:,t].T

    # get morning patients on the first day
    morning_patients = get_morning_patients(hcw_patient_contact, timestep_at_nine)
    n_mp = morning_patients.shape[0]

    hpc = hcw_patient_contact[:d,:,morning_patients,:].sum(axis=3)
    ppc = patient_patient_contact[:d,:,morning_patients,:].sum(axis=3)

    target_R0_list = [2.0, 2.5, 3.0]
    alpha_array = np.zeros((4, len(target_R0_list)))
    beta_array = np.zeros((4, len(target_R0_list)))
    gamma_array = np.zeros((4, len(target_R0_list)))

    for target_R0_idx, target_R0 in enumerate(target_R0_list):

        #################################
        # Step1: get V20. (Volumn on exp/exp (20%) model
        #################################
        # These beta and gamma gives us shedding 20% shedding in P.
        beta_exp20 = 3.01 # ramp_up
        gamma_exp20 = 2.0 # ramp_down
        daily_shedding_exp20 = np.flip(get_daily_shedding(W, T, beta_exp20, gamma_exp20))
        alpha_exp20 = get_alpha(target_R0, daily_shedding_exp20, hpc, ppc, n_mp, n_hcw, n_patient)
        V_exp20 = (daily_shedding_exp20 * alpha_exp20).sum()
        # V_20 = get_volume(0.2, alpha_exp20, gamma_exp20, T)
        # print("exp/exp (20%). V:{:.4f}".format(V_exp20))

        #################################
        # Step2: get V60. (Volumn on exp/exp (60%) model
        # Goal: Keep adjusting gamma, beta, alpha to get the two shedding models to have same volume
        #################################
        # These beta and gamma gives us shedding 60% shedding in P.
        beta_exp60 = 1.246 # ramp_up
        gamma_exp60 = 2.0 # ramp_down
        daily_shedding_exp60 = np.flip(get_daily_shedding(W, T, beta_exp60, gamma_exp60))
        alpha_exp60 = get_alpha(target_R0, daily_shedding_exp60, hpc, ppc, n_mp, n_hcw, n_patient)
        V_exp60 = (daily_shedding_exp60 * alpha_exp60).sum()
        # V_60 = get_volume(0.6, alpha_exp60, gamma_exp60, T)
        # print("exp/exp (60%). V:{:.4f}".format(V_exp60))
        # print("get_volume_exp60: {:.4f}".format(V_60))
        # print("get_volume_exp20: {:.4f}".format(V_20))
        while True:
            # print("V_exp60: {:.4f}".format(V_exp60))
            # if abs(V_exp20 - V_exp60) < 0.0001:
            if V_exp20 < V_exp60:
                break
            # break
            # gamma_exp60 = update_gamma(V_exp60, V_exp20, 0.6, alpha_exp60, T)
            gamma_exp60 += 0.05
            V_exp60 = get_volume(0.6, alpha_exp60, gamma_exp60, T)
            beta_exp60 = get_beta(V_exp60, 0.6, alpha_exp60, W)
            daily_shedding_exp60 = np.flip(get_daily_shedding(W, T, beta_exp60, gamma_exp60))
            alpha_exp60 = get_alpha(target_R0, daily_shedding_exp60, hpc, ppc, n_mp, n_hcw, n_patient)
            V_exp60 = (daily_shedding_exp60 * alpha_exp60).sum()

        alpha_array[2:,target_R0_idx] = alpha_exp20, alpha_exp60
        beta_array[2:,target_R0_idx] = beta_exp20, beta_exp60
        gamma_array[2:,target_R0_idx] = gamma_exp20, gamma_exp60  
        print("Target R0: {:.1f}".format(target_R0))
        print("Volume: exp/exp(20%): {:.4f}, exp/exp(60%): {:.4f}".format(V_exp20, V_exp60))

    index_list = ["not_used", "not_used", "exp/exp(20%)", "exp/exp(60%)"]
    df_alpha = pd.DataFrame(data=alpha_array, columns=target_R0_list, index=index_list)
    df_beta = pd.DataFrame(data=beta_array, columns=target_R0_list, index=index_list)
    df_gamma = pd.DataFrame(data=gamma_array, columns=target_R0_list, index=index_list)

    df_alpha.to_csv("dialysis/data/df_alpha.csv", index=True)
    df_beta.to_csv("dialysis/data/df_beta.csv", index=True)
    df_gamma.to_csv("dialysis/data/df_gamma.csv", index=True)
