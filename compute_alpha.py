"""
Dialysis COVID19 simulation 

Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Aug, 2020

Computes alpha to match the desired R0.
"""

import argparse
import pandas as pd
import numpy as np
import datetime 

# Disease model: note that you need to change exp/exp based on W and T
def get_D(Dtype, W, T):
    daily_shedding = np.zeros((W + T))
    # uni/uni: 5% asymptomatic spread
    if Dtype == 0:
        daily_shedding[:T] = 1
        daily_shedding[T:] = 5*T / (95*W)
    # uni/uni: 35% asymptomatic spread
    elif Dtype == 1:
        daily_shedding[:T] = 1
        daily_shedding[T:] = 35*T / (65*W)
    # exp/exp: 5% asymptomatic spread
    elif Dtype == 2:
        daily_shedding[T-1] = 1
        # Infectivity during incubation period
        for idx in range(T, W + T):
            # daily_shedding[idx] = 1/7.7 * daily_shedding[idx-1]
            daily_shedding[idx] = 1/3.0 * daily_shedding[idx-1]
        # Infectivity during infectious period
        for idx in range(T-2, -1, -1):
            daily_shedding[idx] = 1/1.5 * daily_shedding[idx+1]
    # exp/exp: 35% asymptomatic spread
    elif Dtype == 3:
        daily_shedding[T-1] = 1
        # Infectivity during incubation period
        for idx in range(T, W + T):
            # daily_shedding[idx] = 1/1.48 * daily_shedding[idx-1]
            daily_shedding[idx] = 1/1.27 * daily_shedding[idx-1]
        # Infectivity during infectious period
        for idx in range(T-2, -1, -1):
            daily_shedding[idx] = 1/1.5 * daily_shedding[idx+1]
    return daily_shedding

def compute_R0(alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient):

    p = alpha * daily_shedding
    p_hpc = np.tile(p, (n_mp, n_hcw, 1)).swapaxes(0, 2)
    p_ppc = np.tile(p, (n_mp, n_patient, 1)).swapaxes(0, 2)

    R0_hpc = (1 - np.prod(np.power((1-p_hpc), hpc), axis=0)).sum() / n_mp
    R0_ppc = (1 - np.prod(np.power((1-p_ppc), ppc), axis=0)).sum() / n_mp

    return R0_hpc + R0_ppc

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

    # Dtype = 2
    Dtype_list = range(4)
    R0_list = [2, 2.5, 3]

    alpha_array = np.zeros((len(Dtype_list), len(R0_list)))
    for i, Dtype in enumerate(Dtype_list):
        for j, target_R0 in enumerate(R0_list):
            daily_shedding = np.flip(get_D(Dtype, W, T))
            d = daily_shedding.shape[0]
            hpc = hcw_patient_contact[:d,:,morning_patients,:].sum(axis=3)
            # ppc = patient_patient_contact[:d,:,morning_patients,:].sum(axis=3)
            ppc = patient_patient_contact[:d,:,morning_patients,:].sum(axis=3)

            diff = 1
            low_alpha = 0.00001
            high_alpha = 1
            # low_R0 is almost 0 adn high_R0 is about 12;
            # hence, we can use these low_alpha and high_alpha in binary search to find target R0
            low_R0 = compute_R0(low_alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient)
            high_R0 = compute_R0(high_alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient)
            while True:
                # print("alpha: {}, R0: {}".format(alpha, R0))
                alpha = (low_alpha + high_alpha) / 2
                R0 = compute_R0(alpha, daily_shedding, hpc, ppc, n_mp, n_hcw, n_patient)
                diff = abs(target_R0 - R0)
                if diff < 0.001:
                    print("Dtype: {}, alpha: {}, R0: {}".format(Dtype, alpha, R0))
                    alpha_array[i, j] = alpha
                    break
                elif R0 > target_R0:
                    high_alpha = alpha
                elif R0 < target_R0:
                    low_alpha = alpha

    np.save("dialysis/data/alpha_array", alpha_array)
