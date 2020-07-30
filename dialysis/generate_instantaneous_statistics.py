"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script generates instantaneous network statistics
"""

import argparse
# import igraph
import pandas as pd
import numpy as np
from graph_statistics import *
import networkx as nx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance

    # Load Patient arrays
    npzfile = np.load("contact_data/patient_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hcw_patient_contact = npzfile["hcw_patient_contact_arrays"]
    patient_patient_contact = npzfile["patient_patient_contact_arrays"]
    npzfile.close()

    simulation_period = hcw_patient_contact.shape[0]
    # Load HCW arrays (note that there are 5 array. we're using only hcw_hcw_contact here.)
    npzfile = np.load("contact_data/hcw_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hcw_hcw_contact = npzfile["hcw_hcw_contact"]
    npzfile.close()

    # Make hcw_hcw_contact in the same shape as other contact arrays, then zero out the contacts on Sunday
    hcw_hcw_contact = np.repeat(np.expand_dims(hcw_hcw_contact, axis=0), simulation_period, axis=0)
    hcw_hcw_contact[6,:,:,:] = 0
    hcw_hcw_contact[13,:,:,:] = 0
    hcw_hcw_contact[20,:,:,:] = 0
    hcw_hcw_contact[27,:,:,:] = 0

    n_days = hcw_patient_contact.shape[0]
    n_hcw = hcw_patient_contact.shape[1]
    n_patient = hcw_patient_contact.shape[2]
    n_timesteps = hcw_patient_contact.shape[3]
    n_total = n_hcw + n_patient

    # d = 0
    # t = 1000

    # Prepare a table on estra statistics.
    index_list = ["overall degree", "HCP degree", "patient degree"]
    column_name = ["mean", "std", "max"]
    # there are 4 sundays within 1 month period
    statistic_array = np.zeros((n_days-4, n_timesteps, len(index_list), len(column_name)))

    sunday_cnt = 0
    degree_zero_cnt = 0
    for d in range(n_days):
    # for d in [0]:
        if d in [6, 13, 20, 27]:
            sunday_cnt += 1
            continue
        d_idx = d - sunday_cnt
        for t in range(n_timesteps):
        # for t in [1000]:
            A = np.zeros((n_total, n_total)).astype(int)
            A[:n_hcw, :n_hcw] = (hcw_hcw_contact[d,:,:,t] + hcw_hcw_contact[d,:,:,t].T)
            A[:n_hcw, n_hcw:n_total] = hcw_patient_contact[d,:,:,t]
            A[n_hcw:n_total, :n_hcw] = hcw_patient_contact[d,:,:,t].T
            A[n_hcw:n_total, n_hcw:n_total] = (patient_patient_contact[d,:,:,t] + patient_patient_contact[d,:,:,t].T)

            G = nx.from_numpy_matrix(A, parallel_edges=False)

            # degree
            degree_dict = dict(G.degree)
            degree_array = np.array(list(degree_dict.values()))
            degree_array_geq1 = degree_array[np.nonzero(degree_array)[0]]
            if degree_array_geq1.size == 0:
                statistic_array[d_idx, t, 0] = [np.mean(degree_array), np.std(degree_array), np.max(degree_array)]
                degree_zero_cnt += 1
            else:
                statistic_array[d_idx, t, 0] = [np.mean(degree_array_geq1), np.std(degree_array_geq1), np.max(degree_array_geq1)]

            #degree and weighted degree, just of HCPs
            HCP_degree_array = degree_array[:n_hcw]
            HCP_degree_array_geq1 = HCP_degree_array[np.nonzero(HCP_degree_array)[0]]
            if HCP_degree_array_geq1.size == 0:
                statistic_array[d_idx, t, 1] = [np.mean(HCP_degree_array), np.std(HCP_degree_array), np.max(HCP_degree_array)]
            else:
                statistic_array[d_idx, t, 1] = [np.mean(HCP_degree_array_geq1), np.std(HCP_degree_array_geq1), np.max(HCP_degree_array_geq1)]

            #degree and weighted degree, just of patients
            patient_degree_array = degree_array[n_hcw:]
            patient_degree_array_geq1 = patient_degree_array[np.nonzero(patient_degree_array)[0]]
            if patient_degree_array_geq1.size == 0:
                statistic_array[d_idx, t, 2] = [np.mean(patient_degree_array), np.std(patient_degree_array), np.max(patient_degree_array)]
            else:
                statistic_array[d_idx, t, 2] = [np.mean(patient_degree_array_geq1), np.std(patient_degree_array_geq1), np.max(patient_degree_array_geq1)]

    print("G_dt with no degree: {}".format(degree_zero_cnt))
    df_instantaneous_statistics = pd.DataFrame(data=np.mean(statistic_array, axis=(0,1)), index=index_list, columns=column_name)
    df_instantaneous_statistics.to_csv("tables/statistics/instantaneous_network_statistics.csv") 
