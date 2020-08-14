"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: Aug, 2020

Description: This script generates network statistics
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

    hcw_hcw = hcw_hcw_contact.sum(axis=(0,-1))
    hcw_patient = hcw_patient_contact.sum(axis=(0,-1))
    patient_patient = patient_patient_contact.sum(axis=(0,-1))

    n_hcw = hcw_patient.shape[0]
    n_patient = hcw_patient.shape[1]
    n_patient_MWF = n_patient // 2
    n_total = n_hcw + n_patient

    A = np.zeros((n_total, n_total)).astype(int)
    A[:n_hcw, :n_hcw] = (hcw_hcw + hcw_hcw.T)
    A[:n_hcw, n_hcw:n_total] = hcw_patient
    A[n_hcw:n_total, :n_hcw] = hcw_patient.T
    A[n_hcw:n_total, n_hcw:n_total] = (patient_patient + patient_patient.T)

    G = nx.from_numpy_matrix(A, parallel_edges=False)

    n, m, k_mean, k_max, std, cc, c, assortativity, n_giant, m_giant = generate_graph_statistics(G)

    # Set node attributes
    attrs = {}
    for i in range(n_hcw):
        attrs[i] = {"type": 'HCP'}
    for i in range(n_hcw,n_total):
        attrs[i] = {"type": 'patient'}
    nx.set_node_attributes(G, attrs)

    # Prepare a table on estra statistics.

    index_list = ["overall degree", "overall weighted degree", "HCP degree", "HCP weighted degree", "patient degree", "patient weighted degree", "HCP-HCP edge weight", "HCP-patient edge weight", "patient-patient edge weight"]
    column_name = ["mean", "std", "max"]
    statistic_array = np.zeros((len(index_list), len(column_name)))

    # degree
    degree_dict = dict(G.degree)
    degree_array = np.array(list(degree_dict.values()))
    statistic_array[0] = [np.mean(degree_array), np.std(degree_array), np.max(degree_array)]
    print("Overall")
    print("degree. mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(degree_array), np.std(degree_array), np.max(degree_array)))

    #weighted degree
    weighted_degree_dict = dict(G.degree(weight='weight'))
    weighted_degree_array = np.array(list(weighted_degree_dict.values()))
    statistic_array[1] = [np.mean(weighted_degree_array), np.std(weighted_degree_array), np.max(weighted_degree_array)]
    print("weighted degree. mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(weighted_degree_array), np.std(weighted_degree_array), np.max(weighted_degree_array)))
    print("weighted degree (hrs/day). mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(weighted_degree_array)* 8 / 60 / 60 / 26, np.std(weighted_degree_array)* 8 / 60 / 60 / 26, np.max(weighted_degree_array)* 8 / 60 / 60 / 26))

    #degree and weighted degree, just of HCPs
    print()
    print("HCP")
    HCP_degree_array = degree_array[:n_hcw]
    HCP_weighted_degree_array = weighted_degree_array[:n_hcw]
    statistic_array[2] = [np.mean(HCP_degree_array), np.std(HCP_degree_array), np.max(HCP_degree_array)]
    statistic_array[3] = [np.mean(HCP_weighted_degree_array), np.std(HCP_weighted_degree_array), np.max(HCP_weighted_degree_array)]
    print("HCP_degree. mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_degree_array), np.std(HCP_degree_array), np.max(HCP_degree_array)))
    print("HCP_weighted_degree. mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_weighted_degree_array), np.std(HCP_weighted_degree_array), np.max(HCP_weighted_degree_array)))
    print("HCP_weighted_degree (hrs/day). mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_weighted_degree_array)* 8 / 60 / 60 / 26, np.std(HCP_weighted_degree_array)* 8 / 60 / 60 / 26, np.max(HCP_weighted_degree_array)* 8 / 60 / 60 / 26))

    #degree and weighted degree, just of patients
    print()
    print("patient")
    patient_degree_array = degree_array[n_hcw:]
    patient_weighted_degree_array = weighted_degree_array[n_hcw:]
    statistic_array[4] = [np.mean(patient_degree_array), np.std(patient_degree_array), np.max(patient_degree_array)]
    statistic_array[5] = [np.mean(patient_weighted_degree_array), np.std(patient_weighted_degree_array), np.max(patient_weighted_degree_array)]
    print("patient_degree. mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(patient_degree_array), np.std(patient_degree_array), np.max(patient_degree_array)))
    print("patient_weighted_degree. mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(patient_weighted_degree_array), np.std(patient_weighted_degree_array), np.max(patient_weighted_degree_array)))
    print("patient_weighted_degree (hrs/day). mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(patient_weighted_degree_array)* 8 / 60 / 60 / 26, np.std(patient_weighted_degree_array)* 8 / 60 / 60 / 26, np.max(patient_weighted_degree_array)* 8 / 60 / 60 / 26))

    # preprocess for edge weights
    HCP_HCP_edge_weights = []
    HCP_patient_edge_weights = []
    patient_patient_edge_weights = []

    for edge in G.edges:
        if edge[0] < n_hcw and edge[1] < n_hcw:
            HCP_HCP_edge_weights.append(G.edges[edge]["weight"])
        elif edge[0] >= n_hcw and edge[1] >= n_hcw:
            patient_patient_edge_weights.append(G.edges[edge]["weight"])
        else:
            HCP_patient_edge_weights.append(G.edges[edge]["weight"])

    #mean, max, std dev of weight of HCP-HCP edges
    print()
    print("weight of HCP-HCP edges")
    HCP_HCP_edge_weights = np.array(HCP_HCP_edge_weights)
    statistic_array[6] = [np.mean(HCP_HCP_edge_weights), np.std(HCP_HCP_edge_weights), np.max(HCP_HCP_edge_weights)]
    print("mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_HCP_edge_weights), np.std(HCP_HCP_edge_weights), np.max(HCP_HCP_edge_weights)))
    print("(hrs/day) mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_HCP_edge_weights)* 8 / 60 / 60 / 26, np.std(HCP_HCP_edge_weights)* 8 / 60 / 60 / 26, np.max(HCP_HCP_edge_weights)* 8 / 60 / 60 / 26))

    #mean, max, std dev of weight of HCP-patient edges
    print()
    print("weight of HCP-patient edges")
    HCP_patient_edge_weights = np.array(HCP_patient_edge_weights)
    statistic_array[7] = [np.mean(HCP_patient_edge_weights), np.std(HCP_patient_edge_weights), np.max(HCP_patient_edge_weights)]
    print("mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_patient_edge_weights), np.std(HCP_patient_edge_weights), np.max(HCP_patient_edge_weights)))
    print("(hrs/day) mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(HCP_patient_edge_weights)* 8 / 60 / 60 / 26, np.std(HCP_patient_edge_weights)* 8 / 60 / 60 / 26, np.max(HCP_patient_edge_weights)* 8 / 60 / 60 / 26))

    #mean, max, std dev or weight of patient-patient edges
    print()
    print("weight of patient-patient edges")
    patient_patient_edge_weights = np.array(patient_patient_edge_weights)
    statistic_array[8] = [np.mean(patient_patient_edge_weights), np.std(patient_patient_edge_weights), np.max(patient_patient_edge_weights)]
    print("mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(patient_patient_edge_weights), np.std(patient_patient_edge_weights), np.max(patient_patient_edge_weights)))
    print("(hrs/day) mean, std, max: {:.2f}, {:.2f}, {:.2f}".format(np.mean(patient_patient_edge_weights)* 8 / 60 / 60 / 26, np.std(patient_patient_edge_weights)* 8 / 60 / 60 / 26, np.max(patient_patient_edge_weights)* 8 / 60 / 60 / 26))

    df_additional_statistics = pd.DataFrame(data=statistic_array, index=index_list, columns=column_name)
    df_additional_statistics.to_csv("tables/statistics/additional_network_statistics.csv") 
