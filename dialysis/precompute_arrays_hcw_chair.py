"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script precomputes hcw-hcw contact networks that are happening at following places:
    1. the same chai
    2. the adjacent chair
    3. central area (nurses station)
    4. other remaining places
    5. anywhere described above
"""

import argparse
import pandas as pd
import numpy as np
from itertools import combinations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance

    npzfile = np.load("contact_data/hcw_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hcw_hcw_contact = npzfile["hcw_hcw_contact"]
    hcw_hcw_contact_both_center = npzfile["hcw_hcw_contact_both_center"]
    npzfile.close()

    n_hcw = hcw_hcw_contact.shape[0]
    n_chair = 9
    max_time = hcw_hcw_contact.shape[-1]

    # hcw_chair_dist = np.load("data/hcw_chair_distance_day{}.npy".format(day))
    hcw_chair_dist = np.zeros((n_hcw, n_chair, max_time))
    for h in range(n_hcw):
        df_chair_dist = pd.read_csv("data/HCP_chair_distance/day{}/HCP{}_chair_distance_day{}.csv".format(day, h+1, day))
        hcw_chair_dist[h,:,:] = df_chair_dist.values.T

    hcw_chair_prox = hcw_chair_dist <= 1

    n_hcw = hcw_chair_prox.shape[0]
    max_time = hcw_chair_prox.shape[-1]

    hcw_hcw_adj_chair = np.zeros((n_hcw, n_hcw, max_time)).astype(bool)
    hcw_hcw_same_chair = np.zeros((n_hcw, n_hcw, max_time)).astype(bool)
    
    # 01 12 23 34 45 67 78
    for c in [0,1,2,3,4, 6,7]:
        hcw_chair_prox_adj = hcw_chair_prox[:,c,:] + hcw_chair_prox[:,c+1,:]
        hcw_array, t_array = np.nonzero(hcw_chair_prox_adj)

        t_array_unique, t_array_count = np.unique(t_array, return_counts=True)
        t_array_geq_2hcws = t_array_unique[t_array_count > 1]

        for t in t_array_geq_2hcws:
            hcw_pairs = combinations(hcw_chair_prox_adj[:,t].nonzero()[0], 2)
            for h1, h2 in hcw_pairs:
                # This includes HCW contacts at the same chair as well
                hcw_hcw_adj_chair[h1, h2, t] = True

    for c in range(9):
        hcw_chair_prox_same = hcw_chair_prox[:,c,:]
        hcw_array, t_array = np.nonzero(hcw_chair_prox_same)

        t_array_unique, t_array_count = np.unique(t_array, return_counts=True)
        t_array_geq_2hcws = t_array_unique[t_array_count > 1]

        for t in t_array_geq_2hcws:
            hcw_pairs = combinations(hcw_chair_prox_same[:,t].nonzero()[0], 2)
            for h1, h2 in hcw_pairs:
                hcw_hcw_same_chair[h1, h2, t] = True

    # This includes HCW contacts at the same chair as well
    hcw_hcw_contact_adj_chair = np.logical_and(hcw_hcw_adj_chair, hcw_hcw_contact)
    hcw_hcw_contact_same_chair = np.logical_and(hcw_hcw_same_chair, hcw_hcw_contact)

    hhc_other_places = np.logical_and(hcw_hcw_contact, np.logical_not(hcw_hcw_contact_adj_chair))
    hhc_adj_chair = np.logical_and(hcw_hcw_contact_adj_chair, np.logical_not(hcw_hcw_contact_same_chair))
    hhc_same_chair = hcw_hcw_contact_same_chair

    # Now filter in hcw_hcw contacts from hcw_hcw_adj_chair events
    np.savez("contact_data/hhc_arrays_day{}_{}ft.npz".format(day, contact_distance),
            hhc_same_chair = hhc_same_chair,
            hhc_adj_chair = hhc_adj_chair,
            hhc_both_center = hcw_hcw_contact_both_center,
            hhc_other_places = np.logical_and(hcw_hcw_contact, np.logical_not(hcw_hcw_contact_adj_chair)),
            hhc_total = hhc_same_chair + hhc_adj_chair + hcw_hcw_contact_both_center + hhc_other_places
            )
