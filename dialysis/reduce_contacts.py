"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script precomputes reduced hcw-hcw contacts and patient-patient contacts

These contact arrays are used for following interventions:
    1. Improved social distancing among HCPs
    2. Increased physical separation between dialysis stations
"""

import argparse
import pandas as pd
import numpy as np
from itertools import combinations

def reduce_contacts(contact_array, reduction_rate):
    contact_array_reduced = np.zeros((contact_array.shape)).astype(bool)

    h1_array, h2_array, t_array = np.nonzero(contact_array)
    num_pairs = h1_array.shape[0]
    idx = np.random.choice(np.arange(num_pairs), int(num_pairs*(1-reduction_rate)), replace=False)

    for h1, h2, t in zip(h1_array[idx], h2_array[idx], t_array[idx]):
        contact_array_reduced[h1, h2, t] = True
    return contact_array_reduced

def reduce_patient_contacts(contact_array, reduction_rate):
    contact_array_reduced = np.zeros((contact_array.shape)).astype(bool)
    simulation_period = contact_array.shape[0]

    for d in range(simulation_period):
        p1_array, p2_array, t_array = np.nonzero(contact_array[d,:,:,:])
        num_pairs = p1_array.shape[0]
        idx = np.random.choice(np.arange(num_pairs), int(num_pairs*(1-reduction_rate)), replace=False)

        for p1, p2, t in zip(p1_array[idx], p2_array[idx], t_array[idx]):
            contact_array_reduced[d, p1, p2, t] = True
    return contact_array_reduced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance

    # hhc
    npzfile = np.load("contact_data/hhc_arrays_day{}_{}ft.npz".format(day, contact_distance))
    hhc_same_chair = npzfile["hhc_same_chair"]
    hhc_adj_chair = npzfile["hhc_adj_chair"]
    hhc_both_center = npzfile["hhc_both_center"]
    hhc_other_places = npzfile["hhc_other_places"]
    hhc_total = npzfile["hhc_total"]
    npzfile.close()

    # ppc
    npzfile = np.load("contact_data/patient_arrays_day{}_{}ft.npz".format(day, contact_distance))
    ppc = npzfile["patient_patient_contact_arrays"]
    npzfile.close()

    #########################################################################
    # sd: social distancing: hhc at both_center and other_places get reduced
    # mc: move chairs apart: hhc at adj_chair and ppc get reduced

    # rr: remove rate
    rr25 = 0.25
    hhc_adj_chair_rr25 = reduce_contacts(hhc_adj_chair, rr25)
    hhc_both_center_rr25 = reduce_contacts(hhc_both_center, rr25)
    hhc_other_places_rr25 = reduce_contacts(hhc_other_places, rr25)
    ppc_rr25 = reduce_patient_contacts(ppc, rr25)

    rr50 = 0.50
    hhc_adj_chair_rr50 = reduce_contacts(hhc_adj_chair, rr50)
    hhc_both_center_rr50 = reduce_contacts(hhc_both_center, rr50)
    hhc_other_places_rr50 = reduce_contacts(hhc_other_places, rr50)
    ppc_rr50 = reduce_patient_contacts(ppc, rr50)

    rr75 = 0.75
    hhc_adj_chair_rr75 = reduce_contacts(hhc_adj_chair, rr75)
    hhc_both_center_rr75 = reduce_contacts(hhc_both_center, rr75)
    hhc_other_places_rr75 = reduce_contacts(hhc_other_places, rr75)
    ppc_rr75 = reduce_patient_contacts(ppc, rr75)
    
    np.savez("contact_data/contact_arrays_sd_mc_day{}_{}ft".format(day, contact_distance),
            hhc_adj_chair_rr25 = hhc_adj_chair_rr25,
            hhc_both_center_rr25 = hhc_both_center_rr25,
            hhc_other_places_rr25 = hhc_other_places_rr25,
            ppc_rr25 = ppc_rr25,
            hhc_adj_chair_rr50 = hhc_adj_chair_rr50,
            hhc_both_center_rr50 = hhc_both_center_rr50,
            hhc_other_places_rr50 = hhc_other_places_rr50,
            ppc_rr50 = ppc_rr50,
            hhc_adj_chair_rr75 = hhc_adj_chair_rr75,
            hhc_both_center_rr75 = hhc_both_center_rr75,
            hhc_other_places_rr75 = hhc_other_places_rr75,
            ppc_rr75 = ppc_rr75
            )
