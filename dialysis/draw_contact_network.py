"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: This script draws the contact network where the edges are aggregated over the timespan of 30 days
"""

import argparse
import igraph
import pandas as pd
import numpy as np
import random

# import matplotlib as mpl
# mpl.use('Agg')
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dialysis Unit')
    parser.add_argument('-day', '--day', type=int, default=10,
                        help= 'day of csv file that contains the latent positions of hcws')
    parser.add_argument('-d', '--contact_distance', type=int, default=6,
                        help= 'distance threshold (in feet)')
    args = parser.parse_args()

    day = args.day
    contact_distance = args.contact_distance

    random.seed(5)

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
    Amax = A.max()

    node_names = ["H{}".format(h) for h in range(n_hcw)] + ["P{}".format(p) for p in range(n_patient)]

    g = igraph.Graph.Adjacency(matrix=(A > 0).tolist())
    g.to_undirected()
    # g.es["weight"] = A[A.nonzero()]
    # A_log = np.log(A+1)
    # g.es["weight"] = A_log[A_log.nonzero()] / A_log.max()
    g.vs["label"] = node_names
    # g.vs["color"] = ["cornflowerblue"]*n_hcw + ["chocolate"]*(n_patient//2) + ["burlywood"]*(n_patient//2)
    for i, e in enumerate(g.es):
        e1, e2 = e.tuple
        if e1 < n_hcw and e2 < n_hcw:
            g.es[i]["color"] = "midnightblue"
        elif n_hcw <= e1 < n_hcw+n_patient_MWF and n_hcw <= e2 < n_hcw+n_patient_MWF:
            g.es[i]["color"] = "saddlebrown"
        elif e1 >= n_hcw+n_patient_MWF and e2 >= n_hcw+n_patient_MWF:
            g.es[i]["color"] = "darkgoldenrod"
        else:
            g.es[i]["color"] = "dimgrey"

    for i, e in enumerate(g.es):
        e1, e2 = e.tuple
        g.es[i]["weight"] = 5 * (A[e1,e2] / Amax)

    visual_style = {}
    visual_style["vertex_size"] = 22
    visual_style["vertex_color"] = ["cornflowerblue"]*n_hcw + ["chocolate"]*(n_patient//2) + ["burlywood"]*(n_patient//2)
    visual_style["vertex_label"] = g.vs["label"]
    visual_style["vertex_label_size"] = 10
    visual_style["edge_width"] = g.es["weight"]
    visual_style["edge_color"] = g.es["color"]
    visual_style["layout"] = g.layout("fr")

    igraph.plot(g, "plots/igraph/contact_network.pdf", **visual_style)
