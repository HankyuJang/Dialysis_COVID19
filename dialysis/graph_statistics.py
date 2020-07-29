"""
Author: Hankyu Jang 
Email: hankyu-jang@uiowa.edu
Last Modified: July, 2020

Description: import this script to generate graph statistics
"""

import numpy as np
import networkx as nx

def generate_graph_statistics(G):
    n = nx.number_of_nodes(G)
    m = nx.number_of_edges(G)

    degree_sequence = np.array([d for n, d in nx.degree(G)])
    k_mean = degree_sequence.mean()
    k_max = degree_sequence.max()
    std = np.std(degree_sequence)
    cc = nx.average_clustering(G)
    c = nx.number_connected_components(G)
    assortativity = nx.degree_assortativity_coefficient(G)

    largest_cc = max(nx.connected_components(G), key=len)
    G_giant = G.subgraph(largest_cc)
    n_giant = nx.number_of_nodes(G_giant) 
    m_giant = nx.number_of_edges(G_giant) 

    return [n, m, k_mean, k_max, std, cc, c, assortativity, n_giant, m_giant]

