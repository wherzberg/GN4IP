# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:16:57 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData

x, edge_index_list, clusters_list = simulateGraphData(
    n_samples = 10, 
    n_nodes = 128, 
    n_edges = 1024, 
    n_features = 1, 
    n_pooling_layers = 2
)
print(x.size())
print("Depth={}".format(len(clusters_list)))
print([edge_index.size() for edge_index in edge_index_list])
print([clusters.size() for clusters in clusters_list])