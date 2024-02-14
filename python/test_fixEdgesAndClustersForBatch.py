# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:17:30 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, loaderGNN, fixEdgesAndClustersForBatch

# Simulate some data to make a loader from
n_samples = 10
n_nodes = 16
n_features = 2
n_pooling_layers = 1
x, edge_index_list, clusters_list = simulateGraphData(
    n_samples = n_samples, 
    n_nodes = n_nodes,
    n_features = n_features,   
    n_pooling_layers = n_pooling_layers,
    n_edges = n_nodes*2
)
y, _, _ = simulateGraphData(
    n_samples = n_samples, 
    n_nodes = n_nodes,
    n_features = 1,  
    n_edges = 0,
    n_pooling_layers = n_pooling_layers
)

# Organize the data, make a loader, and get the first batch
data = (x, y, edge_index_list, clusters_list)
batch_size = 3
loader = loaderGNN(data, batch_size=batch_size)
batch = next(iter(loader))

# Test fixing the edges and clusters
print("Before Fixing ({} Nodes in main graph)".format(n_nodes))
print("Edges Sizes")
print([ei.size() for ei in edge_index_list])
print("Clusters Sizes")
print([cl.size() for cl in clusters_list])
ei_list, cl_list = fixEdgesAndClustersForBatch(edge_index_list, clusters_list, batch.batch)
print("After Fixing ({} graphs per batch)".format(batch_size))
print("Edges Sizes")
print([ei.size() for ei in ei_list])
print("Clusters Sizes")
print([cl.size() for cl in cl_list])