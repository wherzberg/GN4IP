# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:26:33 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, GraphUNet, loaderGNN, fixEdgesAndClustersForBatch

import torch

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

# Define the model
model = GraphUNet(channels_in=n_features, channels=7, n_pooling_layers=n_pooling_layers)

# Pass the batch through the model using a forward pass
model.eval()
with torch.no_grad():
    ei_list, cl_list = fixEdgesAndClustersForBatch(edge_index_list, clusters_list, batch.batch)
    yhat = model(batch.x, ei_list, cl_list, batch.batch)
    