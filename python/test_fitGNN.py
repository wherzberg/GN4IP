# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 08:05:50 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, GraphUNet

import torch

# Simulate some data to make a loader from
n_samples = 50
n_nodes = 128
n_features = 2
n_pooling_layers = 0
x, edge_index_list, clusters_list = simulateGraphData(
    n_samples = n_samples, 
    n_nodes = n_nodes,
    n_features = n_features,   
    n_pooling_layers = n_pooling_layers,
    n_edges = 1024
)
y, _, _ = simulateGraphData(
    n_samples = n_samples, 
    n_nodes = n_nodes,
    n_features = 1,  
    n_edges = 0,
    n_pooling_layers = 0
)

# Organize the data and make a loader
data_tr = (x, y, edge_index_list, clusters_list)
data_va = (x, y, edge_index_list, clusters_list)

# Define the model
model = GraphUNet(channels_in=n_features, n_pooling_layers=n_pooling_layers)

# Set up other things
def lossL1(yhat, y):
    loss = torch.mean(torch.abs(yhat - y))
    return loss
params_tr = {
    "n_iterations" : 2,
    "batch_size" : 4,
    "device" : model.getDevice(),
    "learning_rate" : 0.001,
    "max_epochs" : 10,
    "loss_function" : lossL1,
    "patience" : 2,
    "print_freq" : 3
}

# Fit the GNN
training_output = model.fit(data_tr, data_va, params_tr)
print("Fit a GNN. Output has keys:", training_output.keys())

