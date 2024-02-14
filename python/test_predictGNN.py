# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:42:06 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, GraphUNet

import torch


# Simulate some data to make a loader from
n_samples = 10
n_nodes = 128
n_features = 2
n_pooling_layers = 1
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

# Organize the data
data_pr = (x, y, edge_index_list, clusters_list)

# Define the model
model = GraphUNet(channels_in=n_features, n_pooling_layers=n_pooling_layers)

# Set up other things
def lossL1(yhat, y):
    loss = torch.mean(torch.abs(yhat - y))
    return loss
params_pr = {
    "device" : model.getDevice(),
    "loss_function" : lossL1,
    "print_freq" : 1
}

predict_output = model.predict(data_pr, params_pr)
print("Predicted with a GNN. Output has keys:", predict_output.keys())
print("yhat:", predict_output["yhat"].size())