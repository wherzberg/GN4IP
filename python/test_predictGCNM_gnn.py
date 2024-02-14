# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:54:51 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, GraphUNet, fitGCNM, predictGCNM
import torch


# Simulate some data to make a loader from
n_samples = 10
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
data_va = (x[0:5], y[0:5], edge_index_list, clusters_list)

# Define the model
model = GraphUNet(channels_in=n_features, n_pooling_layers=n_pooling_layers)

# Set up other things
def lossL1(yhat, y):
    loss = torch.mean(torch.abs(yhat - y))
    return loss
def update1(x, yhat):
    print("Computing updates...")
    output = torch.cat((x[:,:,0:1], yhat), dim=2)
    return output
params_tr = {
    "n_iterations" : 2,
    "batch_size" : 10,
    "device" : model.getDevice(),
    "learning_rate" : 0.001,
    "max_epochs" : 10,
    "loss_function" : lossL1,
    "update_function" : update1,
    "patience" : 1,
    "print_freq" : 0
}

# Fit the GCNM
print("Fitting the GCNM!!")
training_outputs = fitGCNM(model, data_tr, data_va, params_tr)

# Predict with the GCNM
params_tr["print_freq"] = 1
state_dicts = [a["state_dict"] for a in training_outputs]
print()
print()
print("Predicting with the GCNM!!")
predict_outputs = predictGCNM(model, state_dicts, data_va, params_tr)
print("Predicted with a GCNM. Output list of dict has keys:", predict_outputs[0].keys())

