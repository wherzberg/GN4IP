# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:51:47 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGraphData, GraphUNet, loaderGNN, trainEpochGNN

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
data = (x, y, edge_index_list, clusters_list)
loader = loaderGNN(data, batch_size=3)

# Define the model
model = GraphUNet(channels_in=n_features, n_pooling_layers=n_pooling_layers)

# Set up other things
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
def lossL1(yhat, y):
    loss = torch.mean(torch.abs(yhat - y))
    return loss

loss = trainEpochGNN(model, loader, data[2], data[3], optimizer, lossL1)
print(loss)

# Test cat
loss_tr = torch.zeros((0,))
for i in range(4):
    loss_tr = torch.cat((loss_tr, trainEpochGNN(model, loader, data[2], data[3], optimizer, lossL1).unsqueeze(0)))
print(loss_tr)
