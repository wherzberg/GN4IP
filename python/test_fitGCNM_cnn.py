# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:30:33 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGridData, ConvUNet, fitGCNM
import torch


n_samples = 100
n_features = 2
height = 32
width = 32
x = simulateGridData(
    n_samples = n_samples,
    n_features = n_features,
    height = height,
    width = width    
)
y = simulateGridData(
    n_samples = n_samples,
    n_features = 1,
    height = height,
    width = width    
)
data_tr = (x, y)
data_va = (x, y)

# Define the model
model = ConvUNet(channels_in=n_features, channels=7, n_pooling_layers=2)

# Set up other things
def lossL1(yhat, y):
    loss = torch.mean(torch.abs(yhat - y))
    return loss
def update1(x, yhat):
    print("Computing updates...")
    output = torch.cat((x[:, 0:1, :, :], yhat), dim=1)
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
training_outputs = fitGCNM(model, data_tr, data_va, params_tr)
print("Fit a GCNM. Output list of dict has keys:", training_outputs[0].keys())