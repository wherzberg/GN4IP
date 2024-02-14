# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:25:03 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGridData, ConvUNet

import torch


n_samples = 100
n_features = 3
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

training_output = model.fit(data_tr, data_va, params_tr)
print("Fit a CNN. Output has keys:", training_output.keys())
