# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 10:15:02 2024

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
data_pr = (x, y)

# Define the model
model = ConvUNet(channels_in=n_features, channels=7, n_pooling_layers=2)

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
print("Predicted with a CNN. Output has keys:", predict_output.keys())
print("yhat:", predict_output["yhat"].size())
