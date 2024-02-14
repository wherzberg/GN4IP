# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:00:12 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGridData, loaderCNN, ConvUNet
import torch

n_samples = 100
n_features = 3
height = 32
width = 32
depth = 32
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
print("x:", x.size(), x.type())
print("y:", y.size())

data = (x, y)
loader2d1 = loaderCNN(data, batch_size=5)
xi, yi = next(iter(loader2d1))
print("  xi:", xi.size())

# Define the model
model = ConvUNet(channels_in=n_features, channels=7, n_pooling_layers=2)

# Pass the batch through the model using a forward pass
model.eval()
with torch.no_grad():
    yhat = model(xi)
print("yhat:", yhat.size())
    