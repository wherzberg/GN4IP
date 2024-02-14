# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:57:31 2024

@author: Billy_Herzberg
"""


from GN4IP import simulateGridData, ConvUNet, loaderCNN, trainEpochCNN

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
    width = width,
    depth = depth
)
y = simulateGridData(
    n_samples = n_samples,
    n_features = 1,
    height = height,
    width = width,
    depth = depth
)
print("x:", x.size())
print("y:", y.size())

# Make a loader from the data
data = (x, y)
loader = loaderCNN(data, batch_size=5)

# Define the model
#model = ConvUNet(channels_in=n_features, channels=7, n_pooling_layers=2)
model = ConvUNet(dim=3, channels_in=n_features, channels=7, n_pooling_layers=2)

# Set up other things
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
def lossL1(yhat, y):
    loss = torch.mean(torch.abs(yhat - y))
    return loss

loss = trainEpochCNN(model, loader, optimizer, lossL1)
print(loss)

# Test cat
loss_va = torch.zeros((0,))
for i in range(4):
    loss_va = torch.cat((loss_va, trainEpochCNN(model, loader, optimizer, lossL1).unsqueeze(0)))
print(loss_va)