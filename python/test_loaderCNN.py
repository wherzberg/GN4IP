# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 09:09:37 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGridData, loaderCNN

n_samples = 100
height = 24
width = 32
depth = 16
x = simulateGridData(
    n_samples = n_samples,
    n_features = 3,
    height = height,
    width = width    
)
y = simulateGridData(
    n_samples = n_samples,
    n_features = 1,
    height = height,
    width = width    
)
print("x:", x.size())
print("y:", y.size())

data = (x, y)
loader2d1 = loaderCNN(data)
xi, yi = next(iter(loader2d1))
print("xi:", xi.size())
print("yi:", yi.size())

loader2d2 = loaderCNN(data, batch_size=2)
xi, yi = next(iter(loader2d2))
print("xi:", xi.size())
print("yi:", yi.size())



x = simulateGridData(
    n_samples = n_samples,
    n_features = 3,
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

data = (x, y)
loader3d1 = loaderCNN(data)
xi, yi = next(iter(loader3d1))
print("xi:", xi.size())
print("yi:", yi.size())

loader3d2 = loaderCNN(data, batch_size=2)
xi, yi = next(iter(loader3d2))
print("xi:", xi.size())
print("yi:", yi.size())