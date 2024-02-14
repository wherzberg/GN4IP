# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 07:50:17 2024

@author: Billy_Herzberg
"""

from GN4IP import simulateGridData

# Simulate 2d data by leaving depth=None (default)
x = simulateGridData(
    n_samples = 10, 
    n_features = 1, 
    height = 16, 
    width = 32
)
print(x.size())

# Simulate 3d data by defining depth=20 (any int>0)
x = simulateGridData(
    n_samples = 12, 
    n_features = 2,  
    height = 16,
    width = 32,
    depth = 20
)
print(x.size())