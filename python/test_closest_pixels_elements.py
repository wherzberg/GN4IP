# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 08:12:35 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch


# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/test_data_v0.mat"
data = loadmat(filename_load)
X = torch.tensor(data["X2"]).unsqueeze(2)
closest_pixels = torch.tensor(data["closest_pixels"].astype(int), dtype=torch.int) - 1
closest_elements = torch.tensor(data["closest_elements"].astype(int), dtype=torch.int) - 1
n_pixels = data["n_pixels"].item()
print("Loaded data for {} samples".format(X.size(0)))


# Embed X from the mesh to a pixel grid and get dimensions in order
Xgrid = X[:, closest_elements].reshape(X.size(0), n_pixels, n_pixels).unsqueeze(1)
print("Embedded inputs to have shape: {}".format(Xgrid.shape))

# Embed Xgrid from the pixel grid back to a mesh
Xmesh = torch.reshape(Xgrid.squeeze(), (X.size(0), -1))
Xmesh = Xmesh[:, closest_pixels]
print("Embedded inputs to have shape: {}".format(Xmesh.shape))

# Save results to show in matlab
output_dict = {
    "x_in" : X.numpy().squeeze(),
    "x_grid" : Xgrid.numpy().squeeze(),
    "x_mesh" : Xmesh.numpy().squeeze()    
}
savemat("../data/mesh_to_grid_to_mesh_test.mat", output_dict)