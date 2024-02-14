# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:37:08 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
from GN4IP import GraphUNet


# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/test_data_v0.mat"
data = loadmat(filename_load)
X = torch.tensor(data["X2"]).unsqueeze(2)
edge_index_list = [torch.tensor(a[0].astype(int), dtype=torch.int) - 1 for a in data["edge_index_list"]]
clusters_list = [torch.tensor(a[0].astype(int), dtype=torch.int).squeeze() - 1 for a in data["clusters_list"]]
print("Loaded data for {} samples".format(X.size(0)))

# Standardize the data
Xmean = X.mean()
Xstd  = X.std()
Xstandard = (X - Xmean) / Xstd

# Organize the data and make a loader
data_pr = (Xstandard, None, edge_index_list, clusters_list)

# Define the model and load in parameters
filename_model   = "../data/graph_unet_model.pt"
channels_in      = Xstandard.size(2)
channels         = 16
channels_out     = 1
convolutions     = 2
n_pooling_layers = 4
model = GraphUNet(
    channels_in      = channels_in,
    channels         = channels,
    channels_out     = channels_out,
    convolutions     = convolutions,
    n_pooling_layers = n_pooling_layers
)
model.load_state_dict(torch.load(filename_model))

# Set up other things
params_pr = {
    "device" : model.getDevice(),
    "print_freq" : 1
}

# Test the GNN on the validation data
predict_output = model.predict(data_pr, params_pr)
print("Predicted with a GNN. Output has keys:", predict_output.keys())

# Unstandardize the model outputs
data = loadmat("../data/graph_unet_train_output.mat")
Ymean = data["y_mean"]
Ystd  = data["y_std"]
predict_output["yhat"] = predict_output["yhat"] * Ystd + Ymean

# Save some data
for key in predict_output.keys():
    if type(predict_output[key]) is torch.Tensor:
        predict_output[key] = predict_output[key].detach().squeeze().numpy()
filename_save = "../data/graph_unet_test_output_v0.mat"
savemat(filename_save, predict_output)
print("Saved output as {}".format(filename_save))

