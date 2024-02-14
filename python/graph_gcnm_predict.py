# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:14:25 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
from main import GraphUNet, predictGCNM


# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/test_data_v0.mat"
data = loadmat(filename_load)
X0 = torch.tensor(data["X0"]).unsqueeze(2)
X1 = torch.tensor(data["X1"]).unsqueeze(2)
edge_index_list = [torch.tensor(a[0].astype(int), dtype=torch.int) - 1 for a in data["edge_index_list"]]
clusters_list = [torch.tensor(a[0].astype(int), dtype=torch.int).squeeze() - 1 for a in data["clusters_list"]]
print("Loaded data for {} samples".format(X0.size(0)))

# Standardize the data
#Xmean = X.mean()
#Xstd  = X.std()
#Ymean = Y.mean()
#Ystd  = Y.std()
Xstandard = torch.cat((X0, X1), dim=2) #(X - Xmean) / Xstd

# Organize the data and make a loader
data_pr = (Xstandard, None, edge_index_list, clusters_list)

# Define the model
channels_in      = Xstandard.size(2)
channels         = 32
channels_out     = 1
convolutions     = 4
n_pooling_layers = 0
model = GraphUNet(
    channels_in      = channels_in,
    channels         = channels,
    channels_out     = channels_out,
    convolutions     = convolutions,
    n_pooling_layers = n_pooling_layers
)

# Set up other things
def update1(x, yhat):
    print("Computing updates...")
    output = torch.cat((x[:,:,0:1], yhat), dim=2)
    return output
params_pr = {
    "n_iterations" : 2,
    "device" : model.getDevice(),
    "update_function" : update1,
    "print_freq" : 1
}

# Load the state dicts
filename_model = "../data/graph_gcnm_model_{}.pt"
state_dicts = []
for i in range(params_pr["n_iterations"]):
    state_dicts.append(torch.load(filename_model.format(i)))

# Predict with the GCNM
predict_outputs = predictGCNM(model, state_dicts, data_pr, params_pr)
print("Predicted with a GCNM. Output list of dict has keys:", predict_outputs[0].keys())

# Save some data
filename_save = "../data/graph_gcnm_test_output_v0_model_{}.mat"
for i, predict_output in enumerate(predict_outputs):
    for key in predict_output.keys():
        if type(predict_output[key]) is torch.Tensor:
            predict_output[key] = predict_output[key].detach().squeeze().numpy()
    savemat(filename_save.format(i), predict_output)
    print("Saved output as {}".format(filename_save))

