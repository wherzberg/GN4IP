# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:14:25 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
import matlab.engine
from GN4IP import GraphUNet, predictGCNM

# Start a matlab engine because it will be used in the updates
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath("../matlab/", nargout=1), nargout=0)
print("Started a MATLAB engine and added to the path")

# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/test_data_v0.mat"
data = loadmat(filename_load)
X0 = torch.tensor(data["X0"]).unsqueeze(2)
X1 = torch.tensor(data["X1"]).unsqueeze(2)
edge_index_list = [torch.tensor(a[0].astype(int), dtype=torch.int) - 1 for a in data["edge_index_list"]]
clusters_list = [torch.tensor(a[0].astype(int), dtype=torch.int).squeeze() - 1 for a in data["clusters_list"]]
print("Loaded data for {} samples".format(X0.size(0)))

# Standardize the data
X0mean = X0.mean()
X0std  = X0.std()
X1mean = X1.mean()
X1std  = X1.std()
Xstandard = torch.cat(((X0 - X0mean) / X0std, (X1 - X1mean) / X1std), dim=2)

# Get standardizing info about network outputs too
data = loadmat("../data/graph_gcnm_train_output_0.mat")
Ymean = data["y_mean"]
Ystd  = data["y_std"]


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
    # Get the new sigma by unstandardizing the network output
    sigma = yhat * Ystd + Ymean
    # Compute the update (convert from tensor to matlab double first)
    sigma_mat = matlab.double(sigma.cpu().detach().numpy())
    del_sigma_mat = eng.update_function(sigma_mat, filename_load)
    del_sigma = torch.Tensor(del_sigma_mat).unsqueeze(dim=2)
    # Standardize these and concatenate to form next network's input
    sigma = (sigma - sigma.mean()) / sigma.std()
    del_sigma = (del_sigma - del_sigma.mean()) / del_sigma.std()
    output = torch.cat((sigma, del_sigma), dim=2)
    return output
params_pr = {
    "n_iterations" : 5,
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
    predict_output["yhat"] = predict_output["yhat"] * Ystd + Ymean
    for key in predict_output.keys():
        if type(predict_output[key]) is torch.Tensor:
            predict_output[key] = predict_output[key].detach().squeeze().numpy()
    savemat(filename_save.format(i), predict_output)
    print("Saved output as {}".format(filename_save.format(i)))

