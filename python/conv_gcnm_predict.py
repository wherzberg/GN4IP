# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:27:08 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
import matlab.engine
from GN4IP import ConvUNet, predictGCNM

# Start a matlab engine because it will be used in the updates
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath("../matlab/", nargout=1), nargout=0)
print("Started a MATLAB engine and added to the path")

# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/test_data_v0.mat"
data = loadmat(filename_load)
X0 = torch.tensor(data["X0"]).unsqueeze(2)
X1 = torch.tensor(data["X1"]).unsqueeze(2)
closest_pixels = torch.tensor(data["closest_pixels"].astype(int), dtype=torch.int) - 1
closest_elements = torch.tensor(data["closest_elements"].astype(int), dtype=torch.int) - 1
n_pixels = data["n_pixels"].item()
print("Loaded data for {} samples".format(X0.size(0)))


# Embed X, Y from the mesh to a pixel grid and get dimensions in order
X0grid = X0[:, closest_elements].reshape(X0.size(0), n_pixels, n_pixels).unsqueeze(1)
X1grid = X1[:, closest_elements].reshape(X1.size(0), n_pixels, n_pixels).unsqueeze(1)
print("Embedded inputs and outputs to have shape: {}".format(X0grid.shape))

# Standardize the data
X0mean = X0grid.mean()
X0std  = X0grid.std()
X1mean = X1grid.mean()
X1std  = X1grid.std()
Xstandard = torch.cat(((X0grid - X0mean) / X0std, (X1grid - X1mean) / X1std), dim=1)

# Get standardizing info about network outputs too
data = loadmat("../data/conv_gcnm_train_output_0.mat")
Ymean = data["y_mean"]
Ystd  = data["y_std"]

# Organize the data and make a loader
data_pr = (Xstandard, None)

# Define the model
channels_in      = Xstandard.size(1)
channels         = 8
channels_out     = 1
convolutions     = 2
n_pooling_layers = 0
model = ConvUNet(
    dim              = 2,
    channels_in      = channels_in,
    channels         = channels,
    channels_out     = channels_out,
    convolutions     = convolutions,
    n_pooling_layers = n_pooling_layers,
    kernel_size      = 3
)

# Set up other things
def update1(x, yhat):
    print("Computing updates...")
    # Get the new sigma by unstandardizing the network output
    sigma = yhat * Ystd + Ymean
    # Convert from grid data to the mesh
    sigma_mesh = torch.reshape(sigma.squeeze(), (sigma.size(0), -1))
    sigma_mesh = sigma_mesh[:, closest_pixels].squeeze()
    # Compute the update (convert from tensor to matlab double first)
    sigma_mat = matlab.double(sigma_mesh.cpu().detach().numpy())
    del_sigma_mat = eng.update_function(sigma_mat, filename_load)
    del_sigma = torch.Tensor(del_sigma_mat).unsqueeze(dim=2)
    # Convert from mesh back to grid
    del_sigma = del_sigma[:, closest_elements].reshape(sigma.size(0), n_pixels, n_pixels).unsqueeze(1)
    # Standardize these and concatenate to form next network's input
    sigma = (sigma - sigma.mean()) / sigma.std()
    del_sigma = (del_sigma - del_sigma.mean()) / del_sigma.std()
    output = torch.cat((sigma, del_sigma), dim=1)
    return output
params_pr = {
    "n_iterations" : 5,
    "device" : model.getDevice(),
    "update_function" : update1,
    "print_freq" : 1
}

# Load the state dicts
filename_model = "../data/conv_gcnm_model_{}.pt"
state_dicts = []
for i in range(params_pr["n_iterations"]):
    state_dicts.append(torch.load(filename_model.format(i)))

# Predict with the GCNM
predict_outputs = predictGCNM(model, state_dicts, data_pr, params_pr)
print("Predicted with a GCNM. Output list of dict has keys:", predict_outputs[0].keys())

# Save some data
filename_save = "../data/conv_gcnm_test_output_v0_model_{}.mat"
for i, predict_output in enumerate(predict_outputs):
    predict_output["yhat"] = predict_output["yhat"] * Ystd + Ymean
    for key in predict_output.keys():
        if type(predict_output[key]) is torch.Tensor:
            predict_output[key] = predict_output[key].detach().squeeze().numpy()
    savemat(filename_save.format(i), predict_output)
    print("Saved output as {}".format(filename_save.format(i)))


