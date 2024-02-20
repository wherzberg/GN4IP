# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:51:56 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
import matlab.engine
from GN4IP import ConvUNet, fitGCNM

# Start a matlab engine because it will be used in the updates
eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath("../matlab/", nargout=1), nargout=0)
print("Started a MATLAB engine and added to the path")

# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/training_data_v0.mat"
data = loadmat(filename_load)
X0 = torch.tensor(data["X0"]).unsqueeze(2)
X1 = torch.tensor(data["X1"]).unsqueeze(2)
Y = torch.tensor(data["Y"]).unsqueeze(2)
closest_pixels = torch.tensor(data["closest_pixels"].astype(int), dtype=torch.int) - 1
closest_elements = torch.tensor(data["closest_elements"].astype(int), dtype=torch.int) - 1
n_pixels = data["n_pixels"].item()
print("Loaded data for {} samples".format(X0.size(0)))


# Embed X, Y from the mesh to a pixel grid and get dimensions in order
X0grid = X0[:, closest_elements].reshape(X0.size(0), n_pixels, n_pixels).unsqueeze(1)
X1grid = X1[:, closest_elements].reshape(X1.size(0), n_pixels, n_pixels).unsqueeze(1)
Ygrid = Y[:, closest_elements].reshape(Y.size(0), n_pixels, n_pixels).unsqueeze(1)
print("Embedded inputs and outputs to have shape: {}".format(X0grid.shape))

# Standardize the data
X0mean = X0grid.mean()
X0std  = X0grid.std()
X1mean = X1grid.mean()
X1std  = X1grid.std()
Ymean = Ygrid.mean()
Ystd  = Ygrid.std()
Xstandard = torch.cat(((X0grid - X0mean) / X0std, (X1grid - X1mean) / X1std), dim=1)
Ystandard = (Ygrid - Ymean) / Ystd

# Organize the data and make a loader
split = round(Xstandard.size(0) * 0.8)
data_tr = (Xstandard[0:split], Ystandard[0:split])
data_va = (Xstandard[split: ], Ystandard[split: ])

# Define the model
channels_in      = Xstandard.size(1)
channels         = 8
channels_out     = Ystandard.size(1)
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
def lossL2(yhat, y):
    loss = torch.mean(torch.square(yhat - y))
    return loss
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
params_tr = {
    "n_iterations" : 5,
    "batch_size" : 16,
    "device" : model.getDevice(),
    "learning_rate" : 0.001,
    "max_epochs" : 500,
    "loss_function" : lossL2,
    "update_function" : update1,
    "patience" : 20,
    "print_freq" : 1
}

# Fit the GCNM
training_outputs = fitGCNM(model, data_tr, data_va, params_tr)
print("Fit a GCNM. Output list of dict has keys:", training_outputs[0].keys())

# Save the trained models (the state dict) and other stuff for each model
filename_model = "../data/conv_gcnm_model_{}.pt"
filename_save = "../data/conv_gcnm_train_output_{}.mat"
for i, training_output in enumerate(training_outputs):
    
    torch.save(training_output["state_dict"], filename_model.format(i))
    print("Saved the model as {}".format(filename_model.format(i)))
    
    del training_output["state_dict"]
    training_output["y_mean"] = Ymean
    training_output["y_std"] = Ystd
    for key in training_output.keys():
        if type(training_output[key]) is torch.Tensor:
            training_output[key] = training_output[key].detach().squeeze().numpy()
    savemat(filename_save.format(i), training_output)
    print("Saved output as {}".format(filename_save.format(i)))


