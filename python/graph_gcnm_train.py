# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 07:27:01 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
import matlab.engine
from main import GraphUNet, fitGCNM

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
edge_index_list = [torch.tensor(a[0].astype(int), dtype=torch.int) - 1 for a in data["edge_index_list"]]
clusters_list = [torch.tensor(a[0].astype(int), dtype=torch.int).squeeze() - 1 for a in data["clusters_list"]]
print("Loaded data for {} samples".format(X0.size(0)))

# Standardize the data
X0mean = X0.mean()
X0std  = X0.std()
X1mean = X1.mean()
X1std  = X1.std()
Ymean = Y.mean()
Ystd  = Y.std()
Xstandard = torch.cat(((X0 - X0mean) / X0std, (X1 - X1mean) / X1std), dim=2)
Ystandard = (Y - Ymean) / Ystd

# Organize the data and make a loader
split = round(Xstandard.size(0) * 0.8)
data_tr = (Xstandard[0:split], Ystandard[0:split], edge_index_list, clusters_list)
data_va = (Xstandard[split: ], Ystandard[split: ], edge_index_list, clusters_list)

# Define the model
channels_in      = Xstandard.size(2)
channels         = 32
channels_out     = Ystandard.size(2)
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
def lossL2(yhat, y):
    loss = torch.mean(torch.square(yhat - y))
    return loss
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
params_tr = {
    "n_iterations" : 2,
    "batch_size" : 16,
    "device" : model.getDevice(),
    "learning_rate" : 0.001,
    "max_epochs" : 2,
    "loss_function" : lossL2,
    "update_function" : update1,
    "patience" : 10,
    "print_freq" : 1
}

# Fit the GCNM
training_outputs = fitGCNM(model, data_tr, data_va, params_tr)
print("Fit a GCNM. Output list of dict has keys:", training_outputs[0].keys())

# Save the trained models (the state dict) and other stuff for each model
filename_model = "../data/graph_gcnm_model_{}.pt"
filename_save = "../data/graph_gcnm_train_output_{}.mat"
for i, training_output in enumerate(training_outputs):
    
    torch.save(training_output["state_dict"], filename_model.format(i))
    print("Saved the model as {}".format(filename_model.format(i)))
    
    del training_output["state_dict"]
    for key in training_output.keys():
        if type(training_output[key]) is torch.Tensor:
            training_output[key] = training_output[key].detach().squeeze().numpy()
    savemat(filename_save.format(i), training_output)
    print("Saved output as {}".format(filename_save.format(i)))