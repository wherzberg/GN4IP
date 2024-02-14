# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:52:58 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
from GN4IP import GraphUNet


# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/training_data_v0.mat"
data = loadmat(filename_load)
X = torch.tensor(data["X2"]).unsqueeze(2)
Y = torch.tensor(data["Y"]).unsqueeze(2)
edge_index_list = [torch.tensor(a[0].astype(int), dtype=torch.int) - 1 for a in data["edge_index_list"]]
clusters_list = [torch.tensor(a[0].astype(int), dtype=torch.int).squeeze() - 1 for a in data["clusters_list"]]
print("Loaded data for {} samples".format(X.size(0)))


# Standardize the data
Xmean = X.mean()
Xstd  = X.std()
Ymean = Y.mean()
Ystd  = Y.std()
Xstandard = (X - Xmean) / Xstd
Ystandard = (Y - Ymean) / Ystd

# Organize the data and make a loader
split = round(Xstandard.size(0) * 0.8)
data_tr = (Xstandard[0:split], Ystandard[0:split], edge_index_list, clusters_list)
data_va = (Xstandard[split: ], Ystandard[split: ], edge_index_list, clusters_list)

# Define the model
channels_in      = Xstandard.size(2)
channels         = 16
channels_out     = Ystandard.size(2)
convolutions     = 2
n_pooling_layers = 4
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
params_tr = {
    "batch_size" : 16,
    "device" : model.getDevice(),
    "learning_rate" : 0.001,
    "max_epochs" : 500,
    "loss_function" : lossL2,
    "patience" : 20,
    "print_freq" : 10
}

# Fit the GNN
training_output = model.fit(data_tr, data_va, params_tr)
print("Fit a GNN. Output has keys:", training_output.keys())

# Save the trained model (the state dict)
filename_model = "../data/graph_unet_model.pt"
torch.save(model.state_dict(), filename_model)
print("Saved the model as {}".format(filename_model))

# Save some data
training_output["x_mean"] = Xmean
training_output["x_std"] = Xstd
training_output["y_mean"] = Ymean
training_output["y_std"] = Ystd
for key in training_output.keys():
    if type(training_output[key]) is torch.Tensor:
        training_output[key] = training_output[key].detach().squeeze().numpy()
filename_save = "../data/graph_unet_train_output.mat"
savemat(filename_save, training_output)
print("Saved output as {}".format(filename_save))