# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 08:01:03 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
from GN4IP import ConvUNet


# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/training_data_v0.mat"
data = loadmat(filename_load)
X = torch.tensor(data["X2"]).unsqueeze(2)
Y = torch.tensor(data["Y"]).unsqueeze(2)
closest_pixels = torch.tensor(data["closest_pixels"].astype(int), dtype=torch.int) - 1
closest_elements = torch.tensor(data["closest_elements"].astype(int), dtype=torch.int) - 1
n_pixels = data["n_pixels"].item()
print("Loaded data for {} samples".format(X.size(0)))


# Embed X, Y from the mesh to a pixel grid and get dimensions in order
Xgrid = X[:, closest_elements].reshape(X.size(0), n_pixels, n_pixels).unsqueeze(1)
Ygrid = Y[:, closest_elements].reshape(Y.size(0), n_pixels, n_pixels).unsqueeze(1)
print("Embedded inputs and outputs to have shape: {}".format(Xgrid.shape))

# Standardize the data
Xmean = Xgrid.mean()
Xstd  = Xgrid.std()
Ymean = Ygrid.mean()
Ystd  = Ygrid.std()
Xstandard = (Xgrid - Xmean) / Xstd
Ystandard = (Ygrid - Ymean) / Ystd

# Organize the data and make a loader
split = round(X.size(0) * 0.8)
data_tr = (Xstandard[0:split], Ystandard[0:split])
data_va = (Xstandard[split: ], Ystandard[split: ])

# Define the model
channels_in      = Xstandard.size(1)
channels         = 8
channels_out     = Ystandard.size(1)
convolutions     = 2
n_pooling_layers = 4
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
params_tr = {
    "batch_size" : 16,
    "device" : model.getDevice(),
    "learning_rate" : 0.001,
    "max_epochs" : 500,
    "loss_function" : lossL2,
    "patience" : 20,
    "print_freq" : 10
}

# Fit the CNN
training_output = model.fit(data_tr, data_va, params_tr)
print("Fit a CNN. Output has keys:", training_output.keys())

# Save the trained model (the state dict)
filename_model = "../data/conv_unet_model.pt"
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
filename_save = "../data/conv_unet_train_output.mat"
savemat(filename_save, training_output)
print("Saved output as {}".format(filename_save))