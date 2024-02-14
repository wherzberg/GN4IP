# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:55:45 2024

@author: Billy_Herzberg
"""

from scipy.io import loadmat, savemat
import torch
from GN4IP import ConvUNet


# Load the data from a *.mat file and convert everything to tensors right away
filename_load = "../data/test_data_v0.mat"
data = loadmat(filename_load)
X = torch.tensor(data["X2"]).unsqueeze(2)
closest_pixels = torch.tensor(data["closest_pixels"].astype(int), dtype=torch.int) - 1
closest_elements = torch.tensor(data["closest_elements"].astype(int), dtype=torch.int) - 1
n_pixels = data["n_pixels"].item()
print("Loaded data for {} samples".format(X.size(0)))

# Embed X, Y from the mesh to a pixel grid and get dimensions in order
Xgrid = X[:, closest_elements].reshape(X.size(0), n_pixels, n_pixels).unsqueeze(1)
print("Embedded inputs and outputs to have shape: {}".format(Xgrid.shape))

# Standardize the data
Xmean = Xgrid.mean()
Xstd  = Xgrid.std()
Xstandard = (Xgrid - Xmean) / Xstd

# Organize the data and make a loader
data_pr = (Xstandard, None)


# Define the model and load in parameters
filename_model   = "../data/conv_unet_model.pt"
channels_in      = Xstandard.size(1)
channels         = 8
channels_out     = 1
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
model.load_state_dict(torch.load(filename_model))

# Set up other things
params_pr = {
    "device" : model.getDevice(),
    "print_freq" : 1
}

# Test the CNN on the validation data
predict_output = model.predict(data_pr, params_pr)
print("Predicted with a CNN. Output has keys:", predict_output.keys())

# Unstandardize the model outputs
data = loadmat("../data/conv_unet_train_output.mat")
Ymean = data["y_mean"]
Ystd  = data["y_std"]
predict_output["yhat"] = predict_output["yhat"] * Ystd + Ymean

# Save some data
for key in predict_output.keys():
    if type(predict_output[key]) is torch.Tensor:
        predict_output[key] = predict_output[key].detach().squeeze().numpy()
filename_save = "../data/conv_unet_test_output_v0.mat"
savemat(filename_save, predict_output)
print("Saved output as {}".format(filename_save))