% Check the training results

% clear all; clc; close all;

%% Load the training results

filename_load = "../data/conv_unet_train_output.mat";
% filename_load = "../data/graph_unet_train_output.mat";
% filename_load = "../data/graph_gcnm_train_output_0.mat";
% filename_load = "../data/graph_gcnm_train_output_1.mat";
load(filename_load, "loss_tr", "loss_va")


%% Plot the losses

% How many epochs
epochs = 1:length(loss_tr);

% Make the figure
figure()
plot(epochs, loss_tr, "DisplayName","Training"); hold on;
plot(epochs, loss_va, "DisplayName","Validation"); grid on;
title("Loss")
xlabel("Epoch")
legend("Location","north")
ylim([0 1])

% Add a line for where weights were saved
[~, min_epoch] = min(loss_va);
xline(min_epoch, 'k--', "DisplayName","Best Weights")
