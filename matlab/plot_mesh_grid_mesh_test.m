% Check the mesh to grid to mesh test data

clear all; clc; close all;
run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';


%% Setup

% Load the model
filename_load = "../data/test_data_v0.mat";
load(filename_load, "rec_fmdl");

% Load the data
filename_load = "../data/mesh_to_grid_to_mesh_test.mat";
load(filename_load, "x_in", "x_grid", "x_mesh");


%%

% Plot the first sample on the mesh, grid, and mesh
sample_ind = 1;
figure()
subplot(1, 3, 1)
    img = mk_image(rec_fmdl, x_in(sample_ind, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;
subplot(1, 3, 2)
    imagesc(flipud(squeeze(x_grid(sample_ind, :, :))')); % transpose then flipud for python->matlab row/column fix and then matlab imagesc row flip fix
    axis image; axis off; clim([-0.05 0.35]);
    colorbar
subplot(1, 3, 3)
    img = mk_image(rec_fmdl, x_mesh(sample_ind, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;