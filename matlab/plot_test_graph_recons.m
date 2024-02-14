% Check the test results

clear all; clc; close all;
run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';


%% Load the test results and reconstruction model

% Test name
test_name = "v0";

filename_load = "../data/test_data_" + test_name + ".mat";
load(filename_load, "rec_fmdl", "X2", "Y");
filename_load = "../data/graph_unet_test_output_" + test_name + ".mat";
load(filename_load, "yhat")


%% Loop through the samples and make some figures

for i = 1:10

    figure()
    tl = tiledlayout(1, 3);
    % Network input (graph)
    ax(1) = nexttile;
    img = mk_image(rec_fmdl, X2(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;
    % Network output (graph)
    ax(2) = nexttile;
    img = mk_image(rec_fmdl, yhat(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;
    % Ground Truth (graph)
    ax(3) = nexttile;
    img = mk_image(rec_fmdl, Y(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;

end
