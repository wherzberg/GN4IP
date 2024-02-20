% Check the test results

clear all; clc; %close all;
run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';


%% Load the test results and reconstruction model

% Test name
test_name = "v0";

% How many iterations were done?
n_iterations = 5;

% Load the FEM model, first network input, and ground truth
filename_load = "../data/test_data_" + test_name + ".mat";
load(filename_load, "rec_fmdl", "X0", "Y", "n_pixels", "closest_elements", "closest_pixels");

% Load the output of each neural network and store it
xs = cell(1, n_iterations);
yhats = cell(1, n_iterations);
for i = 1:n_iterations
    filename_load = "../data/conv_gcnm_test_output_" + test_name + "_model_" + num2str(i-1) + ".mat";
    load(filename_load, "x", "yhat")
    xs{i} = x;
    yhats{i} = yhat;
end


%% Loop through the samples and make some figures

for i = 1:5

    figure()
    tl = tiledlayout(2, n_iterations + 2);
    % First network input (graph)
    ax(1) = nexttile;
    img = mk_image(rec_fmdl, X0(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;
    % Network outputs (graph)
    for j = 1:n_iterations
        ax(j+1) = nexttile;
        yhat_graph = squeeze(yhats{j}(i, :, :))'; yhat_graph = yhat_graph(closest_pixels);
        img = mk_image(rec_fmdl, yhat_graph);
        img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
        show_fem(img); eidors_colourbar(img); axis off;
    end
    % Ground Truth (graph)
    ax(n_iterations+2) = nexttile;
    img = mk_image(rec_fmdl, Y(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;

    % Network outputs (grid)
    ax(n_iterations+3) = nexttile; axis off;
    for j = 1:n_iterations
        ax(n_iterations+3+j) = nexttile;
        imagesc(flipud(squeeze(yhats{j}(i, :, :))'));
        axis image; axis off; clim([-0.05 0.35]);
        colorbar
    end
end