% Check the test results

clear all; clc; close all;
run 'path_to_startup\eidors\startup';


%% Load the test results and reconstruction model

% Test name
test_name = "v0";

filename_load = "../data/test_data_" + test_name + ".mat";
load(filename_load, "rec_fmdl", "X2", "Y", "n_pixels", "closest_elements", "closest_pixels");
filename_load = "../data/conv_unet_test_output_" + test_name + ".mat";
load(filename_load, "yhat")


%% Loop through the samples and make some figures

for i = 1:3

    figure()
    tl = tiledlayout(2, 3);
    % Network input (graph)
    ax(1) = nexttile;
    img = mk_image(rec_fmdl, X2(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;
    % Network output (graph)
    ax(2) = nexttile;
    yhat_graph = squeeze(yhat(i, :, :))'; yhat_graph = yhat_graph(closest_pixels);
    img = mk_image(rec_fmdl, yhat_graph);
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;
    % Ground Truth (graph)
    ax(3) = nexttile;
    img = mk_image(rec_fmdl, Y(i, :));
    img.calc_colours.ref_level = 0.15; img.calc_colours.clim = 0.2;
    show_fem(img); eidors_colourbar(img); axis off;

    % Network input (grid)
    ax(4) = nexttile;
    imagesc(flipud(reshape(X2(i, closest_elements), n_pixels, n_pixels)));
    axis image; axis off; caxis([-0.05 0.35]);
    colorbar
    % Network output (grid)
    ax(5) = nexttile;
    imagesc(flipud(squeeze(yhat(i, :, :))'));
    axis image; axis off; caxis([-0.05 0.35]);
    colorbar

end
