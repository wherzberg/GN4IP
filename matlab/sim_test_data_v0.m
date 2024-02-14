% Simulate Testing Data - UNet and GCNM
% Test Set 0 - Similar to training data

clear all; clc; close all;

% Start EIDORS
% if exist('eidors','dir')==0
    run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';
% end


%% Set up

% Where to save things
filename = "../data/test_data_v0.mat";

% Number of samples
n_samples = 10;

% Conductivity Distribution Parameters
background_conds = rand(n_samples, 1) * 0.03 + 0.12;
target_conds = 0.19 + (randi(2, n_samples, 1) * 2 - 3) .* (rand(n_samples, 1) * 0.03 + 0.11);
target_theta = rand(n_samples, 1) * 2*pi;
target_alpha = rand(n_samples, 1) * 0.08;
target_xs = target_alpha .* cos(target_theta);
target_ys = target_alpha .* sin(target_theta);
target_radii = rand(n_samples, 1) * 0.02 + 0.03;
min_cond = 0.001;

% Added noise
noise_level = 0.01;

% How many iterations of the iterative reconstruction algorithm?
n_iterations = 3;

% Embedding to pixel grid of what size
n_pixels = 32;


%% Load the EIDORS models from the training data

% Load some things from the training data
filename_sim = "../data/training_data_v0.mat";
load(filename_sim, "sim_fmdl", "rec_fmdl", "edge_index_list", "clusters_list", "closest_pixels", "closest_elements", "n_pixels");

% Simulation mesh centroids
sim_n_elements = size(sim_fmdl.elems, 1);
sim_centroids = [(sim_fmdl.nodes(sim_fmdl.elems(:, 1), 1) + sim_fmdl.nodes(sim_fmdl.elems(:, 2), 1) + sim_fmdl.nodes(sim_fmdl.elems(:, 3), 1)) / 3, ...
                 (sim_fmdl.nodes(sim_fmdl.elems(:, 1), 2) + sim_fmdl.nodes(sim_fmdl.elems(:, 2), 2) + sim_fmdl.nodes(sim_fmdl.elems(:, 3), 2)) / 3];

% Reconstruction mesh centroids
rec_n_elements = size(rec_fmdl.elems, 1);
rec_centroids = [(rec_fmdl.nodes(rec_fmdl.elems(:, 1), 1) + rec_fmdl.nodes(rec_fmdl.elems(:, 2), 1) + rec_fmdl.nodes(rec_fmdl.elems(:, 3), 1)) / 3, ...
                 (rec_fmdl.nodes(rec_fmdl.elems(:, 1), 2) + rec_fmdl.nodes(rec_fmdl.elems(:, 2), 2) + rec_fmdl.nodes(rec_fmdl.elems(:, 3), 2)) / 3];


%% Simulate the data

% Initialize storage
VSIM = zeros(n_samples, numel(rec_fmdl.stimulation(1).meas_pattern));

% For each sample, set the conductivity and then solve the forward problem
for i = 1:n_samples
    
    % Set the conductivity
    condunctivity = ones(sim_n_elements, 1) * background_conds(i);
    d = sqrt((sim_centroids(:, 1) - target_xs(i)).^2 + (sim_centroids(:, 2) - target_ys(i)).^2);
    condunctivity(d < target_radii(i)) = target_conds(i);

    % Solve the forward problem
    img = mk_image(sim_fmdl, condunctivity);
    U = fwd_solve(img);
%     figure()
%     show_fem(img)

    % Add noise to the simulated data
    Vsim = U.meas + noise_level * mean(abs(U.meas), 1) * normrnd(0, 1, size(U.meas));

    % Store the data
    VSIM(i, :) = Vsim(:);

end



%% Calculate Reconstructions (network inputs)

% Initialize storage
X0 = zeros(n_samples, rec_n_elements); % Initialization: sigma_0
X1 = zeros(n_samples, rec_n_elements); % First update term: del_sigma_0
X2 = zeros(n_samples, rec_n_elements); % Third iterate: sigma_3
Y  = zeros(n_samples, rec_n_elements); % Ground Truth

% Solve the forward problem with conductivity of 1
img1 = mk_image(rec_fmdl, 1);
U1 = fwd_solve(img1);

% For each sample, get initial conductivity and do three iterations
for i = 1:n_samples
    fprintf("Sample %d\n", i);

    % Get ground truth
    conductivity = ones(rec_n_elements, 1) * background_conds(i);
    d = sqrt((rec_centroids(:, 1) - target_xs(i)).^2 + (rec_centroids(:, 2) - target_ys(i)).^2);
    conductivity(d < target_radii(i)) = target_conds(i);
    Y(i, :) = conductivity;

    % Get initialization
    conductivity_hom = VSIM(i, :)' \ U1.meas(:);
    sigma = ones(rec_n_elements, 1) * conductivity_hom;
    X0(i, :) = sigma;


    % Do three iterations of LM algorithm
    for j = 1:n_iterations

        % Solve the forward problem on current sigma
        img = mk_image(rec_fmdl, sigma);
        Ui  = fwd_solve(img);

        % Compute an LM Update
        J = calc_jacobian(img);
        JtJ = J'*J;
        lambda_param = 0.05 * max(diag(JtJ));
        del_sigma = - (J'*J + lambda_param * eye(size(JtJ,1))) \ (J' * (Ui.meas(:) - VSIM(i, :)'));
        if i == 1
            X1(i, :) = del_sigma;
        end

        % Add the update using a linesearch
        step_size_best = fminbnd(@(step_size) objective_function(step_size, sigma, del_sigma, rec_fmdl, VSIM(i, :)), 0, 1);
%         fprintf("Step Size: %.3f\n", step_size_best);
        sigma = sigma + step_size_best * del_sigma;

        % Make sure no elements have a conductivity less than min_cond
        sigma(sigma < min_cond) = min_cond;

    end
    X2(i, :) = sigma;

    % Plot for sanity
%     figure()
%     img = mk_image(rec_fmdl, conductivity);
%     img.calc_colours.ref_level = background_conds(i);
%     img.calc_colours.clim = 0.21;
%     subplot(1,2,1); show_fem(img); eidors_colourbar(img)
%     img = mk_image(rec_fmdl, sigma);
%     img.calc_colours.ref_level = background_conds(i);
%     img.calc_colours.clim = 0.21;
%     subplot(1,2,2); show_fem(img); eidors_colourbar(img)

end




%% Save everything

save(filename, "sim_fmdl", "rec_fmdl", "VSIM", "Y", "X0", "X1", "X2", "edge_index_list", "clusters_list", "closest_pixels", "closest_elements", "n_pixels");
fprintf("Done and saved :)\n");








