% Simulate Training Data - UNet and GCNM

clear all; clc; close all;

% Start EIDORS
% if exist('eidors','dir')==0
    run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';
% end


%% Set up

% Where to save things
filename = "../data/training_data_v0.mat";

% Number of samples
n_samples = 300;

% Model Geometry (meters)
model_radius = 0.14;
model_height = 0;
n_elec = 16;
elec_rings = 1;
elec_width = 0.02;
elec_height = 0.02;

% Simulation mesh size
sim_model_maxsz = 0.005;
sim_elec_maxsz = sim_model_maxsz / 3;

% Reconstruction mesh size
rec_model_maxsz = 0.0065;
rec_elec_maxsz = rec_model_maxsz / 3.5;

% Conductivity Distribution Parameters
background_conds = rand(n_samples, 1) * 0.03 + 0.12;
target_conds = 0.19 + (randi(2, n_samples, 1) * 2 - 3) .* (rand(n_samples, 1) * 0.03 + 0.11);
target_theta = rand(n_samples, 1) * 2*pi;
target_alpha = rand(n_samples, 1) * 0.08;
target_xs = target_alpha .* cos(target_theta);
target_ys = target_alpha .* sin(target_theta);
target_radii = rand(n_samples, 1) * 0.02 + 0.03;
min_cond = 0.001;

% Define current patterns
inj_pat = '{ad}';
meas_pat = '{mono}';
stim_pat_options = {
    'meas_current', ... % Measure on injecting electrodes
    'balance_meas' ... % This grounds the voltage data
};
amplitude = 0.003; % Amp
stim = mk_stim_patterns(n_elec, 1, inj_pat, meas_pat, stim_pat_options, amplitude);

% Added noise
noise_level = 0.01;

% How many iterations of the iterative reconstruction algorithm?
n_iterations = 3;

% How many shared nodes make two elements neighbors in the mesh?
min_shared_nodes = 2;

% How many pooling layers to account for when computing clusters
n_pooling_layers = 4;

% Ratio of each pooling layer
pooling_ratio = 1/4;

% How many repetitions of the kmeans algorithm to perform, taking the best
kmeans_reps = 10;

% Embedding to pixel grid of what size
n_pixels = 32;


%% Create the models

% % Simulation Mesh
% sim_fmdl = ng_mk_ellip_models( ...
%     [model_height, model_radius, model_radius, sim_model_maxsz], ...
%     [n_elec, elec_rings], ...
%     [elec_width, elec_height, sim_elec_maxsz] ...
% );
% % figure();
% % show_fem(sim_fmdl);
% % title(sprintf("Simulation Mesh: %d Elements", size(sim_fmdl.elems, 1)));
% 
% % Reconstruction Mesh
% rec_fmdl = ng_mk_ellip_models( ...
%     [model_height, model_radius, model_radius, rec_model_maxsz], ...
%     [n_elec, elec_rings], ...
%     [elec_width, elec_height, rec_elec_maxsz] ...
% );
% % figure();
% % show_fem(rec_fmdl);
% % title(sprintf("Reconstruction Mesh: %d Elements", size(rec_fmdl.elems, 1)));
% 
% % Set forward solver information for EIDORS
% sim_fmdl.stimulation = stim;
% sim_fmdl.solve      = @fwd_solve_1st_order;
% sim_fmdl.system_mat = @system_mat_1st_order;
% sim_fmdl.jacobian   = @jacobian_adjoint;
% rec_fmdl.stimulation = stim;
% rec_fmdl.solve      = @fwd_solve_1st_order;
% rec_fmdl.system_mat = @system_mat_1st_order;
% rec_fmdl.jacobian   = @jacobian_adjoint;
% 
% % Save the models
% save("../data/eidors_models_v0.mat", "sim_fmdl", "rec_fmdl");

% Load the models
load("../data/eidors_models_v0.mat", "sim_fmdl", "rec_fmdl");

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
VSIM = zeros(n_samples, numel(stim(1).meas_pattern));

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


%% Gather Edges From The Mesh

% Initialize edge_index array (make it larger than necessary)
edge_index = zeros(2, rec_n_elements*20);
counter = 0;

% Loop through the elements and find neighbors for each element
for i = 1:rec_n_elements

    % Get the node indicies for element_i
    nodes_i = rec_fmdl.elems(i, :);

    % See which elements share these nodes
    shared_nodes = zeros(rec_n_elements, length(nodes_i));
    for j = 1:length(nodes_i)
        shared_nodes(:, j) = any(rec_fmdl.elems == nodes_i(j), 2);
    end

    % If elements share a minimum number of nodes, there is an edge there
    sum_shared_nodes = sum(shared_nodes, 2);
    neighbor_elements = find(and(sum_shared_nodes >= min_shared_nodes, sum_shared_nodes < length(nodes_i)));
    n_neighbors = length(neighbor_elements);

    % Add the edges
    edge_index_i = [i * ones(1, n_neighbors); neighbor_elements'];
    edge_index(:, counter+1:counter+n_neighbors) = edge_index_i;
    counter = counter + n_neighbors;

end

% Remove extra columns
edge_index = edge_index(:, 1:counter);

% Sort each column, then sort the columns, and then remove duplicates
edge_index = sort(edge_index, 1);
edge_index = sortrows(edge_index', [1, 2])';
i = 2;
while i <= size(edge_index, 2)
    if edge_index(:, i) == edge_index(:, i-1)
        edge_index(:, i) = [];
    else
        i = i + 1;
    end
end


%% Assign Clusters for Graph Pooling

% Initialize storage (cell arrays) for cluster and edge_index lists
clusters_list = cell(n_pooling_layers, 1);
edge_index_list = cell(n_pooling_layers + 1, 1);
edge_index_list{1} = edge_index;

% Cluster locations start at the centroids of the elements (rec mesh)
% Note that this is hard-coded for 2D meshes!!!
centroids = [(rec_fmdl.nodes(rec_fmdl.elems(:, 1), 1) + rec_fmdl.nodes(rec_fmdl.elems(:, 2), 1) + rec_fmdl.nodes(rec_fmdl.elems(:, 3), 1)) / 3, ...
             (rec_fmdl.nodes(rec_fmdl.elems(:, 1), 2) + rec_fmdl.nodes(rec_fmdl.elems(:, 2), 2) + rec_fmdl.nodes(rec_fmdl.elems(:, 3), 2)) / 3];

% Loop through each pooling layer
n_clusters = rec_n_elements;
edge_index_i = edge_index;
for i = 1:n_pooling_layers

    % Assign clusters using k-means++ algorithm on the centroids
    n_clusters = floor(n_clusters * pooling_ratio);
    centroids_old = centroids;
    [clusters, centroids] = kmeans(centroids, n_clusters, "Replicates", kmeans_reps);

    % Perform union of edges within each cluster
    edge_index_i = reshape(clusters(edge_index_i(:)), 2,[]);
    
    % Sort each column, then sort the columns, and then remove duplicates
    edge_index_i = sort(edge_index_i, 1);
    edge_index_i = sortrows(edge_index_i', [1, 2])';
    j = 2;
    while j <= size(edge_index_i, 2)
        if edge_index_i(:, j) == edge_index_i(:, j-1)
            edge_index_i(:, j) = [];
        else
            j = j + 1;
        end
    end

    % Add clusters and edges to lists
    clusters_list{i} = clusters;
    edge_index_list{i + 1} = edge_index_i;

    % Plot for sanity
%     figure()
%     plot(centroids_old(:, 1), centroids_old(:, 2), '.'); hold on;
%     for j = 1:5
%         plot(centroids_old(clusters==j, 1), centroids_old(clusters==j, 2), 'o');
%         
%     end
%     plot(centroids(1:5, 1), centroids(1:5, 2), 'r*')

    % Plot the edges
%     figure()
%     plot(centroids(:, 1), centroids(:, 2), '*'); hold on;
%     for j = 1:size(edge_index_i, 2)
%         plot(centroids(edge_index_i(:,j), 1), centroids(edge_index_i(:,j), 2), 'b-');
%     end

end


%% Embed Mesh Data to Grid for CNN %%
% The closest pixel->element and element->pixel neighbors are needed for
% knn embedding

% Define the centers of the pixels
x = linspace(-model_radius, model_radius, n_pixels+1);
x = x(1:end-1) + (x(2) - x(1)) / 2;
[grid_x, grid_y] = meshgrid(x, x);
grid_x = grid_x(:);
grid_y = grid_y(:);

% For each pixel, find the nearest mesh element
closest_elements = zeros(size(grid_x));
for i = 1:size(grid_x, 1)
    d = (rec_centroids(:, 1) - grid_x(i)).^2 + (rec_centroids(:, 2) - grid_y(i)).^2;
    [min_d, closest_elements(i)] = min(d);
end

% For each element, find the closest pixel
closest_pixels = zeros(size(rec_centroids, 1), 1);
for i = 1:size(rec_centroids, 1)
    d = (grid_x(:) - rec_centroids(i, 1)).^2 + (grid_y(:) - rec_centroids(i, 2)).^2;
    [min_d, closest_pixels(i)] = min(d);
end

% Try it:
% img1 = mk_image(rec_fmdl, X2(1, :));
% figure()
% show_fem(img1)
% img2 = X2(1, closest_elements);
% img3 = reshape(img2, n_pixels, n_pixels);
% figure()
% imagesc(flipud(img3)); 
% axis square
% img4 = img2(closest_pixels);
% img5 = mk_image(rec_fmdl, img4);
% figure()
% show_fem(img5)


%% Save everything

save(filename, "sim_fmdl", "rec_fmdl", "VSIM", "Y", "X0", "X1", "X2", "edge_index_list", "clusters_list", "closest_pixels", "closest_elements", "n_pixels");
fprintf("Done and saved :)\n");
