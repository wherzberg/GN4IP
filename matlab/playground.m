close all; clear; clc;

% Start EIDORS
if exist('eidors','dir')==0
    run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';
end

%% CREATING A CUSTOM MODEL WITH ELECTRODES%%
% Create a custom 2d elliptical model with defined geometry and electrodes

height = 0;
radius = 1;
maxsz = 0.05;
n_elec = 16;
elec_rings = 1;
elec_width = 0.2;
elec_height = 0.3;
elec_maxsz = maxsz/2;
fmdl= ng_mk_ellip_models( ...
    [height, radius, radius, maxsz], ...
    [n_elec, elec_rings], ...
    [elec_width, elec_height, elec_maxsz] ...
);
figure();
show_fem(fmdl);


%% CREATING A CONDUCTIVITY DISTRIBUTION %%

% Get some info about the model
n_elements = size(fmdl.elems, 1);

% Get the centroid of each element
centroids = [(fmdl.nodes(fmdl.elems(:, 1), 1) + fmdl.nodes(fmdl.elems(:, 2), 1) + fmdl.nodes(fmdl.elems(:, 3), 1)) / 3, ...
             (fmdl.nodes(fmdl.elems(:, 1), 2) + fmdl.nodes(fmdl.elems(:, 2), 2) + fmdl.nodes(fmdl.elems(:, 3), 2)) / 3];

% Set the background conductivity to 1
condunctivity = ones(n_elements, 1);

% Set the conductivity of a circle to less than 1
target_radius = 0.3;
target_location = [0.1, 0.3];
target_cond = 1.4;
d = sqrt((centroids(:, 1) - target_location(1)).^2 + (centroids(:, 2) - target_location(2)).^2);
condunctivity(d < target_radius) = target_cond;

% Make an image of this conductivity
img = mk_image(fmdl, condunctivity);
figure()
show_fem(img)

%% SOLVING THE FORWARD PROBLEM %%

% Set values 
L  = 32; % electrodes/plane
c  = 1;  % meas coefficient (mA etc)
cA = -10000; % current coefficient (mA etc);

% Set stimulation patterns
inj_pat = '{ad}';
meas_pat = '{mono}';
stim_pat_options = {
    'meas_current', ... % Measure on injecting electrodes
    'balance_meas' ... % This grounds the voltage data
};
amplitude = 0.003; % Amp
stim = mk_stim_patterns(n_elec, 1, inj_pat, meas_pat, stim_pat_options, amplitude);
fmdl.stimulation = stim;

% Show fig of first pattern
figure()
plot(stim(1).stim_pattern, "*"); title(sprintf("Stim Pattern %d",1)); xlim([0.5,n_elec+0.5]);

% Set forward solver information for EIDORS
fmdl.solve      = @fwd_solve_1st_order;
fmdl.system_mat = @system_mat_1st_order;
fmdl.jacobian   = @jacobian_adjoint;

% Solve the forward problem
img = mk_image(fmdl, condunctivity); % Remake the image here with the updated fmdl
U = fwd_solve(img);
Umeas = reshape(U.meas, n_elec, []);
figure(); 
imagesc(Umeas); colorbar;


%% SOLVING THE INVERSE PROBLEM %%

% Set some options here
n_iterations = 10;

% Solve the forward problem for sigma = 1;
img1   = mk_image(fmdl, 1);
U1 = fwd_solve(img1);

% Ground this data also
U1meas = reshape(U1.meas, n_elec, []);

% Solve for the best fit
conductivity_hom = Umeas(:) \ U1meas(:);

% Initialize a place to store conductivity
sigma = zeros(n_iterations+1, n_elements);
sigma(1,:) = conductivity_hom;

% Start main loop
for it = 1:n_iterations
    
    % Solve the forward problem on current sigma
    img = mk_image(fmdl, sigma(it,:));
    Ui  = fwd_solve(img);
    Uimeas = Ui.meas;
    
    % Compute an update
    J = calc_jacobian(img);
    JtJ = J'*J;
    lambda_param = 0.02 * max(diag(JtJ));
    del_sigma_i = (J'*J + lambda_param * eye(size(JtJ,1))) \ (J' * (Uimeas(:) - Umeas(:)));
    
    % Make the update
    sigma(it+1, :) = sigma(it, :) - del_sigma_i(:)';

    % Make an image of the solution
    img = mk_image(fmdl, sigma(it+1, :));
    figure()
    show_fem(img)
    title(sprintf("Iteration %d", it))
    eidors_colourbar(img)

end