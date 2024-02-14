function del_sigmas = update_function(sigmas, filename)
    % Sigmas is an n by m array containing conductivity distributions for n
    % samples defined over m elements. filename contains the rest of the
    % data needed to compute updates (the model and measurement data)
    run 'C:\Users\Billy\OneDrive - Marquette University\EIDORS\eidors-v3.10-ng\eidors\startup';

    % Load the extra data
    load(filename, "VSIM", "rec_fmdl");

    % Initialize the output
    del_sigmas = zeros(size(sigmas));

    % Loop through the samples to compute updates
    for i = 1:size(sigmas, 1)
        fprintf("Computing update for sample %d\n", i);
        
        % Solve the forward problem
        img = mk_image(rec_fmdl, sigmas(i, :));
        U   = fwd_solve(img);

        % Compute an LM Update
        J = calc_jacobian(img);
        JtJ = J'*J;
        lambda_param = 0.05 * max(diag(JtJ));
        del_sigma = - (J'*J + lambda_param * eye(size(JtJ,1))) \ (J' * (U.meas(:) - VSIM(i, :)'));
        del_sigmas(i, :) = del_sigma;

    end
end