function val = objective_function(step_size, sigma, del_sigma, fmdl, Vmeas)
    % Define the objective function evaluation for a conductivity
    % distribution sigma+step_size*del_sigma on a model fmdl and with
    % measured voltage data Vmeas. This is used in a linesearch algorithm.

    % Solve the forward problem at sigma + step_size * del_sigma
    img = mk_image(fmdl, sigma + step_size * del_sigma);
    U   = fwd_solve(img);

    % Compute the value of the objective function
    val = 0.5 * sqrt(sum((U.meas(:) - Vmeas(:)) .^ 2));

end