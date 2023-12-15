function [DOF_min, E_min, MAES_state, trackers] = IoU_register(centroids1, centroids2, R1, R2, weight, numTrials, centroids1_center, centroids2_center)

% probably bad guess
DOF0 = zeros(6, 1);

obj_handle = @(DOF) IoU_objective(DOF, centroids1, centroids2, R1, R2, weight, centroids1_center, centroids2_center);

% get a good initial guess by randomly sampling rotations
DOF0_best = DOF0;
E0_best = obj_handle(DOF0);
for ii = 1:numTrials
    theta_x = pi * (2 * rand - 1);
    theta_y = pi * (2 * rand - 1);
    theta_z = pi * (2 * rand - 1);
    DOF0 = [zeros(3, 1); theta_x; theta_y; theta_z];
    E0 = obj_handle(DOF0);

    if E0 < E0_best
        E0_best = E0;
        DOF0_best = DOF0;
    end
end

% refine best guess
MAES_state = MAES_initialize(DOF0_best, 1e-1, 200, 1e-5, 32);
[MAES_state, DOF_min, E_min, trackers] = MAES_run(MAES_state, obj_handle, true);

% options = optimoptions('fminunc', 'Display', 'none');
% [DOF_min, E_min] = fminunc(obj_handle, DOF0_best, options);

end

