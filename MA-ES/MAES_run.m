function [MAES_state, x_best, f_best, trackers] = MAES_run(MAES_state, obj_fun, write_output)

I = eye(MAES_state.n);

trackers.mean_loss_tracker = [];
trackers.sigma_tracker = [];
trackers.xdiff_rms_tracker = [];

f_best = inf;
x_best = MAES_state.x;

while MAES_state.t < MAES_state.max_gen && MAES_state.xdiff_rms > MAES_state.tolerance
% while MAES_state.t < MAES_state.max_gen

    %%% exit early

    % run individuals
    z = randn(MAES_state.n, MAES_state.lambda);
    d = MAES_state.M * z;

    f = zeros(MAES_state.lambda, 1);

    x = MAES_state.x;
    sigma = MAES_state.sigma;

    x_trial = x + sigma * d;
    for ii = 1:MAES_state.lambda
        f_ = obj_fun(x_trial(:,ii));
        f(ii) = f_;
        
%         f(ii) = MAES_state.f(MAES_state.x + MAES_state.sigma * d(:,ii));
    end

    [~, sorted_ind] = sort(f, 'ascend'); % minimiazation. ('descend') for maximization
    sorted_ind_mu = sorted_ind(1:MAES_state.mu);

    MAES_state.x = MAES_state.x + MAES_state.sigma * d(:,sorted_ind_mu) * MAES_state.w;

    MAES_state.xdiff_rms = rms(MAES_state.x - x); % add tracker for this

    mean_loss = sum(f(sorted_ind_mu) .* MAES_state.w);

    trackers.mean_loss_tracker(end+1) = mean_loss;
    trackers.sigma_tracker(end+1) = sigma;
    trackers.xdiff_rms_tracker(end+1) = MAES_state.xdiff_rms;

    if f(sorted_ind(1)) < f_best
        f_best = f(sorted_ind(1));
        x_best = x + sigma * d(:,sorted_ind(1));
    end

    if write_output
        fprintf('Step %d: loss = %e, best loss = %e, x_diff_rms = %e, sigma = %e\n', ...
            MAES_state.t, mean_loss, f_best, MAES_state.xdiff_rms, MAES_state.sigma);
    end

    MAES_state.p_sigma = (1 - MAES_state.c_sigma) * MAES_state.p_sigma + ...
        sqrt(MAES_state.mu_w * MAES_state.c_sigma * (2 - MAES_state.c_sigma)) * z(:,sorted_ind_mu) * MAES_state.w;

    MAES_state.M = MAES_state.M * (I + MAES_state.c_1 / 2 * (MAES_state.p_sigma * MAES_state.p_sigma.' - I) + ...
        MAES_state.c_mu / 2 * (z(:,sorted_ind_mu) * diag(MAES_state.w) * z(:,sorted_ind_mu).' - I));

    p_sigma_norm2 = sum(MAES_state.p_sigma.^2);
    MAES_state.sigma = MAES_state.sigma * exp(MAES_state.c_sigma / 2 * (p_sigma_norm2 / MAES_state.n - 1));

    MAES_state.t = MAES_state.t + 1;

    

    
end

end

