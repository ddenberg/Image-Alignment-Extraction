function MAES_state = MAES_initialize(x_init, sigma, max_gen, tolerance, population_size)

x_init = x_init(:); %make sure input is a column vector

MAES_state.n = length(x_init); % dimensionality
% MAES_state.f = obj_function_handle;
MAES_state.max_gen = max_gen;
MAES_state.tolerance = tolerance;
MAES_state.xdiff_rms = Inf;

MAES_state.sigma = sigma;
MAES_state.t = 0;
MAES_state.x = x_init;
MAES_state.p_sigma = zeros(MAES_state.n, 1);
MAES_state.M = eye(MAES_state.n);

% population size
if ~isempty(population_size)
    MAES_state.lambda = population_size;
else
    MAES_state.lambda = 4 + floor(3 * log(MAES_state.n));
end
% number of recombinations
MAES_state.mu = floor(MAES_state.lambda / 2);

% weights
MAES_state.w = log(MAES_state.mu + 0.5) - log(1:MAES_state.mu).';
MAES_state.w = MAES_state.w / sum(MAES_state.w);

MAES_state.mu_w = 1 / sum(MAES_state.w.^2);
MAES_state.c_sigma = (MAES_state.mu_w + 2) / (MAES_state.n + MAES_state.mu_w + 5);
MAES_state.c_1 = 2 / ((MAES_state.n + 1.3)^2 + MAES_state.mu_w);
MAES_state.c_mu = min(1 - MAES_state.c_1, 2 * (MAES_state.mu_w - 2 + 1 / MAES_state.mu_w) / ...
                                            ((MAES_state.n + 2)^2  + MAES_state.mu_w));


end

