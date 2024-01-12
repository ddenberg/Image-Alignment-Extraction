function [mu_out, bm] = fit_gmm2_1D(X, tol_bm, tol_diff)

% TRY:
% tol_bm = 0.4;
% tol_diff = 0.1;

[bm, ~] = bimodality_coeff(X);

if bm > tol_bm

    % gmm1 = fitgmdist(X, 1, 'Options', statset('MaxIter', 500, 'Display', 'off'), 'Replicates', 5, 'Start', 'randSample');
    % NLL1 = gmm1.NegativeLogLikelihood;

    % initialize gmm with a guess "close" to the minimum
    mean_all = mean(X);
    filter_ind = X > mean_all;
    mu1_init = mean(X(filter_ind));
    mu2_init = mean(X(~filter_ind));
    var0_init = var(X(filter_ind));
    var1_init = var(X(~filter_ind));
    Sigma_init = zeros(1, 1, 2);
    Sigma_init(1,1,1) = var0_init;
    Sigma_init(1,1,2) = var1_init;
    init_struct = struct('mu', [mu1_init; mu2_init], 'Sigma', Sigma_init, 'ComponentProportion', [0.5, 0.5]);

    gmm2 = fitgmdist(X, 2, 'Options', statset('MaxIter', 5e3, 'Display', 'off', 'TolFun', 1e-8), 'Start', init_struct);
    % gmm2_rand = fitgmdist(X, 2, 'Options', statset('MaxIter', 500, 'Display', 'off'), 'Replicates', 50, 'Start', 'randSample');
    % NLL2 = gmm2.NegativeLogLikelihood;
    
    mu0 = gmm2.mu(1);
    mu1 = gmm2.mu(2);
    % sigma0 = sqrt(gmm2.Sigma(1,1,1));
    % sigma1 = sqrt(gmm2.Sigma(1,1,2));
    w0 = gmm2.ComponentProportion(1);
    w1 = gmm2.ComponentProportion(2);
    
    w_diff = abs(w0 - w1); 
    if w_diff > tol_diff
        if w0 > w1
            mu_out = mu0;
        else
            mu_out = mu1;
        end
    else
%         mu_out = max(mu0, mu1);
        mu_out = mean(X);
    end
else
    mu_out = mean(X);
end


% figure;
% histogram(X, 'Normalization', 'pdf', 'DisplayName', 'Data');
% hold on;
% 
% x_space = linspace(min(X), max(X), 1000).';
% plot(x_space, pdf(gmm2, x_space), 'LineWidth', 2, 'DisplayName', 'GMM Pdf');
% 
% prob1 = w0 * normpdf(x_space, mu0, sigma0);
% prob2 = w1 * normpdf(x_space, mu1, sigma1);
% plot(x_space, prob1, 'LineWidth', 2, 'DisplayName', 'Component 1');
% plot(x_space, prob2, 'LineWidth', 2, 'DisplayName', 'Component 2');
% 
% plot(x_space, normpdf(x_space, gmm1.mu, sqrt(gmm1.Sigma)), 'LineWidth', 2, 'DisplayName', 'Normal Dist');
% 
% title(['NLL1 = ', num2str(NLL1), ', NLL2 = ', num2str(NLL2)]);
% 
% legend('show');

end