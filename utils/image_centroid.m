function [centroid, MAES_state, trackers, debug_stack] = image_centroid(img, resXY, resZ, ...
    downsample_factor, outside_var_weight, min_radius, max_radius, population_size, ...
    overwrite_centroid, overwrite_centroid_tolerance)

img = isotropicSample_bilinear(img, resXY, resZ, downsample_factor);

[X, Y, Z] = ndgrid(1:size(img, 1), 1:size(img, 2), 1:size(img, 3));

DOF0 = zeros(4, 1);

% min_radius = 15;
% max_radius = mean(size(img)) / 4;
min_radius = min_radius * downsample_factor;
max_radius = max_radius * downsample_factor;

if ~isempty(overwrite_centroid)
    overwrite_centroid(3) = (overwrite_centroid(3) - 1) * resZ / resXY + 1;
    
    min_center = overwrite_centroid - overwrite_centroid_tolerance;
    max_center = overwrite_centroid + overwrite_centroid_tolerance;

    min_center = min_center * downsample_factor;
    max_center = max_center * downsample_factor;

else    
    min_center = 2 * min_radius * ones(3, 1);
    max_center = size(img);
    max_center = max_center(:) - min_center;
end

% min_center(1:2) = [75; 45]; % vertical min; horizontal min
% max_center(1:2) = [100; 80]; % vertical max; horizontal max

obj_fun_handle = @(DOF) ChanVese_objective(DOF, img, X, Y, Z, outside_var_weight, ...
    min_center, max_center, min_radius, max_radius);

MAES_state = MAES_initialize(DOF0, 1e0, 200, 1e-5, population_size);
[MAES_state, DOF_min, ~, trackers] = MAES_run(MAES_state, obj_fun_handle, true);

[centroid, radius] = DOF_map(DOF_min, min_center, max_center, min_radius, max_radius);

% make debug slice through center of centroid
mask = (X - centroid(1)).^2 + (Y - centroid(2)).^2 + (Z - centroid(3)).^2 <= radius.^2;
% z_ind = round(centroid(3));
BW = edge3(mask, 'sobel', 0.5);
debug_stack = img + max(img, [], 'all') * BW;

% rescale centroid to full resolution
centroid = centroid / downsample_factor;

end

function F = ChanVese_objective(DOF, img_iso, X, Y, Z, outside_var_weight, min_center, max_center, min_radius, max_radius)
[center, radius] = DOF_map(DOF, min_center, max_center, min_radius, max_radius);

mask = (X - center(1)).^2 + (Y - center(2)).^2 + (Z - center(3)).^2 <= radius.^2;

outside_var = var(img_iso(~mask));

F = radius.^2 + outside_var_weight * outside_var;
end

function [center, radius] = DOF_map(DOF, min_center, max_center, min_radius, max_radius)
% center = sigmoid(DOF(1:3)) .* (max_center - min_center) + min_center;
% radius = sigmoid(DOF(4)) .* (max_radius - min_radius) + min_radius;
center = actfun(DOF(1:3)) .* (max_center - min_center) + min_center;
radius = actfun(DOF(4)) .* (max_radius - min_radius) + min_radius;
end

function y = actfun(x)
y = 0.5 * (1 + sin(x));
end

function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end

