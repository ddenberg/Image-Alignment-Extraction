function [DOF_out, centroids1_center, centroids2_center, E_min, MAES_state, trackers] = ...
    register_segmentation(seg1, seg2, resXY, resZ, downsample_factor, numTrials, weight)

% make segmentations isotropic
downsample_factor = max(min(downsample_factor, 1), 0);
seg1 = isotropicSample_nearest(seg1, resXY, resZ, downsample_factor);
seg2 = isotropicSample_nearest(seg2, resXY, resZ, downsample_factor);

% Compute the volumes of each cell to make an estimate of the nuclei radii
stats1 = regionprops3(seg1, 'Volume');
volumes1 = stats1.Volume;
nz_ind1 = find(volumes1 > 0);

stats2 = regionprops3(seg2, 'Volume');
volumes2 = stats2.Volume;
nz_ind2 = find(volumes2 > 0);

% compute centroids for each nucleus (seg1)
centroids1 = zeros(length(nz_ind1), 3);
for ii = 1:length(nz_ind1)
    ind = find(seg1 == nz_ind1(ii));
    [I,J,K] = ind2sub(size(seg1), ind);

    centroids1(ii,:) = [mean(I), mean(J), mean(K)];
end

% compute centroids for each nucleus (seg2)
centroids2 = zeros(length(nz_ind2), 3);
for ii = 1:length(nz_ind2)
    ind = find(seg2 == nz_ind2(ii));
    [I,J,K] = ind2sub(size(seg2), ind);

    centroids2(ii,:) = [mean(I), mean(J), mean(K)];
end

volumes1 = volumes1(nz_ind1);
volumes2 = volumes2(nz_ind2);
radii1 = (3 * volumes1 / (4 * pi)).^(1/3);
radii2 = (3 * volumes2 / (4 * pi)).^(1/3);

centroids1_center = sum(volumes1 .* centroids1, 1) / sum(volumes1);
centroids2_center = sum(volumes2 .* centroids2, 1) / sum(volumes2);

% temp swap xy
centroids1_swap = centroids1;
centroids1_swap(:,1) = centroids1(:,2);
centroids1_swap(:,2) = centroids1(:,1);

centroids2_swap = centroids2;
centroids2_swap(:,1) = centroids2(:,2);
centroids2_swap(:,2) = centroids2(:,1);

centroids1_center_swap = centroids1_center;
centroids1_center_swap(1) = centroids1_center(2);
centroids1_center_swap(2) = centroids1_center(1);

centroids2_center_swap = centroids2_center;
centroids2_center_swap(1) = centroids2_center(2);
centroids2_center_swap(2) = centroids2_center(1);

[DOF_min, E_min, MAES_state, trackers] = IoU_register(centroids1_swap, centroids2_swap, radii1, radii2, ...
    weight, numTrials, centroids1_center_swap, centroids2_center_swap);

theta_x = DOF_min(4);
theta_y = DOF_min(5);
theta_z = DOF_min(6);

rotation = [theta_x; theta_y; theta_z] / pi; % Rescale rotations between -1 and 1

% rescale by downsample_factor
centroids1_center = centroids1_center / downsample_factor;
centroids2_center = centroids2_center / downsample_factor;

translation = DOF_min(1:3) / downsample_factor;
% translation(2) = DOF_min(1) / downsample_factor;
% translation(1) = DOF_min(2) / downsample_factor;

DOF_out = [translation; rotation];

% % swap x and y
% temp = DOF_min(4);
% DOF_min(4) = DOF_min(5);
% DOF_min(5) = temp;

% temp = DOF_min(1);
% DOF_min(1) = DOF_min(2);
% DOF_min(2) = temp;

% rotation_mat = rotation_matrix(theta_x, theta_y, theta_z);
% rotation_mat_swap = rotation_mat;
% rotation_mat_swap(1,1) = rotation_mat(2,2);
% rotation_mat_swap(1,2) = rotation_mat(2,1);
% rotation_mat_swap(1,3) = rotation_mat(2,3);
% rotation_mat_swap(2,1) = rotation_mat(1,2);
% rotation_mat_swap(2,2) = rotation_mat(1,1);
% rotation_mat_swap(2,3) = rotation_mat(1,3);
% rotation_mat_swap(3,1) = rotation_mat(3,2);
% rotation_mat_swap(3,2) = rotation_mat(3,1);

% theta_x_swap = atan(rotation_mat_swap(3,2) / rotation_mat_swap(3,3));
% theta_y_swap = asin(-rotation_mat_swap(3,1));
% theta_z_swap = atan(rotation_mat_swap(2,1) / rotation_mat_swap(1,1));


% centroids1_transform = centroids1 - centroids1_center;
% centroids2_transform = (centroids2 - centroids2_center) * rotation.' + translation;

% cla;
% hold on;
% scatter3(centroids1_transform(:,1), centroids1_transform(:,2), centroids1_transform(:,3), '.');
% scatter3(centroids2_transform(:,1), centroids2_transform(:,2), centroids2_transform(:,3), '.');
% axis equal vis3d;

end

