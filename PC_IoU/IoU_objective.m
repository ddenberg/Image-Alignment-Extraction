function E = IoU_objective(DOF, centroids1, centroids2, R1, R2, weight, centroids1_center, centroids2_center)

theta_x = DOF(4);
theta_y = DOF(5);
theta_z = DOF(6);
rotation = rotation_matrix(theta_x, theta_y, theta_z);
translation = reshape(DOF(1:3), 1, 3);

centroids1_transform = centroids1 - centroids1_center;
centroids2_transform = (centroids2 - centroids2_center) * rotation.' + translation;

N = size(centroids1, 1);
M = size(centroids2, 1);

P = sphere_IoU(centroids1_transform, centroids2_transform, R1, R2);
p = (1 - weight) * sum(P, 2) / M + weight / N;
E = -sum(log(p)) / N;

end
