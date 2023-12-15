function [eulerAngles, translation] = param_embedding(params, length_scale, angle_scale)

params = params(:);

translation = sin(params(1:3).') * length_scale;
eulerAngles = sin(params(4:6).') * angle_scale;
% default angle_scale should be 180;
end