function [loss, moving_warp] = rigid_loss(fixed, moving, params, length_scale, angle_scale, ...
    fixed_center, fixed_hpair, fixed_vpair, fixed_zpair, moving_center, moving_hpair, moving_vpair, moving_zpair, resXY, resZ)

params = params(:);

% translation = params(1:3).' * length_scale;
% eulerAngles = params(4:6).' * 180;
% translation = params(1:3).';
% eulerAngles = params(4:6).';
[eulerAngles, translation] = param_embedding(params, length_scale, angle_scale);

tform = rigidtform3d(eulerAngles, translation);

moving_ref = imref3d(size(moving));
moving_ref.XWorldLimits = moving_hpair - moving_center(1);
moving_ref.YWorldLimits = moving_vpair - moving_center(2);
moving_ref.ZWorldLimits = (moving_zpair - 1) * resZ / resXY + 1 - moving_center(3);

% moving_ref.XWorldLimits = moving_ref.XWorldLimits - mean(moving_ref.XWorldLimits);
% moving_ref.YWorldLimits = moving_ref.YWorldLimits - mean(moving_ref.YWorldLimits);
% moving_ref.ZWorldLimits = moving_ref.ZWorldLimits - mean(moving_ref.ZWorldLimits);

fixed_ref = imref3d(size(fixed));
% fixed_ref.XWorldLimits = fixed_ref.XWorldLimits - mean(fixed_ref.XWorldLimits);
% fixed_ref.YWorldLimits = fixed_ref.YWorldLimits - mean(fixed_ref.YWorldLimits);
% fixed_ref.ZWorldLimits = fixed_ref.ZWorldLimits - mean(fixed_ref.ZWorldLimits);
fixed_ref.XWorldLimits = fixed_hpair - fixed_center(1);
fixed_ref.YWorldLimits = fixed_vpair - fixed_center(2);
fixed_ref.ZWorldLimits = (fixed_zpair - 1) * resZ / resXY + 1 - fixed_center(3);


moving_warp = imwarp(moving, moving_ref, tform, 'OutputView', fixed_ref, 'interp', 'linear');

% loss = 1 - corr(moving_warp(:), fixed(:));
% loss = pdist2(moving_warp(:).', fixed(:).', 'minkowski', 5);

loss = rms(moving_warp - fixed, 'all');

end


function y = sigmoid(x)
y = 1 ./ (1 + exp(-x));
end
