function [loss, moving_warp] = rigid_loss_noembedding(fixed, moving, params, length_scale, angle_scale, ...
    fixed_center, fixed_hpair, fixed_wpair, fixed_dpair, ...
    moving_center, moving_hpair, moving_wpair, moving_dpair)

params = params(:);

translation = params(1:3).' * length_scale;
eulerAngles = params(4:6).' * angle_scale;

tform = rigidtform3d(eulerAngles, translation);

moving_ref = imref3d(size(moving));
moving_ref.XWorldLimits = moving_hpair - moving_center(1);
moving_ref.YWorldLimits = moving_wpair - moving_center(2);
moving_ref.ZWorldLimits = moving_dpair - moving_center(3);
% moving_ref.ZWorldLimits = (moving_dpair - 1) * resZ / resXY + 1 - moving_center(3);

fixed_ref = imref3d(size(fixed));
fixed_ref.XWorldLimits = fixed_hpair - fixed_center(1);
fixed_ref.YWorldLimits = fixed_wpair - fixed_center(2);
fixed_ref.ZWorldLimits = fixed_dpair - fixed_center(3);
% fixed_ref.ZWorldLimits = (fixed_dpair - 1) * resZ / resXY + 1 - fixed_center(3);


moving_warp = imwarp(moving, moving_ref, tform, 'OutputView', fixed_ref, 'interp', 'linear');

loss = rms(moving_warp - fixed, 'all');
% loss = -sum(moving_warp .* fixed, 'all') / (numel(fixed) - 1);



end
