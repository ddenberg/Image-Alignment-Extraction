function loss = rotation_loss(moving, fixed, params, translation, moving_z_center)

params = params(:);

% translation = params(1:3).' * length_scale;
eulerAngles = params(1:3).' * 180;
% translation = params(1:3).';
% eulerAngles = params(4:6).';

tform = rigidtform3d(eulerAngles, translation);

ref = imref3d(size(fixed));
ref.XWorldLimits = ref.XWorldLimits - mean(ref.XWorldLimits);
ref.YWorldLimits = ref.YWorldLimits - mean(ref.YWorldLimits);
ref.ZWorldLimits = ref.ZWorldLimits - moving_z_center;

moving_warp = imwarp(moving, ref, tform, 'OutputView', ref, 'interp', 'linear');

% loss = 1 - corr(moving_warp(:), fixed(:));
% loss = pdist2(moving_warp(:).', fixed(:).', 'minkowski', 5);

loss = rms(moving_warp - fixed, 'all');

end
