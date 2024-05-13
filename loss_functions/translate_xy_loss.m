function [loss, moving_warp] = translate_xy_loss(fixed, moving, params, length_scale)

translation = translate_xy_param_embedding(params, length_scale);

tform = transltform3d(translation);

moving_warp = imwarp(moving, tform, 'OutputView', imref3d(size(fixed)), 'interp', 'linear');

loss = rms(moving_warp - fixed, 'all');

end

