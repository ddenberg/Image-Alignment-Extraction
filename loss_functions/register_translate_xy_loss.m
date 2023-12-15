function loss = register_translate_xy_rms(moving, fixed, params)

% params = [x_offset; y_offset]

translation = [params(1:2).', 0];
tform = transltform3d(translation);

moving_warp = imwarp(moving, tform, 'OutputView', imref3d(size(fixed)));

loss = rms(moving_warp - fixed, 'all');

end

