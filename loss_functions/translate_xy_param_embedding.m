function translation = translate_xy_param_embedding(params, length_scale)

params = params(:);

translation = [sin(params(1:2).') * length_scale, 0];
end