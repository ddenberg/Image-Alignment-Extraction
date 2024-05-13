function [img_crop, hpair, wpair, dpair] = isotropic_crop(img, center, height, width, depth, resXY, resZ, interp)

% Calculate the size of the input image
[imgHeight, imgWidth, ~] = size(img);

center = round(center);

% Calculate start and end indices for cropping (xy only)
startRow = max(center(1) - floor(height / 2), 1);
endRow = min(center(1) + floor(height / 2), imgHeight);
startCol = max(center(2) - floor(width / 2), 1);
endCol = min(center(2) + floor(width / 2), imgWidth);

% Crop the image (xy only)
img_crop = img(startRow:endRow, startCol:endCol, :);

% Calculate padding sizes (xy only)
padBeforeRow = max(floor(height / 2) - center(1) + 1, 0);
padAfterRow = max(center(1) + floor(height / 2) - imgHeight, 0);
padBeforeCol = max(floor(width / 2) - center(2) + 1, 0);
padAfterCol = max(center(2) + floor(width / 2) - imgWidth, 0);

hpair = [startRow - padBeforeRow, endRow + padAfterRow];
wpair = [startCol - padBeforeCol, endCol + padAfterCol];

% Apply padding
img_crop = padarray(img_crop, [padBeforeRow, padBeforeCol, 0], 0, 'pre');
img_crop = padarray(img_crop, [padAfterRow, padAfterCol, 0], 0, 'post');

% transform to isotropic resolution
if strcmp(interp, 'bilinear')
    img_crop = isotropicSample_bilinear(img_crop, resXY, resZ, 1);
elseif strcmp(interp, 'nearest')
    img_crop = isotropicSample_nearest(img_crop, resXY, resZ, 1);
else
    error("interp must be one of 'bilinear' or 'nearest'.");
end

[~, ~, imgDepth] = size(img_crop);

% Calculate start and end indices for cropping (z only)
startDepth = max(center(3) - floor(depth / 2), 1);
endDepth = min(center(3) + floor(depth / 2), imgDepth);

% Crop the image (z only)
img_crop = img_crop(:, :, startDepth:endDepth);

% Calculate padding sizes (z only)
padBeforeDepth = max(floor(depth / 2) - center(3) + 1, 0);
padAfterDepth = max(center(3) + floor(depth / 2) - imgDepth, 0);

dpair = [startDepth - padBeforeDepth, endDepth + padAfterDepth];

% apply padding
img_crop = padarray(img_crop, [0, 0, padBeforeDepth], 0, 'pre');
img_crop = padarray(img_crop, [0, 0, padAfterDepth], 0, 'post');

end

