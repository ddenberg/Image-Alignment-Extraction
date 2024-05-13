function [img_crop, hpair, wpair] = xy_crop(img, center, height, width)

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

end

