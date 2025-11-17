function compute_image_centroids_maxproj(maxproj_file, image_path, output_path, ...
    frames_to_compute, numThreads, resXY, resZ, image_format, ...
    overwrite_centroid, overwrite_centroid_tolerance)
% maxproj_file: /path/to/filename.h5 generated from max_projection_across_time.m
% image_path: path to image directory containing images from a single channel/camera
% output_path: where to save the output centers
% frames_to_compute: [start_frame:end_frame]
% numThreads: 8 or 16 work well
% varargin{1}: overwrite_centroid: Should only be specified if you want to overwrite the centroid
%                                  pass as [c_x, c_y, c_z] where c_i are the pixel coordinates
% varargin{2}: overwrite_centroid_tolerance: Amount of pixels to deviate from 'overwrite_centroid"

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% centroid paramters
downsample_factor = 0.1;
outside_var_weight = 1e5;
max_zscore = 100;
min_percentile = 1;
max_percentile = 99;
min_radius = 150;
max_radius = 400; % 400
population_size = 16;

% anisotropy parameters
if isempty(resXY)
    resXY = 0.208;
end
if isempty(resZ)
    resZ = 2.0;
end

image_ext = 'klb';
read_klb = true;
if strcmpi(image_format, 'klb')
    image_ext = 'klb';
elseif strcmpi(image_format, 'tif')
    image_ext = 'tif';
    read_klb = false;
elseif strcmpi(image_format, 'tiff')
    image_ext = 'tiff';
    read_klb = false;
end

% read in max projection
img = h5read(maxproj_file, '/maxproj');

% normalize max projection
P = prctile(img, [min_percentile, max_percentile], 'all');

img = single(img);
img_bg_mean = mean(img(img >= P(1) & img <= P(2)), 'all');
img_bg_std = std(img(img >= P(1) & img <= P(2)), [], 'all');
img = (img - img_bg_mean) / img_bg_std;
img = max(min(img, max_zscore), 0); % clip large z-scores and clip at 0

if ~isempty(overwrite_centroid)
    overwrite_centroid = overwrite_centroid(:);
    if ~all(size(overwrite_centroid) == [3, 1])
        error('overwrite_centroid does not have the correct shape. Must be [3, 1]');
    end
end

[img_centroid, MAES_state, trackers, debug_stack] = ...
    image_centroid(img, resXY, resZ, downsample_factor, ...
    outside_var_weight, min_radius, max_radius, population_size, ...
    overwrite_centroid, overwrite_centroid_tolerance);

save(fullfile(output_path, 'maxproj'), 'img_centroid', 'MAES_state', 'trackers', ...
    'debug_stack');

% get filenames in each directory (excluding .label and .tif images)
[img_filenames, img_filename_folders] = get_filenames(image_path, {image_ext}, {'._'});

% get each filename's corresponding frame number
img_frames = get_frame_ids(img_filenames);

% loop through each pair of frames to align 
for ii = 1:length(frames_to_compute)

    % get nuclear, long, and short filenames
    img_ind = find(img_frames == frames_to_compute(ii));

    % skip if one of the images is not present
    if isempty(img_ind)
        continue;
    end

    img_file = fullfile(img_filename_folders{img_ind}, img_filenames{img_ind});

    % read in nuclear images
    if read_klb
        img = readKLBstack(img_file, numThreads);
    else
        img = tiffreadVolume(img_file);
        img = permute(img, [2, 1, 3]);
    end
    
    % normalize images
    P = prctile(img, [min_percentile, max_percentile], 'all');
    
    img = single(img);
    img_bg_mean = mean(img(img >= P(1) & img <= P(2)), 'all');
    img_bg_std = std(img(img >= P(1) & img <= P(2)), [], 'all');
    % img = (img - img_bg_mean) / img_bg_std;
    % img = max(min(img, max_zscore), 0); % clip large z-scores and clip at 0

    save(fullfile(output_path, ['frame_', num2str(frames_to_compute(ii))]), ...
        'img_centroid', 'img_bg_mean', 'img_bg_std');


    fprintf('Frame %d, Done!\n', frames_to_compute(ii));
end

end
