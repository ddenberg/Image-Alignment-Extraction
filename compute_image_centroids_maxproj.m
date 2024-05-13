% clc;
% clear;
function compute_image_centroids_maxproj(maxproj_file, image_path, output_path, ...
    frames_to_compute, numThreads, varargin)
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


% maxproj_file = 'D:\Posfai_Lab\MouseData\230917_st10\histone_maxproj_permute.tif';
% image_path = '/scratch/gpfs/ddenberg/230521/st8/histone';

% output_path = './output/230917_st10/histone_centers';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% frames_to_compute = 0:140;
% frames_to_compute = first_frame:last_frame; %,118,121,123,125,126,127,128,129,130];
% numThreads = 6;

% centroid paramters
downsample_factor = 0.1;
outside_var_weight = 1e4;
max_zscore = 25;
min_percentile = 1;
max_percentile = 99;
min_radius = 150;
max_radius = 400;
population_size = 16;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% read in nuclear images
% img = readKLBstack(maxproj_file, numThreads);
% img = tiffreadVolume(maxproj_file);
img = h5read(maxproj_file, '/maxproj');

% normalize images
P = prctile(img, [min_percentile, max_percentile], 'all');

img = single(img);
img_bg_mean = mean(img(img >= P(1) & img <= P(2)), 'all');
img_bg_std = std(img(img >= P(1) & img <= P(2)), [], 'all');
img = (img - img_bg_mean) / img_bg_std;
img = max(min(img, max_zscore), 0); % clip large z-scores and clip at 0

overwrite_centroid = [];
overwrite_centroid_tolerance = [];
if ~isempty(varargin)
    overwrite_centroid = varargin{1};
    overwrite_centroid = overwrite_centroid(:);
    if ~all(size(overwrite_centroid) == [3, 1])
        error('overwrite_centroid does not have the correct shape. Must be [3, 1]');
    end

    if length(varargin) == 2
        overwrite_centroid_tolerance = varargin{2};
    end
end

[img_centroid, MAES_state, trackers, debug_stack] = ...
    image_centroid(img, resXY, resZ, downsample_factor, ...
    outside_var_weight, min_radius, max_radius, population_size, ...
    overwrite_centroid, overwrite_centroid_tolerance);

save(fullfile(output_path, 'maxproj'), 'img_centroid', 'MAES_state', 'trackers', ...
    'debug_stack');

% get filenames in each directory (excluding .label and .tif images)
[img_filenames, img_filename_folders] = get_filenames(image_path, {'klb'}, {});

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
    img = readKLBstack(img_file, numThreads);
    
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
