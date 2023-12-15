clc;
clear;

addpath('loss_functions');
addpath('MA-ES');
addpath('PC_IoU');

long_path = 'D:/Posfai_Lab/MouseData/230101_st19_extract/long_nanog';
short_path = 'D:/Posfai_Lab/MouseData/230101_st19_extract/short_gata6';

% path_to_nuc_cam = '/scratch/gpfs/ddenberg/230320_st1/histone';
% path_to_long_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/long_nanog';
% path_to_short_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/short_gata6';
% path_to_nuc_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/histone';

output_path = './output/230101_st19/long_short_centers';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

long_short_tform = load('./output/230101_st19/align_long_short/tform_xy_average.mat');

frames_to_compute = [32,52,56,72,88,108];
numThreads = 6;

% centroid paramters
downsample_factor = 0.1;
outside_var_weight = 1e5;
max_zscore = 50;
min_percentile = 1;
max_percentile = 99;
min_radius = 150;
max_radius = 400;
population_size = 32;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[long_filenames, long_filename_folders] = get_filenames(long_path, {'klb'}, {});
[short_filenames, short_filename_folders] = get_filenames(short_path, {'klb'}, {});

% get each filename's corresponding frame number
long_frames = get_frame_ids(long_filenames);
short_frames = get_frame_ids(short_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:length(frames_to_compute)

    % get long and short filenames
    long_ind = find(long_frames == frames_to_compute(ii));
    short_ind = find(short_frames == frames_to_compute(ii));

    % skip if one of the images is not present
    if isempty(long_ind) || isempty(short_ind)
        continue;
    end

    long_file = fullfile(long_filename_folders{long_ind}, long_filenames{long_ind});
    short_file = fullfile(short_filename_folders{short_ind}, short_filenames{short_ind});

    % read in nuclear images
    long_img = readKLBstack(long_file, numThreads);
    short_img = readKLBstack(short_file, numThreads);

    % normalize images
    P_long = prctile(long_img, [min_percentile, max_percentile], 'all');
    P_short = prctile(short_img, [min_percentile, max_percentile], 'all');
    
    long_img = single(long_img);
    long_bg_mean = mean(long_img(long_img >= P_long(1) & long_img <= P_long(2)), 'all');
    long_bg_std = std(long_img(long_img >= P_long(1) & long_img <= P_long(2)), [], 'all');
    long_img = (long_img - long_bg_mean) / long_bg_std;
    long_img = min(long_img, max_zscore); % clip large z-scores

    short_img = single(short_img);
    short_bg_mean = mean(short_img(short_img >= P_short(1) & short_img <= P_short(2)), 'all');
    short_bg_std = std(short_img(short_img >= P_short(1) & short_img <= P_short(2)), [], 'all');
    short_img = (short_img - short_bg_mean) / short_bg_std;
    short_img = min(short_img, max_zscore); % clip large z-scores

    % warp short channel
    short_img = imwarp(short_img, long_short_tform.tform, 'OutputView', imref3d(size(long_img)));

    [img_centroid, MAES_state, trackers, debug_slice] = ...
        image_centroid(short_img + long_img, resXY, resZ, downsample_factor, ...
        outside_var_weight, min_radius, max_radius, population_size);

   
    save(fullfile(output_path, ['frame_', num2str(frames_to_compute(ii))]), ...
        'img_centroid', 'long_bg_mean', 'long_bg_std', 'short_bg_mean', ...
        'short_bg_std', 'MAES_state', 'trackers', 'debug_slice');
end
