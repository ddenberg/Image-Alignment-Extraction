clc;
clear;

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% path_to_long_cam = 'D:/Posfai_Lab/MATLAB/Mouse_G6N/Stack19_WT/transcripton_factors/long';
% path_to_short_cam = 'D:/Posfai_Lab/MATLAB/Mouse_G6N/Stack19_WT/transcripton_factors/short';
path_to_long_cam = '/scratch/gpfs/ddenberg/230212_st6_extract/Ch0long_nanog';
path_to_short_cam = '/scratch/gpfs/ddenberg/230212_st6_extract/Ch0short_gata6';

output_path = './output/230212_st6/align_long_short';

frames_to_align = 0:10:130;
numThreads = 16;

% crop box for increasing performance
% Stack 19
% hpair = [650, 650+870];
% vpair = [150, 150+830];
% zpair = [8, 72];

% 230212_st6
hpair = [630, 630+820];
vpair = [360, 360+780];
zpair = [8, 70];

% downsample factor
downsample_factor = 0.25;

% get filenames in each directory (excluding .label and .tif images)
[long_cam_filenames, long_cam_folders] = get_filenames(path_to_long_cam, {'Long', 'klb'}, {'label', 'tif'});
[short_cam_filenames, short_cam_folders] = get_filenames(path_to_short_cam, {'Short', 'klb'}, {'label', 'tif'});

% get each filename's corresponding frame number
long_cam_frames = get_frame_ids(long_cam_filenames);
short_cam_frames = get_frame_ids(short_cam_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)

for ii = 1:length(frames_to_align)

    % get long and short inds
    long_ind = find(long_cam_frames == frames_to_align(ii));
    short_ind = find(short_cam_frames == frames_to_align(ii));

    % skip if one of the images is not present
    if isempty(short_ind) || isempty(long_ind)
        continue;
    end

    % get long and short filenames
    long_file = fullfile(long_cam_folders{long_ind}, long_cam_filenames{long_ind});
    short_file = fullfile(short_cam_folders{short_ind}, short_cam_filenames{short_ind});

    % read in long and short images
    long_raw = readKLBstack(long_file, numThreads);
    short_raw = readKLBstack(short_file, numThreads);

    % crop long and short images
    long_crop = long_raw(hpair(1):hpair(2), vpair(1):vpair(2), zpair(1):zpair(2));
    short_crop = short_raw(hpair(1):hpair(2), vpair(1):vpair(2), zpair(1):zpair(2));
    long_crop = single(long_crop);
    short_crop = single(short_crop);

    % get z-scores for long and short images
    long_crop = zscore(long_crop, 0, 'all');
    short_crop = zscore(short_crop, 0, 'all');

    % for long-short alignment downsample only in x-y
    % this changes the anisotropy factor but we can do this because we are only doing x-y translation registration
    long_crop_ds = imresize(long_crop, downsample_factor, "bilinear");
    short_crop_ds = imresize(short_crop, downsample_factor, "bilinear");

    % do down sampled registration first
    % initialize parameters for optimization
    x_init = [0, 0];% 0 x-y offset initial guess
    sigma = 5;
    obj_function_handle = @(x) register_translate_xy_loss(short_crop_ds, long_crop_ds, x);
    max_gen = 100;
    tol = 1e-5;
    population_size = 32;
    MAES_state_ds = MAES_initialize(x_init, sigma, max_gen, tol, population_size);
    
    tic;
    [MAES_state_ds, x_min_ds, trackers_ds] = MAES_run(MAES_state_ds, obj_function_handle, true);
    toc;

    % do full resolution registration next
    % initialize parameters for optimization
    x_init = x_min_ds / downsample_factor; % use downsampled initial guess
    sigma = 3e-1;
    obj_function_handle = @(x) register_translate_xy_loss(short_crop, long_crop, x);
    max_gen = 40;
    tol = 1e-5;
    population_size = 8;
    MAES_state = MAES_initialize(x_init, sigma, max_gen, tol, population_size);
    
    tic;
    [MAES_state, x_min, trackers] = MAES_run(MAES_state, obj_function_handle, true);
    toc;

    translation = [x_min(1:2).', 0];
    tform = transltform3d(translation);

    save(fullfile(output_path, ['tform_xy_frame_', num2str(frames_to_align(ii))]), ...
        'tform', 'x_min', 'MAES_state_ds', 'trackers_ds', 'MAES_state', 'trackers');
end
