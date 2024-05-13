% clc;
% clear;
function align_histone_fusedLS(path_to_long, path_to_long_centers, path_to_short, ...
    path_to_short_centers, path_to_histone, path_to_histone_centers, path_to_long_short_align, ...
    output_path, frames_to_align, numThreads)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% path_to_long = 'D:\Posfai_Lab\MouseData\230101_st19_extract\long_nanog';
% path_to_short = 'D:\Posfai_Lab\MouseData\230101_st19_extract\short_gata6';
% path_to_histone = 'D:\Posfai_Lab\MouseData\230101_st19_extract\histone';

% path_to_long_short_centers = './output/230101_st19/long_short_centers';
% path_to_histone_centers = './output/230101_st19/histone_centers';

% path_to_long = '/scratch/gpfs/ddenberg/230101_st19_extract/long_nanog';
% path_to_short = '/scratch/gpfs/ddenberg/230101_st19_extract/short_gata6';
% path_to_histone = '/scratch/gpfs/ddenberg/230101_st19_extract/histone';

% output_path = './output/230101_st19/align_LS_histone';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

long_short_align_struct = load(path_to_long_short_align);
% long_short_tform = load('./output/230101_st19/align_long_short/tform_xy_average.mat');

% frames_to_align = 100;
% numThreads = 6;

% crop box for increasing performance
crop_height = 900;
crop_width = 900;
crop_depth = 900;

% cap maximum z score
max_zscore = 100;

% downsample factor (list of values for registration steps)
downsample_factor = [0.25];
sigma_init = [1e-1];
max_gen_init = [200];
population_size_init = [10];
tol = 1e-5;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[histone_filenames, histone_filename_folders] = get_filenames(path_to_histone, {'klb'}, {});
[long_filenames, long_filename_folders] = get_filenames(path_to_long, {'klb'}, {});
[short_filenames, short_filename_folders] = get_filenames(path_to_short, {'klb'}, {});
[histone_centers_filenames, histone_centers_filename_folders] = ...
    get_filenames(path_to_histone_centers, {'mat'}, {});
[long_centers_filenames, long_centers_filename_folders] = get_filenames(path_to_long_centers, {'mat'}, {});
[short_centers_filenames, short_centers_filename_folders] = get_filenames(path_to_short_centers, {'mat'}, {});

% get each filename's corresponding frame number
histone_frames = get_frame_ids(histone_filenames);
long_frames = get_frame_ids(long_filenames);
short_frames = get_frame_ids(short_filenames);
histone_centers_frames = get_frame_ids(histone_centers_filenames);
long_centers_frames = get_frame_ids(long_centers_filenames);
short_centers_frames = get_frame_ids(short_centers_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:length(frames_to_align)

    % get nuclear, long, and short filenames
    histone_ind = find(histone_frames == frames_to_align(ii));
    long_ind = find(long_frames == frames_to_align(ii));
    short_ind = find(short_frames == frames_to_align(ii));
    histone_center_ind = find(histone_centers_frames == frames_to_align(ii));
    long_center_ind = find(long_centers_frames == frames_to_align(ii));
    short_center_ind = find(short_centers_frames == frames_to_align(ii));

    % skip if one of the images is not present
    if isempty(short_ind) || isempty(long_ind) || isempty(histone_ind) || ...
            isempty(histone_center_ind) || isempty(long_center_ind) || isempty(short_center_ind)
        continue;
    end

    histone_file = fullfile(histone_filename_folders{histone_ind}, histone_filenames{histone_ind});
    long_file = fullfile(long_filename_folders{long_ind}, long_filenames{long_ind});
    short_file = fullfile(short_filename_folders{short_ind}, short_filenames{short_ind});
    histone_center_file = fullfile(histone_centers_filename_folders{histone_center_ind}, ...
                                   histone_centers_filenames{histone_center_ind});
    long_center_file = fullfile(long_centers_filename_folders{long_center_ind}, ...
                                long_centers_filenames{long_center_ind});
    short_center_file = fullfile(short_centers_filename_folders{short_center_ind}, ...
                                 short_centers_filenames{short_center_ind});

    %% read histone
    histone_img = readKLBstack(histone_file, numThreads);
    histone_center_struct = load(histone_center_file);
    histone_centroid = histone_center_struct.img_centroid;

    fprintf('Frame %d\n', frames_to_align(ii));

    % normalize image
    histone_img = (single(histone_img) - histone_center_struct.img_bg_mean) / histone_center_struct.img_bg_std;

    % crop histone
    [histone_crop, histone_hpair, histone_wpair, histone_dpair] = isotropic_crop(histone_img, histone_centroid, ...
        crop_height, crop_width, crop_depth, resXY, resZ, 'bilinear');

    histone_crop = min(histone_crop, max_zscore);

    fprintf('  Histone Centroid %f, %f, %f\n', histone_centroid(1), histone_centroid(2), histone_centroid(3));
    fprintf('  Histone crop: %d, %d, %d\n', size(histone_crop, 1), size(histone_crop, 2), size(histone_crop, 3));

    clear histone_img;

    %% load long and short
    long_img = readKLBstack(long_file, numThreads);
    short_img = readKLBstack(short_file, numThreads);

    % load long center
    long_center_struct = load(long_center_file);
    long_centroid = long_center_struct.img_centroid;

    % load short center
    short_center_struct = load(short_center_file);

    % normalize images
    long_img = (single(long_img) - long_center_struct.img_bg_mean) / long_center_struct.img_bg_std;
    short_img = (single(short_img) - short_center_struct.img_bg_mean) / short_center_struct.img_bg_std;

    % warp short channel
    short_img = imwarp(short_img, long_short_align_struct.translation_tform, ...
        'OutputView', imref3d(size(long_img)), 'interp', 'linear');

    % create crops
    [long_crop, long_hpair, long_wpair, long_dpair] = isotropic_crop(long_img, long_centroid, ...
        crop_height, crop_width, crop_depth, resXY, resZ, 'bilinear');

    short_crop = isotropic_crop(short_img, long_centroid, ...
        crop_height, crop_width, crop_depth, resXY, resZ, 'bilinear');

    LS_crop = long_crop + short_crop;
    LS_crop = min(LS_crop, max_zscore);

    fprintf('  LS centroid: %f, %f, %f\n', long_centroid(1), long_centroid(2), long_centroid(3));
    fprintf('  LS crop: %d, %d, %d\n', size(LS_crop, 1), size(LS_crop, 2), size(LS_crop, 3));

    clear short_img long_img short_crop long_crop

    % loop over downsample stages and use previous registration as initialization for the next
    rigid_trackers_ds = cell(length(downsample_factor), 1);
    rigid_MAES_state_ds = cell(length(downsample_factor), 1);
    rigid_x_min_ds = cell(length(downsample_factor), 1);
    length_scale = 80;
    angle_scale = 180;
    rigid_tform_ds = cell(length(downsample_factor), 1);
    for jj = 1:length(downsample_factor)
        LS_ds = imresize3(LS_crop, downsample_factor(jj), 'linear');
        histone_ds = imresize3(histone_crop, downsample_factor(jj), 'linear');

        sigma = sigma_init(jj);
        max_gen = max_gen_init(jj);
        population_size = population_size_init(jj);

        rigid_fun_h = @(x) rigid_loss(histone_ds, LS_ds, x, length_scale, angle_scale, ...
                                      histone_centroid, histone_hpair, histone_wpair, histone_dpair, ...
                                      long_centroid, long_hpair, long_wpair, long_dpair);

        % do down sampled registration first
        % initialize parameters for optimization
        if jj == 1
            x_init = zeros(6, 1);% 0 x-y offset initial guess
        else
            x_init = x_min_ds;
        end

        MAES_state_ds = MAES_initialize(x_init, sigma, max_gen, tol, population_size);
        
        tic;
        [MAES_state_ds, x_min_ds, ~, trackers_ds] = MAES_run(MAES_state_ds, rigid_fun_h, false);
        toc;

        rigid_trackers_ds{jj} = trackers_ds;
        rigid_MAES_state_ds{jj} = MAES_state_ds;
        rigid_x_min_ds{jj} = x_min_ds;
        [eulerAngles, translation] = rigid_param_embedding(x_min_ds, length_scale, angle_scale);
        rigid_tform_ds{jj} = rigidtform3d(eulerAngles, translation);

        if jj == 1
            % create debug registered images
            [~, LS_ds_warp] = rigid_fun_h(x_min_ds);

            debug_stack = zeros([size(LS_ds_warp), 2], 'uint8');

            histone_ds_prctile = prctile(histone_ds, [0.1, 99.9], 'all');
            debug_stack(:,:,:,1) = uint8(rescale(histone_ds, 0, 2^8-1, ...
                'InputMin', histone_ds_prctile(1), 'InputMax', histone_ds_prctile(2)));

            LS_ds_warp_prctile = prctile(LS_ds_warp, [0.1, 99.9], 'all');
            debug_stack(:,:,:,2) = uint8(rescale(LS_ds_warp, 0, 2^8-1, ...
                'InputMin', LS_ds_warp_prctile(1), 'InputMax', LS_ds_warp_prctile(2)));

            debug_maxproj = zeros(size(LS_ds_warp, 1), size(LS_ds_warp, 2), 2, 'single');
            debug_maxproj(:,:,1) = max(histone_ds, [], 3);
            debug_maxproj(:,:,2) = max(LS_ds_warp, [], 3);
        end

    end
    rigid_x_min = rigid_x_min_ds{end};
    [eulerAngles, translation] = rigid_param_embedding(rigid_x_min, length_scale, angle_scale);
    rigid_tform = rigidtform3d(eulerAngles, translation);

    save(fullfile(output_path, ['tform_frame_', num2str(frames_to_align(ii))]), ...
        'crop_height', 'crop_width', 'crop_depth', 'histone_centroid', 'long_centroid', ...
        'rigid_x_min', 'rigid_MAES_state_ds', 'rigid_trackers_ds', 'rigid_x_min_ds', ...
        'length_scale', 'angle_scale', 'rigid_tform_ds', 'rigid_tform', ...
        'debug_stack', 'debug_maxproj');
end

end