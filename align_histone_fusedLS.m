clc;
clear;

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

path_to_long = 'D:\Posfai_Lab\MouseData\230101_st19_extract\long_nanog';
path_to_short = 'D:\Posfai_Lab\MouseData\230101_st19_extract\short_gata6';
path_to_histone = 'D:\Posfai_Lab\MouseData\230101_st19_extract\histone';

path_to_long_short_centers = './output/230101_st19/long_short_centers';
path_to_histone_centers = './output/230101_st19/histone_centers';

% path_to_long = '/scratch/gpfs/ddenberg/230101_st19_extract/long_nanog';
% path_to_short = '/scratch/gpfs/ddenberg/230101_st19_extract/short_gata6';
% path_to_histone = '/scratch/gpfs/ddenberg/230101_st19_extract/histone';

output_path = './output/230101_st19/align_LS_histone';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

long_short_tform = load('./output/230101_st19/align_long_short/tform_xy_average.mat');

frames_to_align = 100;
numThreads = 6;

% crop box for increasing performance
crop_h = 900;
crop_v = 900;
crop_z = 90;

% downsample factor (list of values for registration steps)
downsample_factor = [0.25];
sigma_init = [1e-1];
max_gen_init = [100];
population_size_init = [10];
tol = 1e-5;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[histone_filenames, histone_filename_folders] = get_filenames(path_to_histone, {'klb'}, {});
[long_filenames, long_filename_folders] = get_filenames(path_to_long, {'klb'}, {});
[short_filenames, short_filename_folders] = get_filenames(path_to_short, {'klb'}, {});
[histone_centers_filenames, histone_centers_filename_folders] = get_filenames(path_to_histone_centers, {'mat'}, {});
[LS_centers_filenames, LS_centers_filename_folders] = get_filenames(path_to_long_short_centers, {'mat'}, {});

% get each filename's corresponding frame number
histone_frames = get_frame_ids(histone_filenames);
long_frames = get_frame_ids(long_filenames);
short_frames = get_frame_ids(short_filenames);
histone_centers_frames = get_frame_ids(histone_centers_filenames);
LS_centers_frames = get_frame_ids(LS_centers_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:length(frames_to_align)

    % get nuclear, long, and short filenames
    histone_ind = find(histone_frames == frames_to_align(ii));
    long_ind = find(long_frames == frames_to_align(ii));
    short_ind = find(short_frames == frames_to_align(ii));
    histone_center_ind = find(histone_centers_frames == frames_to_align(ii));
    LS_center_ind = find(LS_centers_frames == frames_to_align(ii));

    % skip if one of the images is not present
    if isempty(short_ind) || isempty(long_ind) || isempty(histone_ind) || ...
            isempty(histone_center_ind) || isempty(LS_center_ind)
        continue;
    end

    histone_file = fullfile(histone_filename_folders{histone_ind}, histone_filenames{histone_ind});
    long_file = fullfile(long_filename_folders{long_ind}, long_filenames{long_ind});
    short_file = fullfile(short_filename_folders{short_ind}, short_filenames{short_ind});
    histone_center_file = fullfile(histone_centers_filename_folders{histone_center_ind}, ...
                                   histone_centers_filenames{histone_center_ind});
    LS_center_file = fullfile(LS_centers_filename_folders{LS_center_ind}, ...
                              LS_centers_filenames{LS_center_ind});

    %% read histone
    histone_img = readKLBstack(histone_file, numThreads);
    histone_center_struct = load(histone_center_file);
    histone_centroid = histone_center_struct.img_centroid;

    % normalize image
    histone_img = (single(histone_img) - histone_center_struct.img_bg_mean) / histone_center_struct.img_bg_std;

    % crop histone
    histone_hpair = round([histone_centroid(1) - crop_h / 2, histone_centroid(1) + crop_h / 2]);
    histone_hpair = min(max(histone_hpair, 1), size(histone_img, 1));

    histone_vpair = round([histone_centroid(2) - crop_v / 2, histone_centroid(2) + crop_v / 2]);
    histone_vpair = min(max(histone_vpair, 1), size(histone_img, 2));

    histone_zpair = round([histone_centroid(3) * resXY / resZ - crop_z / 2, ...
                       histone_centroid(3) * resXY / resZ + crop_z / 2]);
    histone_zpair = min(max(histone_zpair, 1), size(histone_img, 3));

    histone_crop = histone_img(histone_hpair(1):histone_hpair(2), ...
                               histone_vpair(1):histone_vpair(2), ...
                               histone_zpair(1):histone_zpair(2));

    clear histone_img;

    %% load long and short
    long_img = readKLBstack(long_file, numThreads);
    short_img = readKLBstack(short_file, numThreads);

    % load TF center
    LS_center_struct = load(LS_center_file);
    LS_centroid = LS_center_struct.img_centroid;

    % normalize images
    long_img = (single(long_img) - LS_center_struct.long_bg_mean) / LS_center_struct.long_bg_std;
    short_img = (single(short_img) - LS_center_struct.short_bg_mean) / LS_center_struct.short_bg_std;

    % warp short channel
    short_img = imwarp(short_img, long_short_tform.tform, 'OutputView', imref3d(size(long_img)));

    % create LS crop
    LS_hpair = round([LS_centroid(1) - crop_h / 2, LS_centroid(1) + crop_h / 2]);
    LS_hpair = min(max(LS_hpair, 1), size(long_img, 1));

    LS_vpair = round([LS_centroid(2) - crop_v / 2, LS_centroid(2) + crop_v / 2]);
    LS_vpair = min(max(LS_vpair, 1), size(long_img, 2));

    LS_zpair = round([LS_centroid(3) * resXY / resZ - crop_z / 2, ...
                      LS_centroid(3) * resXY / resZ + crop_z / 2]);
    LS_zpair = min(max(LS_zpair, 1), size(long_img, 3));

    LS_crop = short_img(LS_hpair(1):LS_hpair(2), ...
                        LS_vpair(1):LS_vpair(2), ...
                        LS_zpair(1):LS_zpair(2)) + ...
               long_img(LS_hpair(1):LS_hpair(2), ...
                        LS_vpair(1):LS_vpair(2), ...
                        LS_zpair(1):LS_zpair(2));

    clear short_img long_img;

    % loop over downsample stages and use previous registration as initialization for the next
    rigid_trackers_ds = cell(length(downsample_factor), 1);
    rigid_MAES_state_ds = cell(length(downsample_factor), 1);
    rigid_x_min_ds = cell(length(downsample_factor), 1);
    rigid_f_min_ds = cell(length(downsample_factor), 1);
    length_scale = mean([crop_h, crop_v, crop_z * resZ / resXY]) / 4;
    rigid_tform_ds = cell(length(downsample_factor), 1);
    for jj = 1:length(downsample_factor)
        LS_ds = isotropicSample_bilinear(LS_crop, resXY, resZ, downsample_factor(jj));
        histone_ds = isotropicSample_bilinear(histone_crop, resXY, resZ, downsample_factor(jj));

        sigma = sigma_init(jj);
        max_gen = max_gen_init(jj);
        population_size = population_size_init(jj);

        rigid_fun_h = @(x) rigid_loss(histone_ds, LS_ds, x, length_scale, ...
                                      histone_centroid, histone_hpair, histone_vpair, histone_zpair, ...
                                      LS_centroid, LS_hpair, LS_vpair, LS_zpair, resXY, resZ);

        % do down sampled registration first
        % initialize parameters for optimization
        if jj == 1
            x_init = zeros(6, 1);% 0 x-y offset initial guess
        else
            x_init = x_min_ds;
        end

        MAES_state_ds = MAES_initialize(x_init, sigma, max_gen, tol, population_size);
        
        tic;
        [MAES_state_ds, x_min_ds, f_min_ds, trackers_ds] = MAES_run(MAES_state_ds, rigid_fun_h, false);
        toc;

        rigid_trackers_ds{jj} = trackers_ds;
        rigid_MAES_state_ds{jj} = MAES_state_ds;
        rigid_x_min_ds{jj} = x_min_ds;
        rigid_f_min_ds{jj} = f_min_ds;
        rigid_tform_ds{jj} = rigidtform3d(x_min_ds(4:6).' * 180, x_min_ds(1:3).' * length_scale);

        if jj == 1
            % create debug registered images
            [~, LS_ds_warp] = rigid_fun_h(x_min_ds);

            z_ind = round(histone_centroid(3) * downsample_factor(jj));
            debug_slice = zeros(size(LS_ds_warp, 1), size(LS_ds_warp, 2), 2);
            debug_slice(:,:,1) = histone_ds(:,:,z_ind);
            debug_slice(:,:,2) = LS_ds_warp(:,:,z_ind);
        end

    end
    rigid_x_min = rigid_x_min_ds{end};
    rigid_tform = rigidtform3d(rigid_x_min(4:6).' * 180, rigid_x_min(1:3).' * length_scale);

    save(fullfile(output_path, ['tform_frame_', num2str(frames_to_align(ii))]), ...
        'histone_centroid', 'histone_hpair', 'histone_vpair', 'histone_zpair', ...
        'LS_centroid', 'LS_hpair', 'LS_vpair', 'LS_zpair', ...
        'rigid_x_min', 'rigid_MAES_state_ds', 'rigid_trackers_ds', 'rigid_x_min_ds', ...
        'rigid_f_min_ds', 'length_scale', 'rigid_tform_ds', 'rigid_tform', 'debug_slice');
end