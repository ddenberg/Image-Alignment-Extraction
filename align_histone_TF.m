% clc;
% clear;
function align_histone_TF(path_to_TF, path_to_histone, path_to_TF_centers, path_to_histone_centers, output_path, frames_to_align, numThreads)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% path_to_TF = 'D:/Posfai_Lab/MouseData/230101_st19_extract/long_nanog';
% path_to_histone = 'D:/Posfai_Lab/MouseData/230101_st19_extract/histone';
% path_to_TF_centers = './output/230101_st19/long_nanog_centers';
% path_to_histone_centers = './output/230101_st19/histone_centers';

% output_path = './output/230101_st19/align_long_nanog_histone';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% frames_to_align = [10,20,24,26,28,34,38,58,82,100,122];
% frames_to_align = [72];
% frames_to_align = first_frame:last_frame;
% numThreads = 16;

% crop box for increasing performance
crop_height = 900;
crop_width = 900;
crop_depth = 900;

% parameters for normalization
max_zscore = 100;

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
[TF_filenames, TF_filename_folders] = get_filenames(path_to_TF, {'klb'}, {});
[histone_centers_filenames, histone_centers_filename_folders] = get_filenames(path_to_histone_centers, {'mat'}, {});
[TF_centers_filenames, TF_centers_filename_folders] = get_filenames(path_to_TF_centers, {'mat'}, {});

% get each filename's corresponding frame number
histone_frames = get_frame_ids(histone_filenames);
TF_frames = get_frame_ids(TF_filenames);
histone_centers_frames = get_frame_ids(histone_centers_filenames);
TF_centers_frames = get_frame_ids(TF_centers_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:length(frames_to_align)

    % get nuclear, long, and short filenames
    histone_ind = find(histone_frames == frames_to_align(ii));
    TF_ind = find(TF_frames == frames_to_align(ii));
    histone_center_ind = find(histone_centers_frames == frames_to_align(ii));
    TF_center_ind = find(TF_centers_frames == frames_to_align(ii));

    % skip if one of the images is not present
    if isempty(TF_ind) || isempty(histone_ind) || isempty(histone_center_ind) || isempty(TF_center_ind)
        continue;
    end

    histone_file = fullfile(histone_filename_folders{histone_ind}, histone_filenames{histone_ind});
    TF_file = fullfile(TF_filename_folders{TF_ind}, TF_filenames{TF_ind});
    histone_center_file = fullfile(histone_centers_filename_folders{histone_center_ind}, ...
                                   histone_centers_filenames{histone_center_ind});
    TF_center_file = fullfile(TF_centers_filename_folders{TF_center_ind}, ...
                                   TF_centers_filenames{TF_center_ind});

    % read in nuclear image
    histone_img = readKLBstack(histone_file, numThreads);
    
    % load histone center
    histone_center_struct = load(histone_center_file);
    histone_centroid = histone_center_struct.img_centroid;
    
    fprintf('Frame %d\n', frames_to_align(ii));
    fprintf('  Histone Centroid %f, %f, %f\n', histone_centroid(1), histone_centroid(2), histone_centroid(3));

    % normalize image
    histone_img = (single(histone_img) - histone_center_struct.img_bg_mean) / histone_center_struct.img_bg_std;

    % crop histone
    [histone_crop, histone_hpair, histone_wpair, histone_dpair] = isotropic_crop(histone_img, histone_centroid, ...
        crop_height, crop_width, crop_depth, resXY, resZ, 'bilinear');

    % cap z score
    histone_crop = min(histone_crop, max_zscore);

    clear histone_img;

    % read in TF image
    TF_img = readKLBstack(TF_file, numThreads);

    % load TF center
    TF_center_struct = load(TF_center_file);
    TF_centroid = TF_center_struct.img_centroid;

    fprintf('  TF Centroid %f, %f, %f\n', TF_centroid(1), TF_centroid(2), TF_centroid(3));

    % normalize images
    TF_img = (single(TF_img) - TF_center_struct.img_bg_mean) / TF_center_struct.img_bg_std;

    % create crop
    [TF_crop, TF_hpair, TF_wpair, TF_dpair] = isotropic_crop(TF_img, TF_centroid, ...
        crop_height, crop_width, crop_depth, resXY, resZ, 'bilinear');

    % cap z score
    TF_crop = min(TF_crop, max_zscore);

    clear TF_img;    

    % loop over downsample stages and use previous registration as initialization for the next
    rigid_trackers_ds = cell(length(downsample_factor), 1);
    rigid_MAES_state_ds = cell(length(downsample_factor), 1);
    rigid_x_min_ds = cell(length(downsample_factor), 1);
    rigid_tform_ds = cell(length(downsample_factor), 1);

    length_scale = 80;
    angle_scale = 180;
    
    for jj = 1:length(downsample_factor)
        TF_ds = imresize3(TF_crop, downsample_factor(jj), 'linear');
        histone_ds = imresize3(histone_crop, downsample_factor(jj), 'linear');

        sigma = sigma_init(jj);
        max_gen = max_gen_init(jj);
        population_size = population_size_init(jj);

        rigid_fun_h = @(x) rigid_loss(histone_ds, TF_ds, x, length_scale, angle_scale, ...
                                      histone_centroid, histone_hpair, histone_wpair, histone_dpair, ...
                                      TF_centroid, TF_hpair, TF_wpair, TF_dpair);

        % do down sampled registration first
        % initialize parameters for optimization
        if jj == 1
            x_init = zeros(6, 1); % initial guess
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
            [~, TF_ds_warp] = rigid_fun_h(x_min_ds);

%             debug_stack = zeros([size(TF_ds_warp), 2], 'single');
%             debug_stack(:,:,:,1) = histone_ds;
%             debug_stack(:,:,:,2) = TF_ds_warp;

            debug_stack = zeros([size(TF_ds_warp), 2], 'uint8');

            histone_ds_prctile = prctile(histone_ds, [0.1, 99.9], 'all');
            debug_stack(:,:,:,1) = uint8(rescale(histone_ds, 0, 2^8-1, ...
                'InputMin', histone_ds_prctile(1), 'InputMax', histone_ds_prctile(2)));

            TF_ds_warp_prctile = prctile(TF_ds_warp, [0.1, 99.9], 'all');
            debug_stack(:,:,:,2) = uint8(rescale(TF_ds_warp, 0, 2^8-1, ...
                'InputMin', TF_ds_warp_prctile(1), 'InputMax', TF_ds_warp_prctile(2)));

            debug_maxproj = zeros(size(TF_ds_warp, 1), size(TF_ds_warp, 2), 2, 'single');
            debug_maxproj(:,:,1) = max(histone_ds, [], 3);
            debug_maxproj(:,:,2) = max(TF_ds_warp, [], 3);
        end

    end
    rigid_x_min = rigid_x_min_ds{end};
    [eulerAngles, translation] = rigid_param_embedding(rigid_x_min, length_scale, angle_scale);
    rigid_tform = rigidtform3d(eulerAngles, translation);

    save(fullfile(output_path, ['tform_frame_', num2str(frames_to_align(ii))]), ...
        'crop_height', 'crop_width', 'crop_depth', 'histone_centroid', ...
        'TF_centroid', 'rigid_x_min', 'rigid_MAES_state_ds', 'rigid_trackers_ds', ...
        'rigid_x_min_ds', 'length_scale', 'angle_scale', 'rigid_tform_ds', 'rigid_tform', ...
        'debug_stack', 'debug_maxproj');
end

end
