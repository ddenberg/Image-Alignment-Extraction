% clc;
% clear;
function align_histone_TF(path_to_TF, path_to_histone, path_to_TF_centers, path_to_histone_centers, output_path, frames_to_align, numThreads)

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
crop_h = 900;
crop_v = 900;
crop_z = 90;

% parameters for normalization
max_zscore = 25;

% downsample factor (list of values for registration steps)
downsample_factor = [0.25];
sigma_init_rigid = [1e-1];
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
    histone_raw = readKLBstack(histone_file, numThreads);
    
    % load histone center
    histone_center_struct = load(histone_center_file);
    histone_centroid = histone_center_struct.img_centroid;

    fprintf('Frame %d, Histone Centroid %f, %f, %f\n', frames_to_align(ii), histone_centroid(1), histone_centroid(2), histone_centroid(3));

    histone_hpair = round([histone_centroid(1) - crop_h / 2, histone_centroid(1) + crop_h / 2]);
    histone_hpair = min(max(histone_hpair, 1), size(histone_raw, 1));

    histone_vpair = round([histone_centroid(2) - crop_v / 2, histone_centroid(2) + crop_v / 2]);
    histone_vpair = min(max(histone_vpair, 1), size(histone_raw, 2));

    histone_zpair = round([histone_centroid(3) * resXY / resZ - crop_z / 2, ...
                           histone_centroid(3) * resXY / resZ + crop_z / 2]);
    histone_zpair = min(max(histone_zpair, 1), size(histone_raw, 3));

    histone_crop = histone_raw(histone_hpair(1):histone_hpair(2), ...
                               histone_vpair(1):histone_vpair(2), ...
                               histone_zpair(1):histone_zpair(2));

    clear histone_raw;

    % read in TF image
    TF_raw = readKLBstack(TF_file, numThreads);

    % load TF center
    TF_center_struct = load(TF_center_file);
    TF_centroid = TF_center_struct.img_centroid;

    fprintf('Frame %d, TF Centroid %f, %f, %f\n', frames_to_align(ii), TF_centroid(1), TF_centroid(2), TF_centroid(3));

    TF_hpair = round([TF_centroid(1) - crop_h / 2, TF_centroid(1) + crop_h / 2]);
    TF_hpair = min(max(TF_hpair, 1), size(TF_raw, 1));

    TF_vpair = round([TF_centroid(2) - crop_v / 2, TF_centroid(2) + crop_v / 2]);
    TF_vpair = min(max(TF_vpair, 1), size(TF_raw, 2));

    TF_zpair = round([TF_centroid(3) * resXY / resZ - crop_z / 2, ...
                      TF_centroid(3) * resXY / resZ + crop_z / 2]);
    TF_zpair = min(max(TF_zpair, 1), size(TF_raw, 3));

    TF_crop = TF_raw(TF_hpair(1):TF_hpair(2), ...
                     TF_vpair(1):TF_vpair(2), ...
                     TF_zpair(1):TF_zpair(2));

    clear TF_raw;

    % Convert to float32
    histone_crop = single(histone_crop);
    TF_crop = single(TF_crop);
    
    % normalize histone and TF images 
    histone_crop = (histone_crop - histone_center_struct.img_bg_mean) / histone_center_struct.img_bg_std;
    TF_crop = (TF_crop - TF_center_struct.img_bg_mean) / TF_center_struct.img_bg_std;

    % cap value of zscores
    histone_crop = min(histone_crop, max_zscore);
    TF_crop = min(TF_crop, max_zscore);
    

    % loop over downsample stages and use previous registration as initialization for the next
    rigid_trackers_ds = cell(length(downsample_factor), 1);
    rigid_MAES_state_ds = cell(length(downsample_factor), 1);
    rigid_x_min_ds = cell(length(downsample_factor), 1);
    length_scale = mean([crop_h, crop_v, crop_z * resZ / resXY]) / 4;
    angle_scale = 180;
    rigid_tform_ds = cell(length(downsample_factor), 1);
    for jj = 1:length(downsample_factor)
        TF_ds = isotropicSample_bilinear(TF_crop, resXY, resZ, downsample_factor(jj));
        histone_ds = isotropicSample_bilinear(histone_crop, resXY, resZ, downsample_factor(jj));

        sigma = sigma_init_rigid(jj);
        max_gen = max_gen_init(jj);
        population_size = population_size_init(jj);

        rigid_fun_h = @(x) rigid_loss(histone_ds, TF_ds, x, length_scale, angle_scale, ...
                                      histone_centroid, histone_hpair, histone_vpair, histone_zpair, ...
                                      TF_centroid, TF_hpair, TF_vpair, TF_zpair, resXY, resZ);

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
        [eulerAngles, translation] = param_embedding(x_min_ds, length_scale, angle_scale);
        rigid_tform_ds{jj} = rigidtform3d(eulerAngles, translation);

        if jj == 1
            % create debug registered images
            [~, TF_ds_warp] = rigid_fun_h(x_min_ds);

            z_ind = round(histone_centroid(3) * downsample_factor(jj));
            debug_slice_center = zeros(size(TF_ds_warp, 1), size(TF_ds_warp, 2), 2);
            debug_slice_center(:,:,1) = histone_ds(:,:,z_ind);
            debug_slice_center(:,:,2) = TF_ds_warp(:,:,z_ind);

            debug_slice_maxproj = zeros(size(TF_ds_warp, 1), size(TF_ds_warp, 2), 2);
            debug_slice_maxproj(:,:,1) = max(histone_ds, [], 3);
            debug_slice_maxproj(:,:,2) = max(TF_ds_warp, [], 3);
        end

    end
    rigid_x_min = rigid_x_min_ds{end};
    [eulerAngles, translation] = param_embedding(rigid_x_min, length_scale, angle_scale);
    rigid_tform = rigidtform3d(eulerAngles, translation);

    save(fullfile(output_path, ['tform_frame_', num2str(frames_to_align(ii))]), ...
        'histone_centroid', 'histone_hpair', 'histone_vpair', 'histone_zpair', ...
        'TF_centroid', 'TF_hpair', 'TF_vpair', 'TF_zpair', ...
        'rigid_x_min', 'rigid_MAES_state_ds', 'rigid_trackers_ds', 'rigid_x_min_ds', ...
        'length_scale', 'angle_scale', 'rigid_tform_ds', 'rigid_tform', ...
        'debug_slice_center', 'debug_slice_maxproj');
end

end
