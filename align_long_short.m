function align_long_short(path_to_long_images, path_to_long_centers, ...
    path_to_short_images, path_to_short_centers, output_path, frames_to_align, ...
    numThreads, image_format, overwrite_offset)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% output_path = './output/230212_st6/align_long_short';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% crop box for increasing performance
crop_height = 900;
crop_width = 900;

% parameters for normalization
max_zscore = 100;

% downsample factor (list of values for registration steps)
sigma_init = 1e0;
max_gen_init = 200;
population_size_init = 8;
tol = 1e-5;

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

if ~isempty(overwrite_offset)
    overwrite_offset = overwrite_offset(:);
    if ~all(size(overwrite_offset) == [2, 1])
        error('overwrite_centroid does not have the correct shape. Must be [2, 1]');
    end
end

% get filenames in each directory (excluding .label and .tif images)
[long_filenames, long_folders] = get_filenames(path_to_long_images, {image_ext}, {'._'});
[short_filenames, short_folders] = get_filenames(path_to_short_images, {image_ext}, {'._'});
[long_centers_filenames, long_centers_filename_folders] = get_filenames(path_to_long_centers, {'mat'}, {'._'});
[short_centers_filenames, short_centers_filename_folders] = get_filenames(path_to_short_centers, {'mat'}, {'._'});

% get each filename's corresponding frame number
long_frames = get_frame_ids(long_filenames);
short_frames = get_frame_ids(short_filenames);
long_centers_frames = get_frame_ids(long_centers_filenames);
short_centers_frames = get_frame_ids(short_centers_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)

for ii = 1:length(frames_to_align)

    % get long and short inds
    long_ind = find(long_frames == frames_to_align(ii));
    short_ind = find(short_frames == frames_to_align(ii));
    long_center_ind = find(long_centers_frames == frames_to_align(ii));
    short_center_ind = find(short_centers_frames == frames_to_align(ii));

    % skip if one of the images is not present
    if isempty(short_ind) || isempty(long_ind) || isempty(long_center_ind) || isempty(short_center_ind)
        continue;
    end

    % get long and short filenames
    long_file = fullfile(long_folders{long_ind}, long_filenames{long_ind});
    long_center_file = fullfile(long_centers_filename_folders{long_center_ind}, ...
        long_centers_filenames{long_center_ind});
    short_file = fullfile(short_folders{short_ind}, short_filenames{short_ind});
    short_center_file = fullfile(short_centers_filename_folders{short_center_ind}, ...
        short_centers_filenames{short_center_ind});

    % read in long and short images
    if read_klb
        long_img = readKLBstack(long_file, numThreads);
        short_img = readKLBstack(short_file, numThreads);
    else
        long_img = tiffreadVolume(long_file);
        long_img = permute(long_img, [2, 1, 3]);

        short_img = tiffreadVolume(short_file);
        short_img = permute(short_img, [2, 1, 3]);
    end

    % load centers
    long_center_struct = load(long_center_file);
    long_centroid = long_center_struct.img_centroid;
    short_center_struct = load(short_center_file);
%     short_centroid = short_center_struct.img_centroid;

    % crop long image
    long_crop = xy_crop(long_img, long_centroid, crop_height, crop_width);

    clear long_img

    % crop short image
    short_crop = xy_crop(short_img, long_centroid, crop_height, crop_width);

    clear short_img;

    % Convert to float32
    long_crop = single(long_crop);
    short_crop = single(short_crop);
    
    % normalize histone and TF images 
    long_crop = (long_crop - long_center_struct.img_bg_mean) / long_center_struct.img_bg_std;
    short_crop = (short_crop - short_center_struct.img_bg_mean) / short_center_struct.img_bg_std;

    % cap value of zscores
    long_crop = min(long_crop, max_zscore);
    short_crop = min(short_crop, max_zscore);

    % loop over downsample stages and use previous registration as initialization for the next
    sigma = sigma_init(1);
    max_gen = max_gen_init(1);
    population_size = population_size_init(1);
    length_scale = 50; % 100
    translate_fun_h = @(x) translate_xy_loss(long_crop, short_crop, x, length_scale);

    % initialize parameters for optimization
    x_init = zeros(2, 1);
    
    MAES_state = MAES_initialize(x_init, sigma, max_gen, tol, population_size);
    
    tic;
    if isempty(overwrite_offset)
        [MAES_state, x_min, ~, translate_trackers] = MAES_run(MAES_state, translate_fun_h, true);
        translation = translate_xy_param_embedding(x_min, length_scale);
    else
        translate_trackers = [];
        x_min = asin(overwrite_offset / length_scale);
        translation = [overwrite_offset; 0];
    end
    toc;

    translation_tform = transltform3d(translation);

    % create debug registered images
    [~, short_crop_warp] = translate_fun_h(x_min);

    short_crop_warp_ds = imresize(short_crop_warp, 0.25, 'bilinear');
    short_crop_warp_ds_prctile = prctile(short_crop_warp_ds, [0.1, 99.9], 'all');
    short_crop_warp_ds = rescale(short_crop_warp_ds, 0, 255, ...
                'InputMin', short_crop_warp_ds_prctile(1), 'InputMax', short_crop_warp_ds_prctile(2));
    short_crop_warp_ds = uint8(short_crop_warp_ds);

    long_crop_ds = imresize(long_crop, 0.25, 'bilinear');
    long_crop_ds_prctile = prctile(long_crop_ds, [0.1, 99.9], 'all');
    long_crop_ds = rescale(long_crop_ds, 0, 255, ...
                'InputMin', long_crop_ds_prctile(1), 'InputMax', long_crop_ds_prctile(2));
    long_crop_ds = uint8(long_crop_ds);

    debug_stack = cat(4, long_crop_ds, short_crop_warp_ds);

    % debug_maxproj = zeros(size(short_crop_warp, 1), size(short_crop_warp, 2), 2, 'single');
    % debug_maxproj(:,:,1) = max(long_crop, [], 3);
    % debug_maxproj(:,:,2) = max(short_crop_warp, [], 3);
    % 
    % z_ind = round(long_centroid(3) * (resXY / resZ));
    % debug_center = zeros(size(short_crop_warp, 1), size(short_crop_warp, 2), 2, 'single');
    % debug_center(:,:,1) = long_crop(:,:,z_ind);
    % debug_center(:,:,2) = short_crop_warp(:,:,z_ind);

    save(fullfile(output_path, ['tform_xy_frame_', num2str(frames_to_align(ii))]), ...
        'translation_tform', 'x_min', 'MAES_state', 'translate_trackers', ...
        'length_scale', 'debug_stack');

    fprintf('Frame %d, Done!\n', frames_to_align(ii));
end

end
