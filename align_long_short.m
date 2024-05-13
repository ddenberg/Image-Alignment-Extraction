% clc;
% clear;
function align_long_short(path_to_long_images, path_to_long_centers, ...
    path_to_short_images, path_to_short_centers, output_path, frames_to_align, numThreads)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% path_to_long_cam = 'D:/Posfai_Lab/MATLAB/Mouse_G6N/Stack19_WT/transcripton_factors/long';
% path_to_short_cam = 'D:/Posfai_Lab/MATLAB/Mouse_G6N/Stack19_WT/transcripton_factors/short';

% path_to_long_cam = '/scratch/gpfs/ddenberg/230212_st6_extract/Ch0long_nanog';
% path_to_short_cam = '/scratch/gpfs/ddenberg/230212_st6_extract/Ch0short_gata6';

% output_path = './output/230212_st6/align_long_short';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% frames_to_align = 0:10:130;
% numThreads = 16;

% crop box for increasing performance
crop_height = 900;
crop_width = 900;

% parameters for normalization
max_zscore = 100;

% downsample factor (list of values for registration steps)
sigma_init = 1e-1;
max_gen_init = 200;
population_size_init = 6;
tol = 1e-5;

% anisotropy factors
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[long_filenames, long_folders] = get_filenames(path_to_long_images, {'klb'}, {});
[short_filenames, short_folders] = get_filenames(path_to_short_images, {'klb'}, {});
[long_centers_filenames, long_centers_filename_folders] = get_filenames(path_to_long_centers, {'mat'}, {});
[short_centers_filenames, short_centers_filename_folders] = get_filenames(path_to_short_centers, {'mat'}, {});

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
    long_img = readKLBstack(long_file, numThreads);
    short_img = readKLBstack(short_file, numThreads);

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
    length_scale = 100;
    translate_fun_h = @(x) translate_xy_loss(long_crop, short_crop, x, length_scale);

    % initialize parameters for optimization
    x_init = zeros(2, 1);
    
    MAES_state = MAES_initialize(x_init, sigma, max_gen, tol, population_size);
    
    tic;
    [MAES_state, x_min, ~, translate_trackers] = MAES_run(MAES_state, translate_fun_h, true);
    toc;

    translation = translate_xy_param_embedding(x_min, length_scale);
    translation_tform = transltform3d(translation);

    % create debug registered images
    [~, short_crop_warp] = translate_fun_h(x_min);

    debug_maxproj = zeros(size(short_crop_warp, 1), size(short_crop_warp, 2), 2, 'single');
    debug_maxproj(:,:,1) = max(long_crop, [], 3);
    debug_maxproj(:,:,2) = max(short_crop_warp, [], 3);

    z_ind = round(long_centroid(3) * (resXY / resZ));
    debug_center = zeros(size(short_crop_warp, 1), size(short_crop_warp, 2), 2, 'single');
    debug_center(:,:,1) = long_crop(:,:,z_ind);
    debug_center(:,:,2) = short_crop_warp(:,:,z_ind);

    save(fullfile(output_path, ['tform_xy_frame_', num2str(frames_to_align(ii))]), ...
        'translation_tform', 'x_min', 'MAES_state', 'translate_trackers', ...
        'length_scale', 'debug_maxproj', 'debug_center');

    fprintf('Frame %d, Done!\n', frames_to_align(ii));
end

end
