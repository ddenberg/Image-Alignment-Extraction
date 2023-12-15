clc;
clear;

addpath('loss_functions');
addpath('MA-ES');


path_to_long = 'D:\Posfai_Lab\MouseData\230101_st19_extract\long_nanog';
path_to_short = 'D:\Posfai_Lab\MouseData\230101_st19_extract\short_gata6';
path_to_segmentation = 'D:\Posfai_Lab\MouseData\230101_st19_extract\segmentation';

% path_to_long = '/scratch/gpfs/ddenberg/230101_st19_extract/long_nanog';
% path_to_short = '/scratch/gpfs/ddenberg/230101_st19_extract/short_gata6';
% path_to_segmentation = '/scratch/gpfs/ddenberg/230101_st19_extract/segmentation';

align_histone_LS_path = './output/230101_st19/align_LS_histone';

output_path = './output/230101_st19/output_extraction';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

long_short_tform = load('./output/230101_st19/align_long_short/tform_xy_average.mat');

frames_to_align = 0:108;
numThreads = 6;

% crop box for increasing performance
crop_h = 900;
crop_v = 900;
crop_z = 90;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% gmm parameters
tol_bm = 0.4; % bimodality coefficient threshold
tol_diff = 0.1; % gmm weight difference threshold

% get filenames in each directory (excluding .label and .tif images)
[seg_filenames, seg_filename_folders] = get_filenames(path_to_segmentation, {'klb'}, {});
[long_filenames, long_filename_folders] = get_filenames(path_to_long, {'klb'}, {});
[short_filenames, short_filename_folders] = get_filenames(path_to_short, {'klb'}, {});
[tform_filenames, tform_filename_folders] = get_filenames(align_histone_LS_path, {'.mat'}, {});

% get each filename's corresponding frame number
seg_frames = get_frame_ids(seg_filenames);
long_frames = get_frame_ids(long_filenames);
short_frames = get_frame_ids(short_filenames);
tform_frames = get_frame_ids(tform_filenames);

% output lists
extract_long = cell(length(frames_to_align), 1);
extract_short = cell(length(frames_to_align), 1);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
tform_identity = rigidtform3d([0, 0, 0], [0, 0, 0]);
for ii = 1:length(frames_to_align)

    % get nuclear, long, and short filenames
    seg_ind = find(seg_frames == frames_to_align(ii));
    long_ind = find(long_frames == frames_to_align(ii));
    short_ind = find(short_frames == frames_to_align(ii));
    tform_ind = find(tform_frames == frames_to_align(ii));

    % skip if one of the images is not present
    if isempty(short_ind) || isempty(long_ind) || isempty(seg_ind) || isempty(tform_ind)
        continue;
    end

    seg_file = fullfile(seg_filename_folders{seg_ind}, seg_filenames{seg_ind});
    long_file = fullfile(long_filename_folders{long_ind}, long_filenames{long_ind});
    short_file = fullfile(short_filename_folders{short_ind}, short_filenames{short_ind});
    tform_file = fullfile(tform_filename_folders{tform_ind}, tform_filenames{tform_ind});

    % read in nuclear, long, and short images
    seg_img = readKLBstack(seg_file, numThreads);
    long_img = readKLBstack(long_file, numThreads);
    short_img = readKLBstack(short_file, numThreads);
    tform_struct = load(tform_file);

    % warp short channel
    short_img = imwarp(short_img, long_short_tform.tform, 'OutputView', imref3d(size(long_img)));

    % crop images
    seg_crop = seg_img(tform_struct.histone_hpair(1):tform_struct.histone_hpair(2), ...
                       tform_struct.histone_vpair(1):tform_struct.histone_vpair(2), ...
                       tform_struct.histone_zpair(1):tform_struct.histone_zpair(2));
    long_crop = long_img(tform_struct.LS_hpair(1):tform_struct.LS_hpair(2), ...
                         tform_struct.LS_vpair(1):tform_struct.LS_vpair(2), ...
                         tform_struct.LS_zpair(1):tform_struct.LS_zpair(2));
    short_crop = short_img(tform_struct.LS_hpair(1):tform_struct.LS_hpair(2), ...
                          tform_struct.LS_vpair(1):tform_struct.LS_vpair(2), ...
                          tform_struct.LS_zpair(1):tform_struct.LS_zpair(2));

    clear long_img short_img seg_img;

    % create isotropic segmentation, long and short channels
    short_crop_iso = isotropicSample_bilinear(short_crop, resXY, resZ, 1);
    long_crop_iso = isotropicSample_bilinear(long_crop, resXY, resZ, 1);
    seg_crop_iso = isotropicSample_nearest(seg_crop, resXY, resZ, 1);

    % create segmentation (fixed) referencing struct
    seg_ref = imref3d(size(seg_crop_iso));
    seg_ref.XWorldLimits = tform_struct.histone_hpair - tform_struct.histone_centroid(1);
    seg_ref.YWorldLimits = tform_struct.histone_vpair - tform_struct.histone_centroid(2);
    seg_ref.ZWorldLimits = (tform_struct.histone_zpair - 1) * resZ / resXY + 1 - tform_struct.histone_centroid(3);

    % create TF (moving) referencing struct
    LS_ref = imref3d(size(long_crop_iso));
    LS_ref.XWorldLimits = tform_struct.LS_hpair - tform_struct.LS_centroid(1);
    LS_ref.YWorldLimits = tform_struct.LS_vpair - tform_struct.LS_centroid(2);
    LS_ref.ZWorldLimits = (tform_struct.LS_zpair - 1) * resZ / resXY + 1 - tform_struct.LS_centroid(3);

    % warp long channel
    long_crop_iso_nowarp = imwarp(long_crop_iso, LS_ref, tform_identity, 'OutputView', seg_ref);
    long_crop_iso_warp = imwarp(long_crop_iso, LS_ref, tform_struct.rigid_tform, 'OutputView', seg_ref);    

    % use regionprops3 to extract values in long
    stats_long_nowarp = regionprops3(seg_crop_iso, long_crop_iso_nowarp, {'Volume', 'MeanIntensity', 'VoxelValues'});
    stats_long_rigid = regionprops3(seg_crop_iso, long_crop_iso_warp, {'Volume', 'MeanIntensity', 'VoxelValues'});

    % while size(stats_long_rigid, 1) < size(stats_long_nowarp, 1)
    %     stats_long_rigid = vertcat(stats_long_rigid, table(nan, nan, {zeros(0, 'uint16')}, ...
    %         'VariableNames', {'Volume', 'MeanIntensity', 'VoxelValues'}));
    % end

    ids = (1:size(stats_long_nowarp, 1)).';
    filter_ids = ~isnan(stats_long_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_long_nowarp = stats_long_nowarp(filter_ids,:);
    stats_long_rigid = stats_long_rigid(filter_ids,:);

    % do gmm filtering
    long_nowarp_gmm = zeros(length(ids), 1);
    long_rigid_gmm = zeros(length(ids), 1);

    long_nowarp_bm = zeros(length(ids), 1);
    long_rigid_bm = zeros(length(ids), 1);

    % for jj = 1:length(ids) 
    % 
    %     % nowarp
    %     X = double(stats_long_nowarp.VoxelValues{jj});
    %     [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
    %     long_nowarp_gmm(jj) = mu;
    %     long_nowarp_bm(jj) = bm;
    % 
    %     % rigid
    %     X = double(stats_long_rigid.VoxelValues{jj});
    %     [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
    %     long_rigid_gmm(jj) = mu;
    %     long_rigid_bm(jj) = bm;
    % end

    extract_long{ii} = table(repmat(frames_to_align(ii), length(ids), 1), ids, ...
        stats_long_nowarp.Volume, stats_long_nowarp.MeanIntensity, long_nowarp_gmm, long_nowarp_bm, ...
        stats_long_rigid.Volume, stats_long_rigid.MeanIntensity, long_rigid_gmm, long_rigid_bm, ...
        'VariableNames', {'Frame', 'ID', 'Volume_nowarp', 'MeanIntensity_nowarp', 'GMMIntensity_nowarp', 'Bimodality_nowarp', ...
        'Volume_rigid', 'MeanIntensity_rigid', 'GMMIntensity_rigid', 'Bimodality_rigid'});

    fprintf('Frame %d/%d, Long Cam Done!\n', frames_to_align(ii), max(frames_to_align));

    % warp short channel
    short_crop_iso_nowarp = imwarp(long_crop_iso, LS_ref, tform_identity, 'OutputView', seg_ref);
    short_crop_iso_warp = imwarp(short_crop_iso, LS_ref, tform_struct.rigid_tform, 'OutputView', seg_ref);
    
    % use regionprops3 to extract values in short
    stats_short_nowarp = regionprops3(seg_crop_iso, short_crop_iso_nowarp, {'Volume', 'MeanIntensity', 'VoxelValues'});
    stats_short_rigid = regionprops3(seg_crop_iso, short_crop_iso_warp, {'Volume', 'MeanIntensity', 'VoxelValues'});

    % while size(stats_short_rigid, 1) < size(stats_short_nowarp, 1)
    %     stats_short_rigid = vertcat(stats_short_rigid, table(nan, nan, {zeros(0, 'uint16')}, ...
    %         'VariableNames', {'Volume', 'MeanIntensity', 'VoxelValues'}));
    % end

    ids = (1:size(stats_short_nowarp, 1)).';
    filter_ids = ~isnan(stats_short_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_short_nowarp = stats_short_nowarp(filter_ids,:);
    stats_short_rigid = stats_short_rigid(filter_ids,:);

    % do gmm filtering
    short_nowarp_gmm = zeros(length(ids), 1);
    short_rigid_gmm = zeros(length(ids), 1);

    short_nowarp_bm = zeros(length(ids), 1);
    short_rigid_bm = zeros(length(ids), 1);

    % for jj = 1:length(ids) 
    % 
    %     % nowarp
    %     X = double(stats_short_nowarp.VoxelValues{jj});
    %     [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
    %     short_nowarp_gmm(jj) = mu;
    %     short_nowarp_bm(jj) = bm;
    % 
    %     % rigid
    %     X = double(stats_short_rigid.VoxelValues{jj});
    %     [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
    %     short_rigid_gmm(jj) = mu;
    %     short_rigid_bm(jj) = bm;
    % end

    extract_short{ii} = table(repmat(frames_to_align(ii), length(ids), 1), ids, ...
        stats_short_nowarp.Volume, stats_short_nowarp.MeanIntensity, short_nowarp_gmm, short_nowarp_bm, ...
        stats_short_rigid.Volume, stats_short_rigid.MeanIntensity, short_rigid_gmm, short_rigid_bm, ...
        'VariableNames', {'Frame', 'ID', 'Volume_nowarp', 'MeanIntensity_nowarp', 'GMMIntensity_nowarp', 'Bimodality_nowarp', ...
        'Volume_rigid', 'MeanIntensity_rigid', 'GMMIntensity_rigid', 'Bimodality_rigid'});

    fprintf('Frame %d/%d, Short Cam Done!\n', frames_to_align(ii), max(frames_to_align));

    output_file_long = fullfile(output_path, 'extract_long.csv');
    output_file_short = fullfile(output_path, 'extract_short.csv');
    writetable(vertcat(extract_long{:}), output_file_long);
    writetable(vertcat(extract_short{:}), output_file_short);
    
end

% h5create('nuc_affine.h5', '/exported_data', size(nuc_crop), 'Datatype', 'single');
% h5write('nuc_affine.h5', '/exported_data', nuc_crop_affine);

% h5create('nuc_rigid.h5', '/exported_data', size(seg_rigid_warp), 'Datatype', 'uint16');
% h5write('nuc_rigid.h5', '/exported_data', seg_rigid_warp);
% 
% h5create('long.h5', '/exported_data', size(long_crop_iso), 'Datatype', 'uint16');
% h5write('long.h5', '/exported_data', long_crop_iso);
