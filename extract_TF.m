% clc;
% clear;
function extract_TF(path_to_TF, path_to_segmentation, tform_path, output_path, output_name, frames_to_extract, numThreads)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% path_to_TF = 'D:/Posfai_Lab/MouseData/230101_st19_extract/long_nanog';
% path_to_segmentation = 'D:/Posfai_Lab/MouseData/230101_st19_extract/segmentation';
% path_to_TF = '/scratch/gpfs/ddenberg/210501/Ch1Long';
% path_to_segmentation = '/scratch/gpfs/ddenberg/210501/labels_uncropped';

% tform_path = './output/230101_st19/align_short_gata6_histone';

% output_path = './output/230101_st19/output_extraction';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% frames_to_extract = 0:104;
% numThreads = 16;

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
[TF_filenames, TF_filename_folders] = get_filenames(path_to_TF, {'klb'}, {});
[seg_tform_filenames, seg_tform_filename_folders] = get_filenames(tform_path, {'.mat'}, {});

% get each filename's corresponding frame number
seg_frames = get_frame_ids(seg_filenames);
TF_frames = get_frame_ids(TF_filenames);
seg_tform_frames = get_frame_ids(seg_tform_filenames);

% output lists
extract_list = cell(length(frames_to_extract), 1);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
tform_identity = rigidtform3d([0, 0, 0], [0, 0, 0]);
for ii = 1:length(frames_to_extract)

    % get nuclear, long, and short filenames
    seg_ind = find(seg_frames == frames_to_extract(ii));
    TF_ind = find(TF_frames == frames_to_extract(ii));
    seg_tform_ind = find(seg_tform_frames == frames_to_extract(ii));

    % skip if one of the images is not present
    if isempty(TF_ind) || isempty(seg_ind) || isempty(seg_tform_ind)
        continue;
    end

    seg_file = fullfile(seg_filename_folders{seg_ind}, seg_filenames{seg_ind});
    TF_file = fullfile(TF_filename_folders{TF_ind}, TF_filenames{TF_ind});
    seg_tform_file = fullfile(seg_tform_filename_folders{seg_tform_ind}, seg_tform_filenames{seg_tform_ind});

    % read in nuclear, long, and short images
    seg_raw = readKLBstack(seg_file, numThreads);
    TF_raw = readKLBstack(TF_file, numThreads);
    seg_tform_struct = load(seg_tform_file);

    % crop seg, long, and short images
    seg_crop = seg_raw(seg_tform_struct.histone_hpair(1):seg_tform_struct.histone_hpair(2), ...
                       seg_tform_struct.histone_vpair(1):seg_tform_struct.histone_vpair(2), ...
                       seg_tform_struct.histone_zpair(1):seg_tform_struct.histone_zpair(2));
    TF_crop = TF_raw(seg_tform_struct.TF_hpair(1):seg_tform_struct.TF_hpair(2), ...
                     seg_tform_struct.TF_vpair(1):seg_tform_struct.TF_vpair(2), ...
                     seg_tform_struct.TF_zpair(1):seg_tform_struct.TF_zpair(2));

    clear seg_raw TF_raw;

    % create isotropic segmentation and TF channels
    TF_crop_iso = isotropicSample_bilinear(TF_crop, resXY, resZ, 1);
    seg_crop_iso = isotropicSample_nearest(seg_crop, resXY, resZ, 1);

    % create segmentation (fixed) referencing struct
    seg_ref = imref3d(size(seg_crop_iso));
    seg_ref.XWorldLimits = seg_tform_struct.histone_hpair - seg_tform_struct.histone_centroid(1);
    seg_ref.YWorldLimits = seg_tform_struct.histone_vpair - seg_tform_struct.histone_centroid(2);
    seg_ref.ZWorldLimits = (seg_tform_struct.histone_zpair - 1) * resZ / resXY + 1 - seg_tform_struct.histone_centroid(3);

    % create TF (moving) referencing struct
    TF_ref = imref3d(size(TF_crop_iso));
    TF_ref.XWorldLimits = seg_tform_struct.TF_hpair - seg_tform_struct.TF_centroid(1);
    TF_ref.YWorldLimits = seg_tform_struct.TF_vpair - seg_tform_struct.TF_centroid(2);
    TF_ref.ZWorldLimits = (seg_tform_struct.TF_zpair - 1) * resZ / resXY + 1 - seg_tform_struct.TF_centroid(3);

    % warp long channel
    TF_crop_iso_nowarp = imwarp(TF_crop_iso, TF_ref, tform_identity, 'OutputView', seg_ref);
    TF_crop_iso_warp = imwarp(TF_crop_iso, TF_ref, seg_tform_struct.rigid_tform, 'OutputView', seg_ref);

    % use regionprops3 to extract values in long
    stats_TF_nowarp = regionprops3(seg_crop_iso, TF_crop_iso_nowarp, {'Volume', 'MeanIntensity', 'VoxelValues'});
    stats_TF_rigid = regionprops3(seg_crop_iso, TF_crop_iso_warp, {'Volume', 'MeanIntensity', 'VoxelValues'});

    ids = (1:size(stats_TF_nowarp, 1)).';
    filter_ids = ~isnan(stats_TF_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_TF_nowarp = stats_TF_nowarp(filter_ids,:);
    stats_TF_rigid = stats_TF_rigid(filter_ids,:);

    % do gmm filtering
    TF_nowarp_gmm = zeros(length(ids), 1);
    TF_rigid_gmm = zeros(length(ids), 1);

    TF_nowarp_bm = zeros(length(ids), 1);
    TF_rigid_bm = zeros(length(ids), 1);

    for jj = 1:length(ids) 

        % nowarp
        X = double(stats_TF_nowarp.VoxelValues{jj});
        [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
        TF_nowarp_gmm(jj) = mu;
        TF_nowarp_bm(jj) = bm;

        % rigid
        X = double(stats_TF_rigid.VoxelValues{jj});
        [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
        TF_rigid_gmm(jj) = mu;
        TF_rigid_bm(jj) = bm;
    end

    extract_list{ii} = table(repmat(frames_to_extract(ii), length(ids), 1), ids, ...
        stats_TF_nowarp.Volume, stats_TF_nowarp.MeanIntensity, TF_nowarp_gmm, TF_nowarp_bm, ...
        stats_TF_rigid.Volume, stats_TF_rigid.MeanIntensity, TF_rigid_gmm, TF_rigid_bm, ...
        'VariableNames', {'Frame', 'ID', ...
        'Volume_nowarp', 'MeanIntensity_nowarp', 'GMMIntensity_nowarp', 'Bimodality_nowarp', ...
        'Volume_rigid', 'MeanIntensity_rigid', 'GMMIntensity_rigid', 'Bimodality_rigid'});

    fprintf('Frame %d/%d, Done!\n', frames_to_extract(ii), max(frames_to_extract));

    output_file_TF = fullfile(output_path, output_name);
    writetable(vertcat(extract_list{:}), output_file_TF);
    
end

end