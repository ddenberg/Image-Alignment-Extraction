% clc;
% clear;
function extract_histone(path_to_histone, path_to_segmentation, output_path, output_name, frames_to_extract, numThreads)

addpath('loss_functions');
addpath('MA-ES');

% path_to_histone = '/scratch/gpfs/ddenberg/230101_st19_extract/histone';
% path_to_segmentation = '/scratch/gpfs/ddenberg/230101_st19_extract/segmentation';

% output_path = './output/230101_st19/output_extraction';
% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% frames_to_extract = 0:108;
% numThreads = 16;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% gmm parameters
tol_bm = 0.4; % bimodality coefficient threshold
tol_diff = 0.1; % gmm weight difference threshold

% get filenames in each directory (excluding .label and .tif images)
[seg_filenames, seg_filename_folders] = get_filenames(path_to_segmentation, {'klb'}, {});
[histone_filenames, histone_filename_folders] = get_filenames(path_to_histone, {'klb'}, {});

% get each filename's corresponding frame number
seg_frames = get_frame_ids(seg_filenames);
histone_frames = get_frame_ids(histone_filenames);

% output lists
extract_list = cell(length(frames_to_extract), 1);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:length(frames_to_extract)

    % get nuclear, long, and short filenames
    seg_ind = find(seg_frames == frames_to_extract(ii));
    histone_ind = find(histone_frames == frames_to_extract(ii));

    % skip if one of the images is not present
    if isempty(histone_ind) || isempty(seg_ind)
        continue;
    end

    seg_file = fullfile(seg_filename_folders{seg_ind}, seg_filenames{seg_ind});
    histone_file = fullfile(histone_filename_folders{histone_ind}, histone_filenames{histone_ind});

    % read in nuclear, long, and short images
    seg_raw = readKLBstack(seg_file, numThreads);
    histone_raw = readKLBstack(histone_file, numThreads);

    % use regionprops3 to extract values in long
    stats_histone_nowarp = regionprops3(seg_raw, histone_raw, {'Volume', 'MeanIntensity', 'VoxelValues'});

    ids = (1:size(stats_histone_nowarp, 1)).';
    filter_ids = ~isnan(stats_histone_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_histone_nowarp = stats_histone_nowarp(filter_ids,:);

    % do gmm filtering
    histone_nowarp_gmm = zeros(length(ids), 1);
    histone_nowarp_bm = zeros(length(ids), 1);

    for jj = 1:length(ids) 

        % nowarp
        X = double(stats_histone_nowarp.VoxelValues{jj});
        [mu, bm] = fit_gmm2_1D(X, tol_bm, tol_diff);
        histone_nowarp_gmm(jj) = mu;
        histone_nowarp_bm(jj) = bm;
    end

    extract_list{ii} = table(repmat(frames_to_extract(ii), length(ids), 1), ids, ...
        stats_histone_nowarp.Volume, stats_histone_nowarp.MeanIntensity, histone_nowarp_gmm, histone_nowarp_bm, ...
        'VariableNames', {'Frame', 'ID', ...
        'Volume_nowarp', 'MeanIntensity_nowarp', 'GMMIntensity_nowarp', 'Bimodality_nowarp'});

    fprintf('Frame %d/%d, Long Cam Done!\n', frames_to_extract(ii), max(frames_to_extract));

    output_file_histone = fullfile(output_path, output_name);
    writetable(vertcat(extract_list{:}), output_file_histone);
    
end

end
