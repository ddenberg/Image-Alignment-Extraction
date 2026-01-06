function extract_TF_xy(path_to_TF, path_to_segmentation, tform_path, output_path, ...
    output_name, frames_to_extract, numThreads, resXY_img, resZ_img, resXY_seg, resZ_seg, image_format)
% path_to_TF: path to raw images (.klb) for the transcription factor
% path_to_segmentation: path to segmentation (.klb) of histone
% tform_path: directory generated as output to align_histone_TF
% output_path: path of directory of where to save the extraction
% output_name: filename.csv
% frames_to_extract: [start_frame:end_frame] or list of frames to extract
% numThreads: 8 or 16 works well on the cluster

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% anisotropy parameters for raw image
if isempty(resXY_img)
    resXY_img = 0.208;
end
if isempty(resZ_img)
    resZ_img = 2.0;
end

% anisotropy parameters for segmentation
if isempty(resXY_seg)
    resXY_seg = 0.208;
end
if isempty(resZ_seg)
    resZ_seg = 2.0;
end

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

% get filenames in each directory (excluding .label and .tif images)
[seg_filenames, seg_filename_folders] = get_filenames(path_to_segmentation, {image_ext}, {'._'});
[TF_filenames, TF_filename_folders] = get_filenames(path_to_TF, {image_ext}, {'._'});
[tform_filenames, tform_filename_folders] = get_filenames(tform_path, {'.mat'}, {'._'});

% get each filename's corresponding frame number
seg_frames = get_frame_ids(seg_filenames);
TF_frames = get_frame_ids(TF_filenames);
tform_frames = get_frame_ids(tform_filenames);

% output lists
extract_list = cell(length(frames_to_extract), 1);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:length(frames_to_extract)

    % get nuclear, long, and short filenames
    seg_ind = find(seg_frames == frames_to_extract(ii));
    TF_ind = find(TF_frames == frames_to_extract(ii));
    tform_ind = find(tform_frames == frames_to_extract(ii));

    % skip if one of the images is not present
    if isempty(TF_ind) || isempty(seg_ind) || isempty(tform_ind)
        continue;
    end

    seg_file = fullfile(seg_filename_folders{seg_ind}, seg_filenames{seg_ind});
    TF_file = fullfile(TF_filename_folders{TF_ind}, TF_filenames{TF_ind});
    tform_file = fullfile(tform_filename_folders{tform_ind}, tform_filenames{tform_ind});

    % read in nuclear, long, and short images
    if read_klb
        seg_img = readKLBstack(seg_file, numThreads);
        TF_img = readKLBstack(TF_file, numThreads);
    else
        seg_img = tiffreadVolume(seg_file);
        seg_img = permute(seg_img, [2, 1, 3]);

        TF_img = tiffreadVolume(TF_file);
        TF_img = permute(TF_img, [2, 1, 3]);
    end
    tform_struct = load(tform_file);

    % create isotropic versions of each image
    seg_img_iso = isotropicSample_nearest(seg_img, resXY_seg, resZ_seg, 1);
    TF_img_iso = isotropicSample_bilinear(TF_img, resXY_img, resZ_img, 1);

    clear seg_img TF_img

    % create segmentation (fixed) referencing struct
    seg_ref = imref3d(size(seg_img_iso));

    % create TF (moving) referencing struct
    TF_ref = imref3d(size(TF_img_iso));

    % warp TF image
    TF_img_iso_warp = imwarp(TF_img_iso, TF_ref, tform_struct.translation_tform, 'OutputView', seg_ref);

    % use regionprops3 to extract values in image
    stats_TF_nowarp = regionprops3(seg_img_iso, TF_img_iso, ...
        {'Volume', 'MeanIntensity', 'VoxelValues'});
    stats_TF_warp = regionprops3(seg_img_iso, TF_img_iso_warp, ...
        {'Volume', 'MeanIntensity', 'VoxelValues'});

    ids = (1:size(stats_TF_nowarp, 1)).';
    filter_ids = ~isnan(stats_TF_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_TF_nowarp = stats_TF_nowarp(filter_ids,:);
    stats_TF_warp = stats_TF_warp(filter_ids,:);

    extract_list{ii} = table(repmat(frames_to_extract(ii), length(ids), 1), ids, ...
        stats_TF_nowarp.Volume, stats_TF_nowarp.MeanIntensity, ...
        stats_TF_warp.Volume, stats_TF_warp.MeanIntensity, ...
        'VariableNames', {'Frame', 'ID', ...
        'Volume_nowarp', 'MeanIntensity_nowarp', ...
        'Volume_warp', 'MeanIntensity_warp'});

    fprintf('Frame %d/%d, Done!\n', frames_to_extract(ii), max(frames_to_extract));

    output_file_TF = fullfile(output_path, output_name);
    writetable(vertcat(extract_list{:}), output_file_TF);
    
end

end