function extract_histone(path_to_histone, path_to_segmentation, output_path, ...
    output_name, frames_to_extract, numThreads, resXY_img, resZ_img, resXY_seg, resZ_seg, image_format)

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
[histone_filenames, histone_filename_folders] = get_filenames(path_to_histone, {image_ext}, {'._'});

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
    if read_klb
        seg_img = readKLBstack(seg_file, numThreads);
        histone_img = readKLBstack(histone_file, numThreads);
    else
        seg_img = tiffreadVolume(seg_file);
        seg_img = permute(seg_img, [2, 1, 3]);

        histone_img = tiffreadVolume(histone_file);
        histone_img = permute(histone_img, [2, 1, 3]);
    end

    % create isotropic versions of each image
    seg_img_iso = isotropicSample_nearest(seg_img, resXY_seg, resZ_seg, 1);
    histone_img_iso = isotropicSample_bilinear(histone_img, resXY_img, resZ_img, 1);

    clear seg_img TF_img

    % use regionprops3 to extract values in long
    stats_histone_nowarp = regionprops3(seg_img_iso, histone_img_iso, ...
        {'Volume', 'MeanIntensity', 'VoxelValues', 'Centroid'});

    ids = (1:size(stats_histone_nowarp, 1)).';
    filter_ids = ~isnan(stats_histone_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_histone_nowarp = stats_histone_nowarp(filter_ids,:);

    extract_list{ii} = table(repmat(frames_to_extract(ii), length(ids), 1), ids, ...
        stats_histone_nowarp.Volume, stats_histone_nowarp.MeanIntensity, ...
        histone_nowarp_gmm, histone_nowarp_bm, ...
        stats_histone_nowarp.Centroid, ...
        'VariableNames', {'Frame', 'ID', ...
        'Volume_nowarp', 'MeanIntensity_nowarp', 'Centroid'});

    fprintf('Frame %d/%d, Long Cam Done!\n', frames_to_extract(ii), max(frames_to_extract));

    output_file_histone = fullfile(output_path, output_name);
    writetable(vertcat(extract_list{:}), output_file_histone);
    
end

end
