function extract_long_short(path_to_long, path_to_short, path_to_segmentation, ...
    align_histone_LS_path, align_LS_path, output_path, frames_to_align, numThreads, ...
    resXY_img, resZ_img, resXY_seg, resZ_seg, image_format)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% create output folder
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

long_short_struct = load(align_LS_path);

% anisotropy parameters for raw imagg
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
[long_filenames, long_filename_folders] = get_filenames(path_to_long, {image_ext}, {'._'});
[short_filenames, short_filename_folders] = get_filenames(path_to_short, {image_ext}, {'._'});
[tform_filenames, tform_filename_folders] = get_filenames(align_histone_LS_path, {'.mat'}, {'._'});

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
    if read_klb
        seg_img = readKLBstack(seg_file, numThreads);
        long_img = readKLBstack(long_file, numThreads);
        short_img = readKLBstack(short_file, numThreads);
    else
        seg_img = tiffreadVolume(seg_file);
        seg_img = permute(seg_img, [2, 1, 3]);

        long_img = tiffreadVolume(long_file);
        long_img = permute(long_img, [2, 1, 3]);

        short_img = tiffreadVolume(short_file);
        short_img = permute(short_img, [2, 1, 3]);
    end
    tform_struct = load(tform_file);

    % warp short channel
    short_img = imwarp(short_img, long_short_struct.translation_tform, ...
        'OutputView', imref3d(size(long_img)), 'interp', 'linear');

    % crop images
    [seg_crop_iso, seg_hpair, seg_wpair, seg_dpair] = isotropic_crop(seg_img, ...
        tform_struct.histone_centroid, tform_struct.crop_height, ...
        tform_struct.crop_width, tform_struct.crop_depth, resXY_seg, resZ_seg, 'nearest');
    [long_crop_iso, long_hpair, long_wpair, long_dpair] = isotropic_crop(long_img, ...
        tform_struct.long_centroid, tform_struct.crop_height, ...
        tform_struct.crop_width, tform_struct.crop_depth, resXY_img, resZ_img, 'bilinear');
    [short_crop_iso, ~, ~, ~] = isotropic_crop(short_img, ...
        tform_struct.long_centroid, tform_struct.crop_height, ...
        tform_struct.crop_width, tform_struct.crop_depth, resXY_img, resZ_img, 'bilinear');

    clear long_img short_img seg_img;

    % create segmentation (fixed) referencing struct
    seg_ref = imref3d(size(seg_crop_iso));
    seg_ref.XWorldLimits = seg_hpair - tform_struct.histone_centroid(1);
    seg_ref.YWorldLimits = seg_wpair - tform_struct.histone_centroid(2);
    seg_ref.ZWorldLimits = seg_dpair - tform_struct.histone_centroid(3);

    % create TF (moving) referencing struct
    long_ref = imref3d(size(long_crop_iso));
    long_ref.XWorldLimits = long_hpair - tform_struct.long_centroid(1);
    long_ref.YWorldLimits = long_wpair - tform_struct.long_centroid(2);
    long_ref.ZWorldLimits = long_dpair - tform_struct.long_centroid(3);

    % warp long channel
    long_crop_iso_nowarp = imwarp(long_crop_iso, long_ref, tform_identity, 'OutputView', seg_ref);
    long_crop_iso_warp = imwarp(long_crop_iso, long_ref, tform_struct.rigid_tform, 'OutputView', seg_ref);    

    % use regionprops3 to extract values in long
    stats_long_nowarp = regionprops3(seg_crop_iso, long_crop_iso_nowarp, ...
        {'Volume', 'MeanIntensity', 'VoxelValues'});
    stats_long_rigid = regionprops3(seg_crop_iso, long_crop_iso_warp, ...
        {'Volume', 'MeanIntensity', 'VoxelValues'});

    ids = (1:size(stats_long_nowarp, 1)).';
    filter_ids = ~isnan(stats_long_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_long_nowarp = stats_long_nowarp(filter_ids,:);
    stats_long_rigid = stats_long_rigid(filter_ids,:);

    extract_long{ii} = table(repmat(frames_to_align(ii), length(ids), 1), ids, ...
        stats_long_nowarp.Volume, stats_long_nowarp.MeanIntensity, ...
        stats_long_rigid.Volume, stats_long_rigid.MeanIntensity, ...
        'VariableNames', {'Frame', 'ID', 'Volume_nowarp', 'MeanIntensity_nowarp', ...
        'Volume_rigid', 'MeanIntensity_rigid'});

    fprintf('Frame %d/%d, Long Cam Done!\n', frames_to_align(ii), max(frames_to_align));

    % warp short channel
    short_crop_iso_nowarp = imwarp(short_crop_iso, long_ref, tform_identity, 'OutputView', seg_ref);
    short_crop_iso_warp = imwarp(short_crop_iso, long_ref, tform_struct.rigid_tform, 'OutputView', seg_ref);
    
    % use regionprops3 to extract values in short
    stats_short_nowarp = regionprops3(seg_crop_iso, short_crop_iso_nowarp, ...
        {'Volume', 'MeanIntensity', 'VoxelValues'});
    stats_short_rigid = regionprops3(seg_crop_iso, short_crop_iso_warp, ...
        {'Volume', 'MeanIntensity', 'VoxelValues'});

    ids = (1:size(stats_short_nowarp, 1)).';
    filter_ids = ~isnan(stats_short_nowarp.MeanIntensity);
    ids = ids(filter_ids);
    stats_short_nowarp = stats_short_nowarp(filter_ids,:);
    stats_short_rigid = stats_short_rigid(filter_ids,:);

    extract_short{ii} = table(repmat(frames_to_align(ii), length(ids), 1), ids, ...
        stats_short_nowarp.Volume, stats_short_nowarp.MeanIntensity, ...
        stats_short_rigid.Volume, stats_short_rigid.MeanIntensity, ...
        'VariableNames', {'Frame', 'ID', 'Volume_nowarp', 'MeanIntensity_nowarp', ...
        'Volume_rigid', 'MeanIntensity_rigid'});

    fprintf('Frame %d/%d, Short Cam Done!\n', frames_to_align(ii), max(frames_to_align));

    output_file_long = fullfile(output_path, 'extract_long.csv');
    output_file_short = fullfile(output_path, 'extract_short.csv');
    writetable(vertcat(extract_long{:}), output_file_long);
    writetable(vertcat(extract_short{:}), output_file_short);
end

end