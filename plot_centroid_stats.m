clc;
clear;

addpath('utils');

segmentation_tform_path = 'D:\Posfai_Lab\MouseData\240328_st0\Ch0_long_nanog_centers';
% segmentation_tform_path = './output/231214_stack9/cdx2_centers';

[mat_filenames, mat_filename_folders] = get_filenames(segmentation_tform_path, {'.mat'}, {});
mat_frames = get_frame_ids(mat_filenames);

% reorder
[mat_frames, order] = sort(mat_frames);
mat_filenames = mat_filenames(order);
mat_filename_folders = mat_filename_folders(order);

bg_mean = nan(length(mat_frames), 1);
bg_std = nan(length(mat_frames), 1);
centroid = nan(length(mat_frames), 3);
final_loss = nan(length(mat_frames), 1);
final_xdiff = nan(length(mat_frames), 1);
debug_slices = zeros(205, 205, length(mat_frames));
for ii = 1:length(mat_frames)
    mat_file = fullfile(mat_filename_folders{ii}, mat_filenames{ii});

    mat_struct = load(mat_file);

    bg_mean(ii) = mat_struct.img_bg_mean;
    bg_std(ii) = mat_struct.img_bg_std;
    centroid(ii,:) = mat_struct.img_centroid;
    final_loss(ii) = mat_struct.trackers.mean_loss_tracker(end);
    final_xdiff(ii) = mat_struct.trackers.xdiff_rms_tracker(end);
    debug_slices(:,:,ii) = mat_struct.debug_slice;
end

centroid_diff_mag = vecnorm(diff(centroid, 1, 1), 2, 2);

% plot(mat_frames, final_loss);

return;
output_path = './output/230521_st8/Ch2long_centers';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

centroid_manual = [1030, 700, 430];
replace_ind = 1:66;

centroid(replace_ind,:) = repmat(centroid_manual, [length(replace_ind), 1]);

for ii = 1:length(mat_frames)
    mat_file = fullfile(mat_filename_folders{ii}, mat_filenames{ii});

    mat_struct = load(mat_file);
    mat_struct.img_centroid = centroid(ii,:);

    outfile = fullfile(output_path, ['frame_', num2str(mat_frames(ii)), '.mat']);
    save(outfile, '-struct', 'mat_struct');
end
