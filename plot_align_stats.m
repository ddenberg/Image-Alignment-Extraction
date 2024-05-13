clc;
clear;

addpath('utils');

% align_tform_path = 'D:\Posfai_Lab\MouseData\240328_st0\LS_histone_align';
align_tform_path = 'D:\Posfai_Lab\MouseData\240328_st0\Ch1_long_Ch2_long_align';
% align_tform_path = './output/231231_st7/Ch0LS_Ch1long_align';

[tform_filenames, tform_filename_folders] = get_filenames(align_tform_path, {'.mat'}, {});
tform_frames = get_frame_ids(tform_filenames);

% reorder
[tform_frames, order] = sort(tform_frames);
tform_filenames = tform_filenames(order);
tform_filename_folders = tform_filename_folders(order);

xdiff_trackers = cell(length(tform_frames), 1);
angle = nan(length(tform_frames), 1);
translation = nan(length(tform_frames), 3);
translation_magnitude = nan(length(tform_frames), 1);

crop_size = 226;
% crop_size = 901;

debug_maxproj = zeros(crop_size, crop_size, 2, length(tform_frames), 'single');
debug_stack = zeros(crop_size, crop_size, crop_size, 2, length(tform_frames), 'uint8');

for ii = 1:length(tform_frames)
    tform_file = fullfile(tform_filename_folders{ii}, tform_filenames{ii});

    tform_struct = load(tform_file);

    rigid_tform = tform_struct.rigid_tform;
    
    % debug_maxproj(:,:,:,ii) = tform_struct.debug_maxproj;
    % debug_center(:,:,:,ii) = tform_struct.debug_center;
    debug_maxproj(:,:,:,ii) = tform_struct.debug_maxproj;
    debug_stack(:,:,:,:,ii) = tform_struct.debug_stack;

    translation(ii,:) = rigid_tform.Translation;
    translation_magnitude(ii) = norm(rigid_tform.Translation);
    angle(ii) = acosd((trace(rigid_tform.R) - 1) / 2);
    xdiff_trackers{ii} = tform_struct.rigid_trackers_ds{1}.xdiff_rms_tracker;

%     clf;
%     imagesc3d(tform_struct.debug_slice_maxproj);
    fprintf('Frame: %d\n', tform_frames(ii));

    % if tform_frames(ii) == 58
    %     imagesc3d(tform_struct.debug_slice_center);
    % end
end

% plot(tform_frames, angle);
return;


h5create('./temp/debug_stack.h5', '/out', size(debug_stack), 'Datatype', 'uint8');
h5write('./temp/debug_stack.h5', '/out', debug_stack);
