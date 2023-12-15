clc;
clear;

align_tform_path = './output/230917_st10/align_Ch0short_gata6_histone';

[tform_filenames, tform_filename_folders] = get_filenames(align_tform_path, {'.mat'}, {});
tform_frames = get_frame_ids(tform_filenames);

% reorder
[tform_frames, order] = sort(tform_frames);
tform_filenames = tform_filenames(order);
tform_filename_folders = tform_filename_folders(order);

angle = nan(length(tform_frames), 1);
translation_magnitude = nan(length(tform_frames), 1);
debug_slices = zeros(225, 225, 2, length(tform_frames));

for ii = 1:length(tform_frames)
    tform_file = fullfile(tform_filename_folders{ii}, tform_filenames{ii});

    tform_struct = load(tform_file);

    rigid_tform = tform_struct.rigid_tform;
    
    debug_slices(:,:,:,ii) = tform_struct.debug_slice_maxproj;
    translation_magnitude(ii) = norm(rigid_tform.Translation);
    angle(ii) = acosd((trace(rigid_tform.R) - 1) / 2);

    clf;
    imagesc3d(tform_struct.debug_slice_maxproj);
    fprintf('Frame: %d\n', tform_frames(ii));

    % if tform_frames(ii) == 58
    %     imagesc3d(tform_struct.debug_slice_center);
    % end
end

% plot(tform_frames, angle);

