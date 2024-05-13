clc;
clear;

addpath('utils');

align_folder = 'D:\Posfai_Lab\MouseData\240328_st0\LS_align';

frames_to_align = 96:132;

% get filenames
[align_filenames, align_folders] = get_filenames(align_folder, {'.mat'}, {});

% get each filename's corresponding frame number
align_frames = get_frame_ids(align_filenames);

% reorder
[align_frames, order] = sort(align_frames);
align_filenames = align_filenames(order);
align_folders = align_folders(order);

% filter frames
filter_frames = ismember(align_frames, frames_to_align);
align_frames = align_frames(filter_frames);
align_filenames = align_filenames(filter_frames);
align_folders = align_folders(filter_frames);

% output cell array
translation_cell = cell(length(align_frames), 1);

crop_size = 901;
debug_maxproj = zeros(crop_size, crop_size, 2, length(align_frames), 'single');

for ii = 1:length(align_frames)

    % get full filenames
    align_file = fullfile(align_folders{ii}, align_filenames{ii});

    % load files
    align_struct = load(align_file);

    % store x_min
    translation_cell{ii} = align_struct.translation_tform.Translation(:).';

    % store maxproj
    debug_maxproj(:,:,:,ii) = align_struct.debug_maxproj;

    fprintf('Frame %d, Done!\n', align_frames(ii));
end

translations = vertcat(translation_cell{:});

translation_mean = mean(translations, 1);
translation_mean(3) = 0;

translation_tform = transltform3d(translation_mean);

return;
h5create('./temp/debug_maxproj.h5', '/out', size(debug_maxproj), 'Datatype', 'single');
h5write('./temp/debug_maxproj.h5', '/out', debug_maxproj);