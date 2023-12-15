clc;
clear;

align_folder = './output/230212_st6/align_long_short';

frames_to_align = 50:10:130;

% get filenames
[align_filenames, align_folders] = get_filenames(align_folder, {'.mat'}, {});

% get each filename's corresponding frame number
align_frames = get_frame_ids(align_filenames);

% output cell array
x_min_cell = cell(length(frames_to_align), 1);

for ii = 1:length(frames_to_align)

    % get ind of align file
    align_ind = find(align_frames == frames_to_align(ii));

    % skip if not found
    if isempty(align_ind)
        continue;
    end

    % get full filename
    align_file = fullfile(align_folders{align_ind}, align_filenames{align_ind});

    % load file
    load(align_file);

    % store x_min
    x_min_cell{ii} = x_min(:).';

end

x_min_array = vertcat(x_min_cell{:});
x_min_mean = mean(x_min_array, 1);

tform = transltform3d(x_min_mean(1), x_min_mean(2), 0);