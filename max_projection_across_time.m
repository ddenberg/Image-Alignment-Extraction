% clc;
% clear;
function max_projection_across_time(path_to_images, output, frames_to_extract, numThreads)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

% path_to_images = '/scratch/gpfs/ddenberg/231214_stack9/histone';
% path_to_images = '/scratch/gpfs/ddenberg/231214_stack9/cdx2';

% output = '/scratch/gpfs/ddenberg/231214_stack9/histone_maxproj.h5';
% output = './output/230917_st10_histone.h5';

% frames_to_extract = 0:120;
% numThreads = 16;

% get filenames in each directory (excluding .label and .tif images)
[histone_filenames, histone_filename_folders] = get_filenames(path_to_images, {'klb'}, {});

% get each filename's corresponding frame number
histone_frames = get_frame_ids(histone_filenames);

for ii = 1:length(frames_to_extract)

    % get nuclear, long, and short filenames
    histone_ind = find(histone_frames == frames_to_extract(ii));

    % skip if one of the images is not present
    if isempty(histone_ind)
        continue;
    end

    histone_file = fullfile(histone_filename_folders{histone_ind}, histone_filenames{histone_ind});

    histone_raw = readKLBstack(histone_file, numThreads);

    if ~exist('maxproj', 'var')
        maxproj = zeros(size(histone_raw), 'uint16');
    end

    maxproj = max(histone_raw, maxproj);

    fprintf('%d/%d\n', ii, length(frames_to_extract));
    
end

h5create(output, '/maxproj', size(maxproj), 'Datatype', 'uint16');
h5write(output, '/maxproj', maxproj);

end