% clc;
% clear;
% function extract_histone(path_to_histone, path_to_segmentation, output_path, output_name, frames_to_extract)

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

path_to_histone = 'D:\Posfai_Lab\MouseData\230521\st8\Ch2long_sox2';
% path_to_histone = 'D:/Posfai_Lab/MouseData/230101_st19_extract/long_nanog';
% path_to_segmentation = '/scratch/gpfs/ddenberg/230101_st19_extract/segmentation';

% output_path = './output/230101_st19/output_extraction';

frames_to_extract = 0:130;
numThreads = 16;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[histone_filenames, histone_filename_folders] = get_filenames(path_to_histone, {'klb'}, {});

% get each filename's corresponding frame number
histone_frames = get_frame_ids(histone_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
maxproj = zeros(2048, 2048, length(frames_to_extract), 'uint16');

for ii = 1:length(frames_to_extract)

    % get nuclear, long, and short filenames
    histone_ind = find(histone_frames == frames_to_extract(ii));

    % skip if one of the images is not present
    if isempty(histone_ind)
        continue;
    end

    histone_file = fullfile(histone_filename_folders{histone_ind}, histone_filenames{histone_ind});

    histone_raw = readKLBstack(histone_file, numThreads);

    maxproj(:,:,ii) = max(histone_raw, [], 3);

    fprintf('%d/%d\n', ii, length(frames_to_extract));
    
end

h5create('./output/st8_maxproj.h5', '/seq', size(temp), 'Datatype', 'uint16');
h5write('./output/st8_maxproj.h5', '/seq', temp);