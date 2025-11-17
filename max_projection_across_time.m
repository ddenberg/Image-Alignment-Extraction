function max_projection_across_time(path_to_images, output, frames_to_extract, numThreads, image_format)
% path_to_images: Directory containing .klb stacks
% output: /path/to/filename.h5 
% frames_to_extract: [start_frame:end_frame]
% numThreads: Number of threads to use. 8 or 16 work well.

addpath('utils');

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
[img_filenames, img_filename_folders] = get_filenames(path_to_images, {image_ext}, {'._'});

% get each filename's corresponding frame number
img_frames = get_frame_ids(img_filenames);

for ii = 1:length(frames_to_extract)

    % get nuclear, long, and short filenames
    img_ind = find(img_frames == frames_to_extract(ii));

    % skip if one of the images is not present
    if isempty(img_ind)
        continue;
    end

    img_file = fullfile(img_filename_folders{img_ind}, img_filenames{img_ind});

    if read_klb
        img_raw = readKLBstack(img_file, numThreads);
    else
        img_raw = tiffreadVolume(img_file);
        img_raw = permute(img_raw, [2, 1, 3]);
    end

    if ~exist('maxproj', 'var')
        maxproj = zeros(size(img_raw), 'uint16');
    end

    maxproj = max(img_raw, maxproj);

    fprintf('Frame (%d / %d) Done!\n', frames_to_extract(ii), max(frames_to_extract, [], 'all'));
    
end

chunk_size = [min(size(maxproj, 1), 64), min(size(maxproj, 2), 64), min(size(maxproj, 3), 16)];
h5create(output, '/maxproj', size(maxproj), 'Datatype', 'uint16', 'Chunksize', chunk_size, 'Deflate', 5);
h5write(output, '/maxproj', maxproj);

end