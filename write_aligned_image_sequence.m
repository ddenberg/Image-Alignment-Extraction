clc;
clear;

addpath('utils');
addpath('loss_functions');
addpath('MA-ES');

path_to_nuc_cam = 'D:/Posfai_Lab/MouseData/230101_st19_extract/histone';
% path_to_nuc_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/histone';

tform_path = './output/230101_st19/align_histone_sequence';

start_frame = 0;
end_frame = 5;

numThreads = 4;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[nuc_cam_filenames, nuc_cam_filename_folders] = get_filenames(path_to_nuc_cam, {'klb'}, {});
[tform_filenames, tform_filename_folders] = get_filenames(tform_path, {'.mat'}, {});

% get each filename's corresponding frame number
nuc_cam_frames = get_frame_ids(nuc_cam_filenames);
tform_frames = get_frame_ids(tform_filenames);

% sort 'tform_frames' by first column
[tform_frames, order] = sortrows(tform_frames, 1);
tform_filenames = tform_filenames(order);
tform_filename_folders = tform_filename_folders(order);

% confirm that there are no gaps
if any(diff(tform_frames(:,1), 1, 1) ~= 1)
    error('There is a frame pair missing.');
end

start_frame = max(start_frame, min(tform_frames, [], 'all'));
end_frame = min(end_frame, max(tform_frames, [], 'all'));
frame_pairs = [start_frame:end_frame-1;
               start_frame+1:end_frame].';

% create tform array which warps the image at time t to time 'start_frame'
global_tform_cell = cell(size(frame_pairs, 1), 1);
tform_struct_cell = cell(size(frame_pairs, 1), 1);

global_centroid = zeros(1, 3);
global_hpair = zeros(1, 2);
global_vpair = zeros(1, 2);
global_zpair = zeros(1, 2);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:size(frame_pairs, 1)

    [~, tform_ind] = ismember(frame_pairs(ii,:), tform_frames, 'rows');
    
    if ~tform_ind
        error(['Missing tform file for frame pair: [', ...
            num2str(frame_pairs(ii,1)), ', ', num2str(frame_pairs(ii,2)), ']']);
    end

    tform_file = fullfile(tform_filename_folders{tform_ind}, tform_filenames{tform_ind});

    % read tform
    tform_struct = load(tform_file);
    tform_struct_cell{ii} = tform_struct;


    if ii == 1
        global_tform_cell{ii} = tform_struct.rigid_tform;

        global_centroid = tform_struct.seg1_center;
        global_hpair = tform_struct.hpair1;
        global_vpair = tform_struct.vpair1;
        global_zpair = tform_struct.zpair1;
    else
        R = global_tform_cell{ii-1}.R * tform_struct.rigid_tform.R;
        Translation = (global_tform_cell{ii-1}.R * tform_struct.rigid_tform.Translation.').' + ...
            global_tform_cell{ii-1}.Translation;

%         R = tform_struct.rigid_tform.R * global_tform_cell{ii-1}.R;
%         Translation = (tform_struct.rigid_tform.R * global_tform_cell{ii-1}.Translation.').' + ...
%             tform_struct.rigid_tform.Translation;

        global_tform_cell{ii} = rigidtform3d(R, Translation);

%         global_centroid = (tform_struct.seg1_center + (ii - 1) * global_centroid) / ii;
%         global_hpair = (tform_struct.hpair1 + (ii - 1) * global_hpair) / ii;
%         global_vpair = (tform_struct.vpair1 + (ii - 1) * global_vpair) / ii;
%         global_zpair = (tform_struct.zpair1 + (ii - 1) * global_zpair) / ii;
    end

end

global_size = [200, 200, 200];
output_array = zeros([global_size, length(unique(frame_pairs))], 'uint16');

% create global (fixed) referencing struct
fixed_ref = imref3d(global_size);
fixed_ref.XWorldLimits = global_hpair - global_centroid(1);
fixed_ref.YWorldLimits = global_vpair - global_centroid(2);
fixed_ref.ZWorldLimits = (global_zpair - 1) * resZ / resXY + 1 - global_centroid(3);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:size(frame_pairs, 1)

    % get nuclear, long, and short filenames
    nuc_ind1 = find(nuc_cam_frames == frame_pairs(ii,1));
    nuc_ind2 = find(nuc_cam_frames == frame_pairs(ii,2));

    % skip if one of the images is not present
    if isempty(nuc_ind1) || isempty(nuc_ind2)
        continue;
    end

    nuc_file1 = fullfile(nuc_cam_filename_folders{nuc_ind1}, nuc_cam_filenames{nuc_ind1});
    nuc_file2 = fullfile(nuc_cam_filename_folders{nuc_ind2}, nuc_cam_filenames{nuc_ind2});

    tform_struct = tform_struct_cell{ii};

    % read in nuclear images
    % crop seg, long, and short images
    % and make isotropic
    if ii == 1
        nuc_raw1 = readKLBstack(nuc_file1, numThreads);

        nuc_crop1 = nuc_raw1(tform_struct.hpair1(1):tform_struct.hpair1(2), ...
                             tform_struct.vpair1(1):tform_struct.vpair1(2), ...
                             tform_struct.zpair1(1):tform_struct.zpair1(2));
        clear nuc_raw1;

        nuc_crop1 = isotropicSample_bilinear(nuc_crop1, resXY, resZ, 0.25);

        % create (moving) referencing struct
        moving_ref = imref3d(size(nuc_crop1));
        moving_ref.XWorldLimits = tform_struct.hpair1 - tform_struct.seg1_center(1);
        moving_ref.YWorldLimits = tform_struct.vpair1 - tform_struct.seg1_center(2);
        moving_ref.ZWorldLimits = (tform_struct.zpair1 - 1) * resZ / resXY + 1 - tform_struct.seg1_center(3);

        tform_identity = rigidtform3d;
        nuc_crop1_warp = imwarp(nuc_crop1, moving_ref, tform_identity, 'OutputView', fixed_ref, 'FillValues', 118);

        
    end

    nuc_raw2 = readKLBstack(nuc_file2, numThreads);

    nuc_crop2 = nuc_raw2(tform_struct.hpair2(1):tform_struct.hpair2(2), ...
                         tform_struct.vpair2(1):tform_struct.vpair2(2), ...
                         tform_struct.zpair2(1):tform_struct.zpair2(2));
    clear nuc_raw2;

    nuc_crop2 = isotropicSample_bilinear(nuc_crop2, resXY, resZ, 0.25);

    % create (moving) referencing struct
    moving_ref = imref3d(size(nuc_crop2));
    moving_ref.XWorldLimits = tform_struct.hpair2 - tform_struct.seg2_center(1);
    moving_ref.YWorldLimits = tform_struct.vpair2 - tform_struct.seg2_center(2);
    moving_ref.ZWorldLimits = (tform_struct.zpair2 - 1) * resZ / resXY + 1 - tform_struct.seg2_center(3);

    % warp nuc_crop2
    nuc_crop2_warp = imwarp(nuc_crop2, moving_ref, global_tform_cell{ii}, 'OutputView', fixed_ref, 'FillValues', 118);

    % add to output
    if ii == 1
        output_array(:,:,:,ii) = nuc_crop1_warp;
        output_array(:,:,:,ii+1) = nuc_crop2_warp;
    else
        output_array(:,:,:,ii+1) = nuc_crop2_warp;
    end

end

% nuc1_ds_rigid = imwarp(nuc1_ds, tform_rigid, 'OutputView', imref3d(size(nuc1_ds)));

% h5create('temp/nuc1_ds_unwarp.h5', '/exported_data', size(nuc1_ds), 'Datatype', 'single');
% h5write('temp/nuc1_ds_unwarp.h5', '/exported_data', nuc1_ds);

% h5create('temp/nuc1_ds_rigid.h5', '/exported_data', size(nuc1_ds_rigid), 'Datatype', 'single');
% h5write('temp/nuc1_ds_rigid.h5', '/exported_data', nuc1_ds_rigid);

% h5create('temp/nuc2_ds_unwarp.h5', '/exported_data', size(nuc2_ds), 'Datatype', 'single');
% h5write('temp/nuc2_ds_unwarp.h5', '/exported_data', nuc2_ds);
