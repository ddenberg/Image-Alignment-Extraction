clc;
clear;

addpath('loss_functions');
addpath('MA-ES');
addpath('PC_IoU');

path_to_nuc_cam = 'D:/Posfai_Lab/MouseData/230101_st19_extract/histone';
path_to_segmentation = 'D:/Posfai_Lab/MouseData/230101_st19_extract/segmentation';
% path_to_nuc_cam = '/media/david/Seagate_Exp/Posfai_Lab/MouseData/230101_st19_extract/histone';
% path_to_segmentation = '/media/david/Seagate_Exp/Posfai_Lab/MouseData/230101_st19_extract/segmentation';

% path_to_nuc_cam = '/scratch/gpfs/ddenberg/230320_st1/histone';
% path_to_long_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/long_nanog';
% path_to_short_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/short_gata6';
% path_to_nuc_cam = '/scratch/gpfs/ddenberg/230101_st19_extract/histone';

output_path = './output/230101_st19/align_histone_sequence';

start_frame = 2;
end_frame = 3;
frame_pairs = [start_frame:end_frame-1;
               start_frame+1:end_frame].';
numThreads = 6;

% crop box for increasing performance
% hpair = [650, 650+825];
% vpair = [270, 270+735];
% zpair = [5, 72];
% hpair = [580, 580+750];
% vpair = [320, 320+750];
% zpair = [35, 115];

% Stack 19
% hpair = [630, 630+870];
% vpair = [260, 260+840];
% zpair = [4, 90];
crop_h = 900;
crop_v = 900;
crop_z = 90;

% downsample factor (list of values for registration steps)
downsample_factor = [0.25];
sigma_init_rigid = [1e-2];
max_gen_init = [100];
population_size_init = [10];
tol = 1e-5;

% anisotropy parameters
resXY = 0.208;
resZ = 2.0;

% get filenames in each directory (excluding .label and .tif images)
[nuc_cam_filenames, nuc_cam_filename_folders] = get_filenames(path_to_nuc_cam, {'klb'}, {});
[seg_filenames, seg_filename_folders] = get_filenames(path_to_segmentation, {'klb'}, {});

% get each filename's corresponding frame number
nuc_cam_frames = get_frame_ids(nuc_cam_filenames);
seg_frames = get_frame_ids(seg_filenames);

% loop through each pair of frames to align 
% (skip frames where either long/short files aren't present)
for ii = 1:size(frame_pairs, 1)

    % get nuclear, long, and short filenames
    nuc_ind1 = find(nuc_cam_frames == frame_pairs(ii,1));
    nuc_ind2 = find(nuc_cam_frames == frame_pairs(ii,2));
    seg_ind1 = find(seg_frames == frame_pairs(ii,1));
    seg_ind2 = find(seg_frames == frame_pairs(ii,2));

    % skip if one of the images is not present
    if isempty(nuc_ind1) || isempty(nuc_ind2) || isempty(seg_ind1) || isempty(seg_ind2)
        continue;
    end

    nuc_file1 = fullfile(nuc_cam_filename_folders{nuc_ind1}, nuc_cam_filenames{nuc_ind1});
    nuc_file2 = fullfile(nuc_cam_filename_folders{nuc_ind2}, nuc_cam_filenames{nuc_ind2});
    seg_file1 = fullfile(seg_filename_folders{seg_ind1}, seg_filenames{seg_ind1});
    seg_file2 = fullfile(seg_filename_folders{seg_ind2}, seg_filenames{seg_ind2});

    % read in segmentation
    seg1 = readKLBstack(seg_file1, numThreads);
    seg2 = readKLBstack(seg_file2, numThreads);

    [seg_DOF_min, seg1_center, seg2_center, seg_E_min, seg_MAES_state, seg_MAES_trackers] = ...
        register_segmentation(seg1, seg2, resXY, resZ, 0.25, 2e4, 1e-4);

    clear seg1 seg2;

    % read in nuclear images
    nuc_raw1 = readKLBstack(nuc_file1, numThreads);

    hpair1 = round([seg1_center(1) - crop_h / 2, seg1_center(1) + crop_h / 2]);
    hpair1 = min(max(hpair1, 1), size(nuc_raw1, 1));

    vpair1 = round([seg1_center(2) - crop_v / 2, seg1_center(2) + crop_v / 2]);
    vpair1 = min(max(vpair1, 1), size(nuc_raw1, 2));

    zpair1 = round([seg1_center(3) * resXY / resZ - crop_z / 2, seg1_center(3) * resXY / resZ + crop_z / 2]);
    zpair1 = min(max(zpair1, 1), size(nuc_raw1, 3));

    nuc_crop1 = nuc_raw1(hpair1(1):hpair1(2), vpair1(1):vpair1(2), zpair1(1):zpair1(2));

    clear nuc_raw1;
    
    nuc_raw2 = readKLBstack(nuc_file2, numThreads);

    hpair2 = round([seg2_center(1) - crop_h / 2, seg2_center(1) + crop_h / 2]);
    hpair2 = min(max(hpair2, 1), size(nuc_raw2, 1));

    vpair2 = round([seg2_center(2) - crop_v / 2, seg2_center(2) + crop_v / 2]);
    vpair2 = min(max(vpair2, 1), size(nuc_raw2, 2));

    zpair2 = round([seg2_center(3) * resXY / resZ - crop_z / 2, seg2_center(3) * resXY / resZ + crop_z / 2]);
    zpair2 = min(max(zpair2, 1), size(nuc_raw2, 3));

    nuc_crop2 = nuc_raw2(hpair2(1):hpair2(2), vpair2(1):vpair2(2), zpair2(1):zpair2(2));

    clear nuc_raw2;

    % convert to float32
    nuc_crop1 = single(nuc_crop1);
    nuc_crop2 = single(nuc_crop2);

    % get z-scores for long and short images
    nuc_crop1 = zscore(nuc_crop1, 0, 'all');
    nuc_crop2 = zscore(nuc_crop2, 0, 'all');

    % loop over downsample stages and use previous registration as initialization for the next
    rigid_trackers_ds = cell(length(downsample_factor), 1);
    rigid_MAES_state_ds = cell(length(downsample_factor), 1);
    rigid_x_min_ds = cell(length(downsample_factor), 1);
    length_scale = mean([crop_h, crop_v, crop_z * resZ / resXY]) / 4;
    rigid_tform_ds = cell(length(downsample_factor), 1);
    for jj = 1:length(downsample_factor)
        nuc1_ds = isotropicSample_bilinear(nuc_crop1, resXY, resZ, downsample_factor(jj));
        nuc2_ds = isotropicSample_bilinear(nuc_crop2, resXY, resZ, downsample_factor(jj));

        % initialize parameters for optimization
        sigma = sigma_init_rigid(jj);
        max_gen = max_gen_init(jj);
        population_size = population_size_init(jj);

        rigid_fun_h = @(x) rigid_loss(nuc1_ds, nuc2_ds, x, length_scale, ...
                                      seg1_center, hpair1, vpair1, zpair1, ...
                                      seg2_center, hpair2, vpair2, zpair2, ...
                                      resXY, resZ);

        % do down sampled registration first
        if jj == 1
            x_init = seg_DOF_min;
            x_init(1:3) = x_init(1:3) / length_scale;
        else
            x_init = x_min_ds_;
        end        

        MAES_state_ds_ = MAES_initialize(x_init, sigma, max_gen, tol, population_size);

        tic;
        [MAES_state_ds_, x_min_ds_, f_min_, trackers_ds_] = MAES_run(MAES_state_ds_, rigid_fun_h, true);
        toc;

        rigid_trackers_ds{jj} = trackers_ds_;
        rigid_MAES_state_ds{jj} = MAES_state_ds_;
        rigid_x_min_ds{jj} = x_min_ds_;
        rigid_tform_ds{jj} = rigidtform3d(x_min_ds_(4:6).' * 180, x_min_ds_(1:3).' * length_scale);

    end
    rigid_x_min = rigid_x_min_ds{end};

    rigid_tform = rigidtform3d(rigid_x_min(4:6).' * 180, rigid_x_min(1:3).' * length_scale);

    save(fullfile(output_path, ['tform_frames_', num2str(frame_pairs(ii,1)), ...
        '_', num2str(frame_pairs(ii,2))]), 'rigid_x_min', 'rigid_MAES_state_ds', ...
        'rigid_trackers_ds', 'rigid_x_min_ds', 'rigid_tform', 'length_scale', 'rigid_tform_ds', ...
        'hpair1', 'vpair1', 'zpair1', 'hpair2', 'vpair2', 'zpair2', 'seg1_center', 'seg2_center');
end

% nuc1_ds_rigid = imwarp(nuc1_ds, tform_rigid, 'OutputView', imref3d(size(nuc1_ds)));

% h5create('temp/nuc1_ds_unwarp.h5', '/exported_data', size(nuc1_ds), 'Datatype', 'single');
% h5write('temp/nuc1_ds_unwarp.h5', '/exported_data', nuc1_ds);

% h5create('temp/nuc1_ds_rigid.h5', '/exported_data', size(nuc1_ds_rigid), 'Datatype', 'single');
% h5write('temp/nuc1_ds_rigid.h5', '/exported_data', nuc1_ds_rigid);

% h5create('temp/nuc2_ds_unwarp.h5', '/exported_data', size(nuc2_ds), 'Datatype', 'single');
% h5write('temp/nuc2_ds_unwarp.h5', '/exported_data', nuc2_ds);
