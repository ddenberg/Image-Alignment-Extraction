function Vout = isotropicSample(V, resXY, resZ, downsampleFactor, method)
% Resample a 3D volume to isotropic resolution and optionally downsample
%
% Inputs:
%   V                - input 3D array (Nx x Ny x Nz)
%   resXY            - voxel size in x and y
%   resZ             - voxel size in z
%   downsampleFactor - scalar (<=1), e.g. 0.5 halves resolution
%   method           - 'linear' or 'nearest'
%
% Output:
%   Vout             - resampled volume

% Compute scaling factors
scaleXY = 1 * downsampleFactor;
scaleZ  = (resZ / resXY) * downsampleFactor;

new_size = [round(size(V,1) * scaleXY), ...
            round(size(V,2) * scaleXY), ...
            round(size(V,3) * scaleZ)];
% Resize
Vout = imresize3(V, new_size, method);

end