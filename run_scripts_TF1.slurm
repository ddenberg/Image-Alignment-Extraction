#!/bin/bash

#SBATCH --job-name=TF1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
##SBATCH --mail-type=end
##SBATCH --mail-user=ddenberg@princeton.edu
#SBATCH -A molbio

module purge
module load matlab/R2023a

matlab -nodisplay -nosplash -r "extract_TF('/scratch/gpfs/ddenberg/230917_st10/Ch1long_sox2', '/scratch/gpfs/ddenberg/230917_st10/segmentation', './output/230917_st10/align_Ch1long_sox2_histone', './output/230917_st10/extraction', 'extract_sox2.csv', [0:140], 16)"

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/230917_st10/Ch1long_sox2_maxproj_permute.tif', '/scratch/gpfs/ddenberg/230917_st10/Ch1long_sox2', './output/230917_st10/Ch1long_sox2_centers', [0:140], 16)"
## matlab -nodisplay -nosplash -r "align_histone_TF('/scratch/gpfs/ddenberg/230917_st10/Ch1long_sox2', '/scratch/gpfs/ddenberg/230917_st10/histone', './output/230917_st10/Ch1long_sox2_centers', './output/230917_st10/histone_centers', './output/230917_st10/align_Ch1long_sox2_histone', [48,60,64,72], 16)"