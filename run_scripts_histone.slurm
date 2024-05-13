#!/bin/bash

#SBATCH --job-name=hist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:59:00
#SBATCH -A molbio

module purge
module load matlab/R2023a

## matlab -nodisplay -nosplash -r "extract_histone('/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone', '/scratch/gpfs/ddenberg/230101_st8/segmentation', '/scratch/gpfs/ddenberg/230101_st8/extraction', 'extract_histone.csv', [0:104], 16)"

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/231231/231231_st7/Ch1long_histone_maxproj.h5', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch1long_histone', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch1long_histone_centers', [0:160], 16)"
## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog_maxproj.h5', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0long_nanog_centers', [0:160], 16)"
## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6_maxproj.h5', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6_centers', [0:160], 16)"
matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/231231/231231_st7/Ch2long_sox2_maxproj.h5', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch2long_sox2', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch2long_sox2_centers', [0:160], 16)"


## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone_maxproj.h5', '/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone', '/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone_centers', [0:104], 16)"
