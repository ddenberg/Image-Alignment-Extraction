#!/bin/bash

#SBATCH --job-name=TF2
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

matlab -nodisplay -nosplash -r "extract_TF('/scratch/gpfs/ddenberg/230917_st10/Ch0short_gata6', '/scratch/gpfs/ddenberg/230917_st10/segmentation', './output/230917_st10/align_Ch0short_gata6_histone', './output/230917_st10/extraction', 'extract_gata6.csv', [0:140], 16)"

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/230917_st10/Ch0short_gata6_maxproj_permute.tif', '/scratch/gpfs/ddenberg/230917_st10/Ch0short_gata6', './output/230917_st10/Ch0short_gata6_centers', [0:140], 16)"
## matlab -nodisplay -nosplash -r "align_histone_TF('/scratch/gpfs/ddenberg/230917_st10/Ch0short_gata6', '/scratch/gpfs/ddenberg/230917_st10/histone', './output/230917_st10/Ch0short_gata6_centers', './output/230917_st10/histone_centers', './output/230917_st10/align_Ch0short_gata6_histone', [124], 16)"