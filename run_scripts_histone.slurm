#!/bin/bash

#SBATCH --job-name=hist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:59:00
##SBATCH --mail-type=end
##SBATCH --mail-user=ddenberg@princeton.edu
#SBATCH -A molbio

module purge
module load matlab/R2023a

matlab -nodisplay -nosplash -r "extract_histone('/scratch/gpfs/ddenberg/230917_st10/histone', '/scratch/gpfs/ddenberg/230917_st10/segmentation', './output/230917_st10/extraction', 'extract_histone.csv', [0:140], 16)"

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('scratch/gpfs/ddenberg/230917_st10/histone_maxproj_permute.tif', '/scratch/gpfs/ddenberg/230917_st10/histone', './output/230917_st10/histone_centers', [0:140], 16)"