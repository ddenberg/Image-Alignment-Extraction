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

## matlab -nodisplay -nosplash -r "compute_image_centroids_maxproj('/scratch/gpfs/ddenberg/231214_stack9/cdx2_maxproj.h5', '/scratch/gpfs/ddenberg/231214_stack9/cdx2', './output/231214_stack9/cdx2_centers', [0:120], 16)"

## matlab -nodisplay -nosplash -r "align_histone_TF('/scratch/gpfs/ddenberg/231214_stack9/cdx2', '/scratch/gpfs/ddenberg/231214_stack9/histone', './output/231214_stack9/cdx2_centers', './output/231214_stack9/histone_centers', './output/231214_stack9/align_cdx2_histone', [47,53,62], 16)"

matlab -nodisplay -nosplash -r "extract_TF('/scratch/gpfs/ddenberg/230917_st10/Ch1long_sox2', '/scratch/gpfs/ddenberg/230917_st10/segmentation', './output/230917_st10/align_Ch1long_sox2_histone', './output/230917_st10/extraction', 'extract_sox2.csv', [0:120], 16)"



