#!/bin/bash

#SBATCH --job-name=hist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=00:59:00
#SBATCH -A molbio

module purge
module load matlab/R2023a

## matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone', '/scratch/gpfs/ddenberg/230101_st8/Ch1long_histone_maxproj.h5', [0:104], 16)"
## matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog', '/scratch/gpfs/ddenberg/230101_st8/Ch0long_nanog_maxproj.h5', [0:104], 16)"
## matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/230101_st8/Ch0short_gata6', '/scratch/gpfs/ddenberg/230101_st8/Ch0short_gata6_maxproj.h5', [0:104], 16)"


## matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch0short_gata6_maxproj.h5', [0:160], 16)"
matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/231231/231231_st7/Ch2long_sox2', '/scratch/gpfs/ddenberg/231231/231231_st7/Ch2long_sox2_maxproj.h5', [30:160], 16)"