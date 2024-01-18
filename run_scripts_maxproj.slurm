#!/bin/bash

#SBATCH --job-name=hist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
##SBATCH --mail-type=end
##SBATCH --mail-user=ddenberg@princeton.edu
#SBATCH -A molbio

module purge
module load matlab/R2023a

matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/231214_stack9/histone', '/scratch/gpfs/ddenberg/231214_stack9/histone_maxproj.h5', [0:120], 16)"

matlab -nodisplay -nosplash -r "max_projection_across_time('/scratch/gpfs/ddenberg/231214_stack9/cdx2', '/scratch/gpfs/ddenberg/231214_stack9/cdx2_maxproj.h5', [0:120], 16)"