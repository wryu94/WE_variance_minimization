#!/bin/bash

#SBATCH --job-name=flux_calc
#SBATCH --output=flux_calc.errout
#SBATCH --partition=exacloud,bmecondo,gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=96:0:0

# Load AMBER and conda env
source /arc/software/current/amber/amber24/pmemd24/pmemd24/amber.sh
source /home/exacloud/gscratch/ZuckermanLab/ryuwo/miniconda3/etc/profile.d/conda.sh
conda activate variance_min

# Calculate flux from west.h5
w_fluxanl --evol --disable-bootstrap &>> west.log
