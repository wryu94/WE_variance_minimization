#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.errout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --time=336:00:00
#SBATCH -C ssdscratch
#SBATCH --gres=disk:1024

# Load conda env
source /home/exacloud/gscratch/ZuckermanLab/ryuwo/miniconda3/etc/profile.d/conda.sh
conda activate variance_min

# Run script & change to joblib
python script.py
python pkl_to_joblib.py
