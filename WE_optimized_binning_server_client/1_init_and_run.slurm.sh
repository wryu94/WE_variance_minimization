#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=test.errout
#SBATCH --partition=gpu,bmecondo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:0:0

source /arc/software/current/amber/amber24/pmemd24/pmemd24/amber.sh
source /home/exacloud/gscratch/ZuckermanLab/ryuwo/miniconda3/etc/profile.d/conda.sh
conda activate ryuwoproject

# Start server in background
python server.py &
sleep 5   # wait a bit for server to start

# Start WE
./init.sh
./run.sh
