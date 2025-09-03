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

source /opt/installed/amber16/amber.sh

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/groups/ZuckermanLab/ryuwo/ENTER/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/groups/ZuckermanLab/ryuwo/ENTER/etc/profile.d/conda.sh" ]; then
        . "/home/groups/ZuckermanLab/ryuwo/ENTER/etc/profile.d/conda.sh"
    else
        export PATH="/home/groups/ZuckermanLab/ryuwo/ENTER/bin:$PATH"
    fi
fi
unset __conda_setup
#conda activate synd_env
conda activate hamsm_env

w_fluxanl --evol --disable-bootstrap &>> west.log
