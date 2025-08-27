#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test.errout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --time=336:00:00
#SBATCH -C ssdscratch
#SBATCH --gres=disk:1024

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

python test.py
