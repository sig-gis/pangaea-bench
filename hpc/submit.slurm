#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ewg
#SBATCH --mem=256G
#SBATCH --chdir=/ceoas/emapr/pangaea-bench/
#SBATCH --export=ALL

source ~/.bashrc
conda activate pangaea-bench

export PYTHONPATH=/ceoas/emapr/pangaea-bench/:$PYTHONPATH
export LOCAL_RANK_GPU="$2"

bash "$1"