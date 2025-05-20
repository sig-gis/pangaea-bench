#!/bin/bash
#SBATCH --job-name=pangaea-bench-chroma
#SBATCH --output=./logs/chroma_%a.out
#SBATCH --error=./logs/chroma_%a.err
#SBATCH --array=0-14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ewg
#SBATCH --mem=96G
#SBATCH --chdir=/ceoas/emapr/pangaea-bench

conda activate pangaea-bench

# Directory containing your bash scripts
SCRIPT_DIR="./scripts/croma"

# Get sorted list of scripts
SCRIPTS=($(ls "$SCRIPT_DIR"/*.sh | sort))

# Run the script corresponding to this task ID
bash "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"