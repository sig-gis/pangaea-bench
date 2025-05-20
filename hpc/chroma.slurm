#!/bin/bash
#SBATCH --job-name=batch_jobs
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-14                   # 15 jobs: index 0 to 14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=general

# Directory containing your bash scripts
SCRIPT_DIR="./scripts/croma"

# Get sorted list of scripts
SCRIPTS=($(ls "$SCRIPT_DIR"/*.sh | sort))

# Run the script corresponding to this task ID
bash "${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"