#!/bin/bash

BASE_PATH="/ceoas/emapr/pangaea-bench/scripts"

if [ -z "$1" ]; then
  echo "Usage: $0 <folder_name>"
  exit 1
fi
num_gpus="$2"
folder_name="$1"
SCRIPT_DIR="${BASE_PATH}/${folder_name}"

if [ ! -d "$SCRIPT_DIR" ]; then
  echo "Error: Folder '${SCRIPT_DIR}' does not exist."
  exit 1
fi

index=0
for script in "$SCRIPT_DIR"/*.sh; do
  name=$(basename "$script" .sh)
  gpu_id=$((index % $num_gpus))

  sbatch \
    --job-name="pangaea-bench-${folder_name}-${name}" \
    --output="./logs/${name}.out" \
    --error="./logs/${name}.err" \
    --export=ALL,CUDA_VISIBLE_DEVICES=$gpu_id \
    # alternative gpu id assignment
    /ceoas/emapr/pangaea-bench/hpc/submit.slurm "$script" "${gpu_id}"

  index=$((index + 1))
done
