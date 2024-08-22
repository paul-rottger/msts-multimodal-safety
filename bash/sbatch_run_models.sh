#!/usr/bin/bash

#SBATCH --job-name=MSTS
#SBATCH --output=./logs/%A-%a.out
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --qos=gpu-short
#SBATCH --array=0

source ~/.zshrc
conda activate vlm-safety
echo `whereis python`

echo "Launching script with ID $SLURM_ARRAY_TASK_ID"

config_file=$1

# check if the file $config_file/$SLURM_ARRAY_TASK_ID.DONE exists, if so, return
if [ -f "logs/${config_file}/${SLURM_ARRAY_TASK_ID}.DONE" ]; then
    echo "File ${config_file}/${SLURM_ARRAY_TASK_ID}.DONE exists, returning"
    exit 0
fi

echo "BASH: Starting ${MODEL}"
python ./src/run_model.py \
    --config_file "${config_file}" \
    --config_id ${SLURM_ARRAY_TASK_ID}

echo "BASH: Done ${MODEL}"
touch "logs/${config_file}/${SLURM_ARRAY_TASK_ID}.DONE"