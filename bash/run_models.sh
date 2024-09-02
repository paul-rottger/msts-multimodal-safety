#!/usr/bin/bash

conda activate vlm-safety
echo `whereis python`
echo "Launching script with sequential execution"

config_file=$1

echo "BASH: Starting ${MODEL}"

for config_id in {0..9}; do
    python ./src/run_model.py \
        --config_file "${config_file}" \
        --config_id ${config_id}
done

echo "BASH: Done ${MODEL}"