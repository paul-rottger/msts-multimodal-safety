#!/usr/bin/bash

echo `whereis python`
echo "Launching script with sequential execution"

set -e
config_file=$1

echo "BASH: Starting ${MODEL}"

for config_id in {0..9}; do
    python ./src/run_model.py \
        --config_file "${config_file}" \
        --config_id ${config_id} \
        --use_config_file
done

echo "BASH: Done ${MODEL}"