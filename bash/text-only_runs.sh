#!/usr/bin/bash


for id in 0 1; do
    python ./src/2_run_model.py \
        --use_config_file \
        --config_file ./configs/text_only.json \
        --config_id ${id}

done