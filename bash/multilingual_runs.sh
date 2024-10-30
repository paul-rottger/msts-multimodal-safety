#!/usr/bin/bash


for id in 4 14 24 34 44 54; do
    python ./src/2_run_model.py \
        --use_config_file \
        --config_file ./configs/multilingual.json \
        --config_id ${id} \
        --overwrite

done