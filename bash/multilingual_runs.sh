#!/usr/bin/bash


for id in {0..59}; do
    python ./src/run_model.py \
        --use_config_file \
        --config_file ./configs/multilingual.json \
        --config_id ${id}

done