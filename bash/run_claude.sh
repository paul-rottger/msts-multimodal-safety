#!/bin/bash

MODEL="claude-3-5-sonnet-20240620"
PROMPT_SET="./data/prompts_220824.csv"
IMG_DIR="./data/unsafe_images"
OUTPUT_DIR="./results/en"
mkdir -p ${OUTPUT_DIR}

prompt_col=$1
echo "Using as text column: ${prompt_col}"

image_col=$2
echo "Using as image column: ${image_col}"

python ./src/run_model.py \
    --test_set ${PROMPT_SET} \
    --model_name_or_path ${MODEL} \
    --img_dir ${IMG_DIR} \
    --img_path_col "${image_col}" \
    --prompt_col "${prompt_col}" \
    --output_dir ${OUTPUT_DIR}