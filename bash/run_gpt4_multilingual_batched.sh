#!/bin/bash

MODEL="gpt-4o-2024-05-13"
PROMPT_SET="./data/msts_multilingual_prompts_translations.tsv"
IMG_DIR="./data/unsafe_images"

OUTPUT_DIR="./results/openai_batches/"
mkdir -p ${OUTPUT_DIR}


image_col="unsafe_image_id"
echo "Using as image column: ${image_col}"

# for lang in german spanish french chinese korean farsi russian italian hindi arabic; do
for lang in korean; do
    echo "Running for ${lang}"

    for variant in assistance intention; do
        prompt_col="prompt_${variant}_${lang}"
        echo "Using as text column: ${prompt_col}"

        python src/2_run_model.py \
            --test_set ${PROMPT_SET} \
            --model_name_or_path ${MODEL} \
            --img_dir ${IMG_DIR} \
            --img_path_col "${image_col}" \
            --prompt_col "${prompt_col}" \
            --output_dir "${OUTPUT_DIR}/${lang}" \
            --overwrite
    done
done