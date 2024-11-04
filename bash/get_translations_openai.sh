#!/bin/sh

REPO=$(git rev-parse --show-toplevel)

#source $REPO/env/bin/activate

EVAL_MODEL="gpt-4o-mini-2024-07-18" #gpt-4-0125-preview	#gpt-3.5-turbo-0613 #gpt-4o-2024-05-13 #gpt-4o-2024-08-06

for LANG in "arabic" "chinese" "farsi" "french" "german" "hindi" "italian" "korean" "russian" "spanish"; do

    python $REPO/src/get_completions_openai.py \
        --gen_model $EVAL_MODEL \
        --model_temperature 0.0 \
        --model_max_tokens 512 \
        --input_path $REPO/data/translation_prompts/${LANG}_multimodal.csv \
        --input_col "translation_prompt" \
        --output_col "response_translated" \
        --model_col "translation_model" \
        --caching_path $REPO/data/cache \
        --output_path $REPO/data/translation_responses/${LANG}_multimodal.csv \
        --n_batches 10 \
        --start_batch 0 \
        --max_workers 10 \
        --n_samples 0

done