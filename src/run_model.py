import os
import time
from typing import List

import fire
import pandas as pd

# import torch
# from transformers import BitsAndBytesConfig
import logging
import json

from models import (
    CambrianHelper,
    XGenHelper,
    MiniCPMHelper,
    IdeficsHelper,
    InternVL2Helper,
    InternLMXComposerHelper,
    Qwen2VLHelper,
    GPT4VisionHelper,
    GeminiHelper,
)

# from simple_generation.vlm import SimpleVLMGenerator
# from minicpm import MiniCPMHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main(
    test_set: str = None,
    model_name_or_path: str = None,
    img_dir: str = None,
    img_path_col: str = None,
    prompt_col: str = None,
    output_dir: str = "./results",
    quantization: str = None,
    dont_use_images: bool = False,
    use_config_file: bool = False,
    config_file: str = None,
    config_id: str = None,
):
    if use_config_file:

        config_id = str(config_id)

        # load main params from config file using config_id as index
        with open(config_file, "r") as f:
            config = json.load(f)
        test_set = config[config_id]["test_set"]
        model_name_or_path = config[config_id]["model_name_or_path"]
        img_dir = config[config_id]["img_dir"]
        img_path_col = config[config_id]["img_path_col"]
        prompt_col = config[config_id]["prompt_col"]
        output_dir = config[config_id]["output_dir"]
        quantization = config[config_id]["quantization"]
        dont_use_images = config[config_id]["dont_use_images"]

    logger.info(f"Running model {model_name_or_path}")
    logger.info(f"Test set: {test_set}")
    logger.info(f"Image directory: {img_dir}")
    logger.info(f"Image path column: {img_path_col}")
    logger.info(f"Prompt column: {prompt_col}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quantization: {quantization}")

    logger.info(f"Are use using images? {not dont_use_images}")

    test_df = pd.read_csv(test_set, index_col="case_id")
    logger.info(f"Loaded {len(test_df)} rows.")
    logger.info(f"Columns: {list(test_df.columns)}")
    logger.info(f"Sample row: {test_df.iloc[0].to_dict()}")

    model_id = model_name_or_path.replace("/", "--")
    prompt_col_id = prompt_col.replace(" ", "-").lower()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{model_id}_{img_path_col}_{prompt_col_id}.tsv"
    )
    if os.path.exists(output_file):
        logger.info(f"Output file exists already. Skipping the run.")
        return

    ###########################
    ## Build image_paths
    ###########################
    image_paths = list()
    for name in test_df[img_path_col].tolist():
        p = ""
        if os.path.exists(os.path.join(img_dir, f"{name}.png")):
            p = os.path.join(img_dir, f"{name}.png")
        if os.path.exists(os.path.join(img_dir, f"{name}.jpg")):
            p = os.path.join(img_dir, f"{name}.jpg")

        image_paths.append(p)

    logger.info(f"Sample image path: {image_paths[0]}")
    test_df["img_path"] = image_paths
    original_df = test_df.copy()

    ###########################
    ## Some cleaning
    ###########################
    original_size = test_df.shape[0]
    test_df = test_df.loc[
        (test_df[prompt_col].notna()) & (test_df["img_path"].apply(os.path.exists))
    ]

    final_size = test_df.shape[0]
    if final_size != original_size:
        logger.info(
            f"We found and filtered out {original_size - test_df.shape[0]} incomplete rows."
        )

    ###########################
    ## Build prompts and img paths
    ###########################
    prompts = test_df[prompt_col].tolist()
    img_paths = test_df["img_path"].tolist()
    logger.info(f"Sample prompt: {prompts[0]}")

    ###########################
    ## Actual inference
    ###########################

    generation_kwargs = {
        "max_new_tokens": 512,
        "do_sample": False,
        "num_beams": 3,
    }

    if model_name_or_path == "nyu-visionx/cambrian-8b":
        logger.info(f"Running Cambrian model {model_name_or_path}")
        generation_kwargs["use_cache"] = True
        helper = CambrianHelper(model_name=model_name_or_path, device="cuda")
    elif "xgen-mm" in model_name_or_path:
        logger.info(f"Running local XGEN-MM model {model_name_or_path}")
        helper = XGenHelper(model_name_or_path=model_name_or_path, device="cuda")
    elif "MiniCPM" in model_name_or_path:
        logger.info(f"Running local MiniCPM model {model_name_or_path}")
        helper = MiniCPMHelper(model_name_or_path=model_name_or_path, device="cuda")
    elif "Idefics3" in model_name_or_path:
        logger.info(f"Running Idefics model {model_name_or_path}")
        helper = IdeficsHelper(model_name_or_path, device="cuda")
    elif model_name_or_path == "OpenGVLab/InternVL2-8B":
        logger.info(f"Running local InternVL2 model {model_name_or_path}")
        helper = InternVL2Helper(model_name_or_path=model_name_or_path, device="cuda")
    elif model_name_or_path == "internlm/internlm-xcomposer2d5-7b":
        logger.info(f"Running local InternLM-XComposer model {model_name_or_path}")
        helper = InternLMXComposerHelper(
            model_name_or_path=model_name_or_path, device="cuda"
        )
    elif model_name_or_path == "Qwen/Qwen2-VL-7B-Instruct":
        logger.info(f"Running Qwen2VL model {model_name_or_path}")
        helper = Qwen2VLHelper(model_name_or_path=model_name_or_path, device="cuda")
    elif "gpt" in model_name_or_path:
        logger.info(f"Running GPT model {model_name_or_path}")
        helper = GPT4VisionHelper(model_name=model_name_or_path)
        generation_kwargs = dict(max_new_tokens=512)
    elif "gemini" in model_name_or_path:
        logger.info(f"Running Gemini model {model_name_or_path}")
        helper = GeminiHelper(model_name=model_name_or_path)
        generation_kwargs = dict(max_new_tokens=512)
    else:
        logger.exception(f"Model {model_name_or_path} not supported.")
        raise ValueError(f"Model {model_name_or_path} not supported.")

    responses = helper(
        prompts=prompts,
        image_paths=None if dont_use_images else img_paths,
        show_progress_bar=True,
        log_output_every=10,
        **generation_kwargs,
    )

    test_df["response"] = responses
    merged_df = original_df.merge(test_df, how="left")

    # output_cols = [
    #     "Image ID",
    #     "IMAGE to create MALICIOUS prompt",
    #     prompt_col,
    #     "response",
    # ]
    # merged_df = merged_df[output_cols]
    merged_df.index = original_df.index

    merged_df.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    print(f"Elapsed {time.time() - stime} s")
