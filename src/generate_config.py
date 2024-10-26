from itertools import product
import fire
import json

models = [
    "nyu-visionx/cambrian-8b",
    "internlm/internlm-xcomposer2d5-7b",
    "OpenGVLab/InternVL2-8B",
    "openbmb/MiniCPM-V-2_6",
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
    "Qwen/Qwen2-VL-7B-Instruct",
    "HuggingFaceM4/Idefics3-8B-Llama3",
]
prompt_cols = ["prompt_assistance_text", "prompt_intention_text"]
img_path_cols = ["unsafe_image_id"]
img_dirs = ["./data/unsafe_images"]
test_set = "./data/prompts_220824.csv"


def main(output_file: str = "./configs/run_models.json", config: str = "default"):

    i = 0
    runs = dict()

    if config == "default":
        for model, pcol, icol, img_dir in product(
            models, prompt_cols, img_path_cols, img_dirs
        ):
            runs[i] = {
                "model_name_or_path": model,
                "test_set": test_set,
                "img_dir": img_dir,
                "img_path_col": icol,
                "prompt_col": pcol,
                "output_dir": f"./results/en",
                "quantization": False,
                "dont_use_images": False,
            }
            i += 1
    elif config == "multilingual":
        models = ["openbmb/MiniCPM-V-2_6", "OpenGVLab/InternVL2-8B", "Qwen/Qwen2-VL-7B-Instruct"]
        langs = ["german", "spanish", "french", "chinese", "korean", "farsi", "russian", "italian", "hindi", "arabic"]

        for pcol, model, lang, icol, img_dir in product(
            prompt_cols, models, langs, img_path_cols, img_dirs
        ):
            runs[i] = {
                "model_name_or_path": model,
                "test_set": "data/MSTS_prompts_translations.tsv",
                "img_dir": img_dir,
                "img_path_col": icol,
                "prompt_col": f'{"_".join(pcol.split("_")[:2])}_{lang}',
                "output_dir": f"./results/multilingual",
                "quantization": False,
                "dont_use_images": False,
            }
            i += 1


    with open(output_file, "w") as f:
        json.dump(runs, f, indent=4)



if __name__ == "__main__":
    fire.Fire(main)
