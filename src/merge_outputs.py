from generate_config import prompt_cols
import os
import pandas as pd

results_dir = "./results"
lang = "en"
img_col = "unsafe_image_id"

commercial_models = [
    "gpt-4o-2024-05-13",
    "gemini-1.5-pro",
    "claude-3-5-sonnet-20240620",
]
models = [
    "nyu-visionx/cambrian-8b",
    "OpenGVLab/InternVL2-8B",
    "openbmb/MiniCPM-V-2_6",
    "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5",
    "internlm/internlm-xcomposer2d5-7b",
    "HuggingFaceM4/Idefics3-8B-Llama3",
]


def main():
    prompt_dfs = list()
    for prompt_col in prompt_cols:
        is_first = True
        for model in models + commercial_models:
            model_id = model.replace("/", "--")
            df = pd.read_csv(
                os.path.join(
                    results_dir, lang, f"{model_id}_{img_col}_{prompt_col}.tsv"
                ),
                sep="\t",
            )

            if is_first:
                first_df = df.copy()
                first_df[model_id] = first_df["response"]
                first_df["prompt_type"] = prompt_col
                first_df = first_df.drop(columns=["response"])
                is_first = False
            else:
                first_df[model_id] = df["response"]

        prompt_dfs.append(first_df)

    cat_df = pd.concat(prompt_dfs)
    cat_df.to_csv(
        os.path.join(results_dir, lang, f"merged_{img_col}_prompts.tsv"), sep="\t"
    )


if __name__ == "__main__":
    main()
