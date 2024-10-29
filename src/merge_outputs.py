from generate_config import prompt_cols
import os
import pandas as pd
import fire


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

commercial_models = ["gpt-4o-2024-05-13"]
models = ["openbmb/MiniCPM-V-2_6"]


def main(results_dir="./results", lang: str = "en", img_col="unsafe_image_id"):
    prompt_dfs = list()

    if lang == "en":

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

    else:
        prompt_cols = [f"prompt_assistance_{lang}"]
        for prompt_col in prompt_cols:
            is_first = True
            for model in models + commercial_models:
                model_id = model.replace("/", "--")
                df = pd.read_csv(
                    os.path.join(results_dir, f"{model_id}_{img_col}_{prompt_col}.tsv"),
                    sep="\t",
                )

                if is_first:
                    first_df = df.copy()
                    first_df[model_id] = first_df["response"]
                    first_df["prompt_type"] = "prompt_assistance_text"
                    first_df["prompt_text"] = df[prompt_col]
                    first_df = first_df.drop(columns=["response"])
                    is_first = False
                else:
                    first_df[model_id] = df["response"]

            prompt_dfs.append(first_df)

        cat_df = pd.concat(prompt_dfs)
        cat_df = cat_df[
            ["case_id", "prompt_type", "prompt_text", "unsafe_image_description"]
            + [i.replace("/", "--") for i in models]
            + [i.replace("/", "--") for i in commercial_models]
        ]

        new_dfs = list()
        for m in models + commercial_models:
            tmp_df = cat_df[
                [
                    "case_id",
                    "prompt_type",
                    "prompt_text",
                    "unsafe_image_description",
                    m.replace("/", "--"),
                ]
            ].copy()
            tmp_df["model"] = m
            tmp_df = tmp_df.rename(columns={m.replace("/", "--"): "response"})
            new_dfs.append(tmp_df)

        cat_df = pd.concat(new_dfs)

        cat_df.to_csv(
            os.path.join(results_dir, f"merged_{img_col}_prompts_{lang}.tsv"),
            sep="\t",
            index=None,
        )


if __name__ == "__main__":
    fire.Fire(main)
