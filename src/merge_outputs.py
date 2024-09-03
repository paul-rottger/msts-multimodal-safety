from generate_config import models, prompt_cols
import os
import pandas as pd

results_dir = ["./results"]
lang = "en"
img_col = "unsafe_image_id"


def main():
    prompt_dfs = list()
    for prompt_col in prompt_cols:
        is_first = True
        for model in models:
            model_id = model.replace("/", "--")
            df = pd.read_csv(
                os.path.join(
                    results_dir, lang, f"{model_id}_{img_col}_{prompt_col}.tsv"
                ),
                sep="\t",
            )

            if is_first:
                first_df = df.copy()
                first_df[f"{model_id}-{prompt_col}"] = first_df["response"]
                first_df = first_df.drop(columns=["response"])
                is_first = False
            else:
                first_df[f"{model_id}-{prompt_col}"] = df["response"]

        prompt_dfs.append(first_df)

    cat_df = pd.concat(prompt_dfs)
    cat_df.to_csv(
        os.path.join(results_dir, lang, f"merged_{img_col}_prompts.tsv"), sep="\t"
    )


if __name__ == "__main__":
    main()
