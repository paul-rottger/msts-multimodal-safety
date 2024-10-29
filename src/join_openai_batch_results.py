import json
import os
import glob
import fire
import pandas as pd


def main(input_dir, lang):
    files = glob.glob(f"{input_dir}/*.jsonl")
    rows = list()
    for f in files:
        with open(f, "r", encoding="utf8") as file:
            lines = file.readlines()
            for line in lines:
                rows.append(json.loads(line))

    out = list()
    for r in rows:
        out.append(
            {
                "case_id": r["custom_id"],
                "model": r["response"]["body"]["model"],
                "response": r["response"]["body"]["choices"][0]["message"]["content"],
            }
        )
    df = pd.DataFrame(out).sort_values("case_id").set_index("case_id")
    model = out[0]["model"]
    df.to_csv(
        f"{input_dir}/{model}_unsafe_image_id_prompt_assistance_{lang}.tsv", sep="\t"
    )


if __name__ == "__main__":
    fire.Fire(main)
