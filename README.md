# MSTS: A Multimodal Safety Test Suite for Vision-Language Models

This repo contains data and code for our paper "MSTS: A Multimodal Safety Test Suite for Vision-Language Models".

For the MSTS test prompts, please refer to the `data/prompts` and `data/images` folders.


## Repo Structure

├── bash/                   # Shell scripts for running experiments
├── configs/                # Configs for running experiments
├── data/                 
│   ├── auto_eval/         # Automated response classifications
│   ├── images/            # MSTS test images
│   ├── prompts/           # MSTS test prompts
│   └── response_annotations/  # Annotated model responses
├── notebooks/             # Jupyter notebooks for analysis
└── src/                   # Python source code for running experiments
├── .gitignore            # Git ignore rules
├── README.md             # This file
└── requirements.txt      # Python dependencies





GIUSEPPE TO UPDATE THE BELOW




## Getting Started

Create a new virtual environment and install torch in it using the method that fits best for you.

Experiments in this repository have been conducted using `torch==2.4.0`.

Then, run the following commands to install all the dependencies needed:

```bash
mkdir dependencies
cd dependencies && git clone https://github.com/cambrian-mllm/cambrian.git
pip install -r requirements.txt
```

Also, note that to this date (Sept 5th, 2024), Idefics3 is supported by installing transformers from source at this PR:

https://github.com/huggingface/transformers/pull/32473


### Downloading and preprocessing the images

Use the scripts `0_download_images.py` and `1_preprocess_images.py` to retrieve and prepare our collection of unsafe images.