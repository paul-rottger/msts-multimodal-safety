# MSTS: A Multimodal Safety Test Suite for Vision-Language Models

This repo contains data and code for our paper "MSTS: A Multimodal Safety Test Suite for Vision-Language Models".

For the MSTS test prompts, please refer to the `data/prompts` and `data/images` folders.


## Repo Structure

```
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
```


## Getting Started

Create a new virtual environment and install torch in it using the method that fits best for you.

Experiments in this repository have been conducted using `torch==2.4.0`.

Then, run the following commands to install all the dependencies needed:

```bash
# install all pypi requirements
pip install -r requirements.txt

# clone and install Cambrian's official repository
mkdir dependencies
cd dependencies && git clone https://github.com/cambrian-mllm/cambrian.git
```

Also, note that to this date (Sept 5th, 2024), Idefics3 is supported by installing transformers from source at this PR:

https://github.com/huggingface/transformers/pull/32473


### Downloading and preprocessing the images

Use the scripts `0_download_images.py` and `1_preprocess_images.py` to retrieve and prepare our collection of unsafe images.

### Running the experiments

To run the experiments, we use python scripts (`src`) that are invoked through bash runners (`bash`). The input args to python scripts are controlled by config json files (`configs`).

Each bash runner as a telling name to help understand what is being run:
- `run_\*`: runs an arbitrary commercial model;
- `run_models`: run all local models (use `sbatch_run_models` if you have access to SLURM);
- `text-only_runs` and `multilingual_runs` run model completions with the text-only and multilingual prompt variants.
