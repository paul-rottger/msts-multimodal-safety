# MSTS: A Multimodal Safety Test Suite for Vision-Language Models

This repo contains data and code for our paper "[MSTS: A Multimodal Safety Test Suite for Vision-Language Models](https://arxiv.org/abs/2501.10057)".

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

Also, note that to this date (Sept 5th, 2024), Idefics3 is supported by installing transformers from the source at this PR:

https://github.com/huggingface/transformers/pull/32473


### Downloading and preprocessing the images

Use the scripts `0_download_images.py` and `1_preprocess_images.py` to retrieve and prepare our collection of unsafe images.

### Running the experiments

To run the experiments, we use Python scripts (`src`) invoked through bash runners (`bash`). The input args to Python scripts are controlled by config JSON files (`configs`).

Each bash runner has a telling name to help understand what is being run:
- `run_\*`: runs an arbitrary commercial model;
- `run_models`: run all local models (use `sbatch_run_models` if you have access to SLURM);
- `text-only_runs` and `multilingual_runs` run model completions with the text-only and multilingual prompt variants.

## Citation Information
Please consider citing our work if you use data and/or code from this repository.
```bibtex
@misc{röttger2025mstsmultimodalsafetytest,
      title={MSTS: A Multimodal Safety Test Suite for Vision-Language Models}, 
      author={Paul Röttger and Giuseppe Attanasio and Felix Friedrich and Janis Goldzycher and Alicia Parrish and Rishabh Bhardwaj and Chiara Di Bonaventura and Roman Eng and Gaia El Khoury Geagea and Sujata Goswami and Jieun Han and Dirk Hovy and Seogyeong Jeong and Paloma Jeretič and Flor Miriam Plaza-del-Arco and Donya Rooein and Patrick Schramowski and Anastassia Shaitarova and Xudong Shen and Richard Willats and Andrea Zugarini and Bertie Vidgen},
      year={2025},
      eprint={2501.10057},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.10057}, 
}
```
