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