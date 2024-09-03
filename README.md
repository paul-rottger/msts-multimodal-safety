## Getting Started

Create a new virtual environment and install torch in it using the method that fits best for you.

Experiments in this repository have been conducted using `torch==2.4.0`.

Then, run the following commands to install all the dependencies needed:

```bash
mkdir dependencies
cd dependencies && git clone https://github.com/cambrian-mllm/cambrian.git
pip install -r requirements.txt
```

### Downloading and preprocessing the images

Use the scripts `0_download_images.py` and `1_preprocess_images.py` to retrieve and prepare our collection of unsafe images.