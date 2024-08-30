"""Inference code adapted from

https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5/blob/main/demo.ipynb
"""

from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)
from typing import List
from PIL import Image
import torch
from tqdm import tqdm

# model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"


def apply_prompt_template(prompt):
    s = (
        "<|system|>\nA chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
    )
    return s


class XGenHelper:

    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        self.model_name_or_path = model_name_or_path
        self.device = device

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.tokenizer.padding_side = "left"

        self.model.to(device).eval()

    @torch.inference_mode()
    def _forward(self, prompt, image_path, **generation_kwargs):
        """Process a single image and prompt."""

        image = Image.open(image_path)
        image_inputs = self.image_processor([image], image_aspect_ratio="anyres")
        prompt = apply_prompt_template(prompt)
        prompt_inputs = self.tokenizer([prompt], return_tensors="pt")

        input_ids = dict(pixel_values=[image_inputs.get("pixel_values")])
        input_ids |= prompt_inputs

        input_ids = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in input_ids.items()
        }

        output_ids = self.model.generate(
            **input_ids,
            image_size=[image.size],
            pad_token_id=self.tokenizer.pad_token_id,
            **generation_kwargs,
        )
        outputs = (
            self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            .split("<|end|>")[0]
            .strip()
        )

        return outputs

    def __call__(
        self,
        prompts: List[str],
        image_paths: List[str] = None,
        show_progress_bar: bool = True,
        **generation_kwargs,
    ):
        """Generate completions using local images and prompts."""
        assert len(prompts) == len(image_paths)

        completions = list()
        for idx, (prompt, image_path) in tqdm(
            enumerate(zip(prompts, image_paths)),
            desc="Item:",
            disable=not show_progress_bar,
            total=len(prompts),
        ):
            res = self._forward(prompt, image_path, **generation_kwargs)
            completions.append(res)

        return completions
