import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from tqdm import tqdm

from .base import BaseHelper


class InternLMXComposerHelper(BaseHelper):
    def __init__(self, model_name_or_path: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        self.model.tokenizer = self.tokenizer

    @torch.inference_mode()
    def _forward(self, prompt, image_path, **generation_kwargs):
        """Process a single image and prompt."""
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            response, _ = self.model.chat(
                self.tokenizer, prompt, [image_path], **generation_kwargs
            )
        return response
