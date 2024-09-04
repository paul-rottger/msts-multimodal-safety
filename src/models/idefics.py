import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from tqdm import tqdm
from typing import List
import os
import sys

current_folder = os.path.abspath(os.path.dirname(__file__))
dep_folder = os.path.abspath(
    os.path.join(current_folder, "../../dependencies/transformers")
)
sys.path.insert(0, os.path.join(os.getcwd(), "./dependencies/transformers"))


class IdeficsHelper:
    def __init__(self, model_name_or_path: str, device: str):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = (
            AutoModelForVision2Seq.from_pretrained(
                model_name_or_path, torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )

    def _forward(self, prompt, image_path, **generation_kwargs):
        """Process a single image and prompt."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, images=[load_image(image_path)], return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        generated_ids = self.model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_texts
